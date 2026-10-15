"""Inference devices and GPU operations shared by the two stages."""
import os
import warnings
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def get_device():
    backend = os.environ.get("CHARACTERGEN_DEVICE", "auto")
    if backend in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if backend in ("auto", "directml"):
        import torch_directml
        index = int(os.environ.get("CHARACTERGEN_GPU", "0"))
        if not 0 <= index < torch_directml.device_count():
            raise RuntimeError("No DirectML GPU found. Check the AMD graphics driver.")
        warnings.filterwarnings("error", message=".*fall back to run on the CPU.*")
        return torch_directml.device(index)
    raise RuntimeError("CharacterGen requires a GPU: select directml or cuda.")


def is_directml(device):
    return torch.device(device).type == "privateuseone"


def autocast(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if torch.device(device).type == "cuda" else nullcontext()


def grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    """Bilinear sampling using DirectML gather; no CPU grid_sampler fallback."""
    if not is_directml(image.device):
        return F.grid_sample(image, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    if mode != "bilinear" or padding_mode not in ("zeros", "border"):
        raise NotImplementedError("DirectML sampler supports bilinear zeros/border sampling.")
    batch, channels, height, width = image.shape
    if grid.shape[0] != batch or grid.shape[-1] != 2:
        raise ValueError("Expected matching image/grid batches and XY grid coordinates.")
    x, y = grid[..., 0].float(), grid[..., 1].float()
    x = (x + 1) * ((width - 1) / 2) if align_corners else ((x + 1) * width - 1) / 2
    y = (y + 1) * ((height - 1) / 2) if align_corners else ((y + 1) * height - 1) / 2
    if padding_mode == "border":
        x, y = x.clamp(0, width - 1), y.clamp(0, height - 1)
    x0, y0 = x.floor(), y.floor()
    dx, dy = x - x0, y - y0
    flat = image.reshape(batch, channels, -1)
    result = torch.zeros((batch, channels, *grid.shape[1:3]), device=image.device, dtype=image.dtype)
    for xx, yy, weight in ((x0, y0, (1-dx)*(1-dy)), (x0+1, y0, dx*(1-dy)),
                           (x0, y0+1, (1-dx)*dy), (x0+1, y0+1, dx*dy)):
        valid = ((xx >= 0) & (xx < width) & (yy >= 0) & (yy < height))
        index = (yy.clamp(0, height-1).long()*width + xx.clamp(0, width-1).long())
        values = flat.gather(2, index.reshape(batch, 1, -1).expand(-1, channels, -1))
        result = result + values.reshape_as(result) * (weight * valid.to(weight.dtype)).unsqueeze(1).to(image.dtype)
    return result


def sliced_attention(query, key, value, mask=None, scale=None, chunk_size=128):
    """Bound attention workspace on 8 GB GPUs without CUDA/xformers kernels."""
    scale = scale if scale is not None else query.shape[-1] ** -0.5
    heads = []
    for head in range(0, query.shape[0], 4):
        keys = key[head:head+4].transpose(-1, -2)
        chunks = []
        for start in range(0, query.shape[1], chunk_size):
            end = min(start + chunk_size, query.shape[1])
            scores = torch.bmm(query[head:head+4, start:end], keys).float() * scale
            if mask is not None:
                hm = mask[head:head+4] if mask.shape[0] > 1 else mask
                scores += hm if hm.shape[-2] == 1 else hm[:, start:end]
            chunks.append(torch.bmm(scores.softmax(-1).to(value.dtype), value[head:head+4]))
            # DirectML queues work and retains temporaries until completion. A scalar
            # readback fences the GPU; it does not move the attention calculation to CPU.
            if is_directml(query.device):
                chunks[-1].reshape(-1)[0].cpu()
        heads.append(torch.cat(chunks, dim=1))
    return torch.cat(heads)


class DirectMLAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        ndim = hidden_states.ndim
        if ndim == 4:
            batch, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channels, -1).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        source = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if encoder_hidden_states is not None and attn.norm_cross:
            source = attn.norm_encoder_hidden_states(source)
        mask = attn.prepare_attention_mask(attention_mask, source.shape[1], source.shape[0])
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k = attn.head_to_batch_dim(attn.to_k(source))
        v = attn.head_to_batch_dim(attn.to_v(source))
        out = attn.batch_to_head_dim(sliced_attention(q, k, v, mask, attn.scale))
        out = attn.to_out[1](attn.to_out[0](out))
        if ndim == 4:
            out = out.transpose(1, 2).reshape(batch, channels, height, width)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor
