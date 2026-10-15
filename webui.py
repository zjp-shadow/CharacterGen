import os
os.environ.setdefault("HF_HOME", "D:/Codex/models/CharacterGen/huggingface" if os.name == "nt" else os.path.expanduser("~/.cache/huggingface"))
import gradio as gr
from PIL import Image
import glob

import io
import argparse
import os
import random
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np

import torch
import torch.utils.checkpoint
from runtime import get_device, is_directml, autocast, DirectMLAttnProcessor

from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import check_min_version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision import transforms

import sys

sys.path.append("2D_Stage")
sys.path.append("3D_Stage")
from tuneavideo.models.unet_mv2d_condition import UNetMV2DConditionModel
from tuneavideo.models.unet_mv2d_ref import UNetMV2DRefModel
from tuneavideo.models.PoseGuider import PoseGuider
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.util import shifted_noise
from einops import rearrange
import PIL
from PIL import Image
from torchvision.utils import save_image
import json
import cv2

import lrm
import trimesh
from lrm.utils.config import load_config
from refine import refine
from datetime import datetime
import gradio as gr
from pygltflib import GLTF2

import onnxruntime as rt
from huggingface_hub.file_download import hf_hub_download
from rm_anime_bg.cli import get_mask, SCALE
import pymeshlab

from huggingface_hub import hf_hub_download, list_repo_files

MODEL_ROOT = os.environ.get("CHARACTERGEN_MODEL_DIR", ".")


def download_models():
    from huggingface_hub import snapshot_download
    snapshot_download("zjpshadow/CharacterGen", local_dir=MODEL_ROOT,
                      allow_patterns=["2D_Stage/models/checkpoint/pytorch_model*.bin",
                                      "2D_Stage/models/image_encoder/*", "3D_Stage/models/*"])

class rm_bg_api:

    def __init__(self, force_cpu: Optional[bool] = False):
        session_infer_path = hf_hub_download(
            repo_id="skytnt/anime-seg", filename="isnetis.onnx",
        )
        providers: list[str] = ["CPUExecutionProvider"]
        if not force_cpu and "CUDAExecutionProvider" in rt.get_available_providers():
            providers = ["CUDAExecutionProvider"]

        options = rt.SessionOptions()
        if not force_cpu and is_directml(get_device()):
            if "DmlExecutionProvider" not in rt.get_available_providers():
                raise RuntimeError("Install onnxruntime-directml for AMD background removal.")
            providers = [("DmlExecutionProvider", {"device_id": int(os.environ.get("CHARACTERGEN_GPU", "0"))})]
            options.enable_mem_pattern = False
            options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        self.session_infer = rt.InferenceSession(session_infer_path, sess_options=options, providers=providers)

    def remove_background(
        self,
        imgs: list[np.ndarray],
        alpha_min: float,
        alpha_max: float,
    ) -> list:
        process_imgs = []
        for img in imgs:
            img = np.array(img)
            # CHANGE to RGB
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            mask = get_mask(self.session_infer, img)

            mask[mask < alpha_min] = 0.0  # type: ignore
            mask[mask > alpha_max] = 1.0  # type: ignore

            img_after = (mask * img).astype(np.uint8)  # type: ignore
            mask = (mask * SCALE).astype(np.uint8)  # type: ignore
            img_after = np.concatenate([img_after, mask], axis=2, dtype=np.uint8)
            mask = mask.repeat(3, axis=2)
            process_imgs.append(Image.fromarray(img_after))
        return process_imgs

check_min_version("0.24.0")

logger = get_logger(__name__, log_level="INFO")

#7/24/2024 - Add creating a random seed if we pass -1.
def set_seed(seed):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)  # Generate a random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_bg_color(bg_color):
    if bg_color == 'white':
        bg_color = np.array([1., 1., 1.], dtype=np.float32)
    elif bg_color == 'black':
        bg_color = np.array([0., 0., 0.], dtype=np.float32)
    elif bg_color == 'gray':
        bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    elif bg_color == 'random':
        bg_color = np.random.rand(3)
    elif isinstance(bg_color, float):
        bg_color = np.array([bg_color] * 3, dtype=np.float32)
    else:
        raise NotImplementedError
    return bg_color

def process_image(image, totensor):
    if not image.mode == "RGBA":
        image = image.convert("RGBA")

    # Find non-transparent pixels
    non_transparent = np.nonzero(np.array(image)[..., 3])
    min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
    min_y, max_y = non_transparent[0].min(), non_transparent[0].max()    
    image = image.crop((min_x, min_y, max_x, max_y))

    # paste to center
    max_dim = max(image.width, image.height)
    max_height = max_dim
    max_width = int(max_dim / 3 * 2)
    new_image = Image.new("RGBA", (max_width, max_height))
    left = (max_width - image.width) // 2
    top = (max_height - image.height) // 2
    new_image.paste(image, (left, top))

    image = new_image.resize((512, 768), resample=PIL.Image.BICUBIC)
    image = np.array(image)
    image = image.astype(np.float32) / 255.
    assert image.shape[-1] == 4  # RGBA
    alpha = image[..., 3:4]
    bg_color = get_bg_color("gray")
    image = image[..., :3] * alpha + bg_color * (1 - alpha)
    # save image
    new_image = Image.fromarray((image * 255).astype(np.uint8))
    new_image.save("input.png")
    return totensor(image)

class Inference2D_API:

    def __init__(self,
            pretrained_model_path: str,
            image_encoder_path: str,
            ckpt_dir: str,
            validation: Dict,
            local_crossattn: bool = True,
            unet_from_pretrained_kwargs=None,
            unet_condition_type=None,
            use_pose_guider=False,
            use_shifted_noise=False,
            use_noise=True,
            device=None
        ):
        device = get_device() if device is None else torch.device(device)
        self.device = device
        self.validation = validation
        self.use_noise = use_noise
        self.use_shifted_noise = use_shifted_noise
        self.unet_condition_type = unet_condition_type
        image_encoder_path = os.path.join(MODEL_ROOT, "2D_Stage", image_encoder_path)
        ckpt_dir = os.path.join(MODEL_ROOT, "2D_Stage", ckpt_dir)

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        feature_extractor = CLIPImageProcessor()
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        unet = UNetMV2DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", local_crossattn=local_crossattn, config_only=is_directml(device), **unet_from_pretrained_kwargs)
        ref_unet = UNetMV2DRefModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", local_crossattn=local_crossattn, config_only=is_directml(device), **unet_from_pretrained_kwargs)
        if use_pose_guider:
            pose_guider = PoseGuider(noise_latent_channels=4)
        else:
            pose_guider = None

        unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu", mmap=True, weights_only=True)
        if use_pose_guider:
            pose_guider_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_1.bin"), map_location="cpu", mmap=True, weights_only=True)
            ref_unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_2.bin"), map_location="cpu", mmap=True, weights_only=True)
            pose_guider.load_state_dict(pose_guider_params, assign=True)
        else:
            ref_unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_1.bin"), map_location="cpu", mmap=True, weights_only=True)
        unet.load_state_dict(unet_params, assign=True)
        ref_unet.load_state_dict(ref_unet_params, assign=True)

        weight_dtype = torch.float16
        self.weight_dtype = weight_dtype

        text_encoder.to(device, dtype=weight_dtype)
        image_encoder.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        ref_unet.to(device, dtype=weight_dtype)
        unet.to(device, dtype=weight_dtype)

        vae.requires_grad_(False)
        unet.requires_grad_(False)
        ref_unet.requires_grad_(False)
        for model in (text_encoder, image_encoder, vae, ref_unet, unet):
            model.eval()

        noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
        self.validation_pipeline = TuneAVideoPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=self.tokenizer, unet=unet, ref_unet=ref_unet,feature_extractor=feature_extractor,image_encoder=image_encoder,
            scheduler=noise_scheduler
        )
        self.validation_pipeline.enable_vae_slicing()
        if is_directml(device):
            self.validation_pipeline.vae.enable_tiling()
            vae.set_attn_processor(DirectMLAttnProcessor())
        self.validation_pipeline.set_progress_bar_config(disable=True)
        self.generator = torch.Generator(device="cpu" if is_directml(device) else device)

    @torch.no_grad()
    def inference(self, input_image, val_width, val_height, 
                    use_shifted_noise=False, crop=False, seed=100, timestep=20):
        if is_directml(self.device):
            for name in ("text_encoder", "image_encoder", "vae", "ref_unet", "unet"):
                getattr(self.validation_pipeline, name).to(self.device)
        set_seed(seed)
        self.generator.manual_seed(seed if seed >= 0 else torch.initial_seed())
        totensor = transforms.ToTensor()

        metas = json.load(open("./2D_Stage/material/pose.json", "r"))
        cameras = []
        pose_images = []
        input_path = "./2D_Stage/material"
        for lm in metas:
            cameras.append(torch.tensor(np.array(lm[0]).reshape(4, 4).transpose(1,0)[:3, :4], dtype=torch.float32).reshape(-1))
            if not crop:
                pose_images.append(totensor(np.asarray(Image.open(os.path.join(input_path, lm[1])).resize(
                    (val_height, val_width), resample=PIL.Image.BICUBIC)).astype(np.float32) / 255.))
            else:
                pose_image = Image.open(os.path.join(input_path, lm[1]))
                crop_area = (128, 0, 640, 768)
                pose_images.append(totensor(np.array(pose_image.crop(crop_area)).astype(np.float32)) / 255.)
        camera_matrixs = torch.stack(cameras).unsqueeze(0).to(self.device)
        pose_imgs_in = torch.stack(pose_images).to(self.device)
        prompts = "high quality, best quality"
        prompt_ids = self.tokenizer(
            prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids[0]

        # (B*Nv, 3, H, W)
        B = 1
        weight_dtype = self.weight_dtype
        imgs_in = process_image(input_image, totensor)
        imgs_in = rearrange(imgs_in.unsqueeze(0).unsqueeze(0), "B Nv C H W -> (B Nv) C H W")
                
        with autocast(self.device, weight_dtype):
            imgs_in = imgs_in.to(self.device)
            # B*Nv images
            out = self.validation_pipeline(prompt=prompts, image=imgs_in.to(weight_dtype), generator=self.generator, 
                                        num_inference_steps=timestep,
                                        camera_matrixs=camera_matrixs.to(weight_dtype), prompt_ids=prompt_ids, 
                                        height=val_height, width=val_width, unet_condition_type=self.unet_condition_type, 
                                        pose_guider=None, pose_image=pose_imgs_in, use_noise=self.use_noise, 
                                        use_shifted_noise=use_shifted_noise, **self.validation).videos
            out = rearrange(out, "B C f H W -> (B f) C H W", f=self.validation.video_length)

        image_outputs = []
        if not torch.isfinite(out).all():
            raise RuntimeError("The generated views contain non-finite values; no images were exported.")
        for bs in range(4):
            img_buf = io.BytesIO()
            save_image(out[bs], img_buf, format='PNG')
            img_buf.seek(0)
            img = Image.open(img_buf)
            image_outputs.append(img)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_directml(self.device):
            for name in ("text_encoder", "image_encoder", "vae", "ref_unet", "unet"):
                getattr(self.validation_pipeline, name).to("cpu")
        return image_outputs 

def traverse(path, back_proj, smooth_iter):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(f"{path}/model-00.obj")
    image = Image.open(f"{path}/{'refined_texture_kd.jpg' if back_proj else 'texture_kd.jpg'}")
    out_image_path = f"{path}/{'refined_texture_kd.png' if back_proj else 'texture_kd.png'}"
    image.save(out_image_path, 'PNG')
    ms.set_texture_per_mesh(textname=f"{path}/{'refined_texture_kd.png' if back_proj else 'texture_kd.png'}")
    ms.meshing_merge_close_vertices()
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=smooth_iter)
    ms.save_current_mesh(f"{path}/temp-00.obj", save_vertex_normal=False, save_wedge_normal=False, save_vertex_color=False)

    mesh = trimesh.load(f"{path}/temp-00.obj", process=False)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90.0), [-1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180.0), [0, 1, 0]))

    mesh.export(f'{path}/output.glb', file_type='glb')

    image = Image.open(f"{path}/{'refined_texture_kd.png' if back_proj else 'texture_kd.png'}")
    texture = np.array(image)
    vertex_colors = np.zeros((mesh.vertices.shape[0], 4), dtype=np.uint8)

    for vertex_index in range(len(mesh.visual.uv)):
        uv = mesh.visual.uv[vertex_index]
        x = int(uv[0] * (texture.shape[1] - 1))
        y = int((1 - uv[1]) * (texture.shape[0] - 1))

        color = texture[y, x, :3]
        vertex_colors[vertex_index] = [color[0], color[1], color[2], 255]
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors, process=False)

class Inference3D_API:

    def __init__(self, device=None):
        self.cfg = load_config("3D_Stage/configs/infer.yaml", makedirs=False)
        print("Loading system")
        self.device = get_device() if device is None else torch.device(device)
        self.cfg.system.weights = os.path.join(MODEL_ROOT, "3D_Stage", self.cfg.system.weights)
        self.cfg.system.image_tokenizer.pretrained_model_name_or_path = \
            os.path.join(MODEL_ROOT, "3D_Stage", self.cfg.system.image_tokenizer.pretrained_model_name_or_path)
        self.cfg.system.renderer.tet_dir = self.cfg.system.renderer.tet_dir.replace("./", "./3D_Stage/")
        self.cfg.system.exporter.output_path = self.cfg.system.exporter.output_path.replace("./", "./3D_Stage/")
        if is_directml(self.device):
            self.cfg.system.image_tokenizer.enable_memory_efficient_attention = False
            self.cfg.system.backbone.enable_memory_efficient_attention = False
            self.cfg.system.image_tokenizer.enable_gradient_checkpointing = False
            self.cfg.system.backbone.gradient_checkpointing = False
            self.cfg.system.renderer.eval_chunk_size = 8192
            self.cfg.system.decoder.chunk_size = 8192
        self.system = lrm.find(self.cfg.system_cls)(self.cfg.system).to(self.device)
        if is_directml(self.device):
            for name in ("image_tokenizer", "camera_embedder", "tokenizer", "backbone", "post_processor"):
                getattr(self.system, name).half()
            for module in self.system.modules():
                if hasattr(module, "set_processor"):
                    module.set_processor(DirectMLAttnProcessor())
        self.system.eval()

    def process_images(self, img_input0, img_input1, img_input2, img_input3, back_proj, smooth_iter):
        dtype = torch.float16 if is_directml(self.device) else torch.float32
        if is_directml(self.device):
            for name in ("image_tokenizer", "camera_embedder", "tokenizer", "backbone", "post_processor"):
                getattr(self.system, name).to(self.device)
        meta = json.load(open("./3D_Stage/material/meta.json"))
        c2w_cond = [np.array(loc["transform_matrix"]) for loc in meta["locations"]]
        c2w_cond = torch.from_numpy(np.stack(c2w_cond, axis=0))[None].to(device=self.device, dtype=dtype)
        # save four images
        
        rgb_cond = []
        files = [img_input0, img_input1, img_input2, img_input3]
        new_images = []
        for file in files:
            image = np.array(file)
            image = Image.fromarray(image)
            if image.width != image.height:
                max_dim = max(image.width, image.height)
                new_image = Image.new("RGBA", (max_dim, max_dim))
                left = (max_dim - image.width) // 2
                top = (max_dim - image.height) // 2
                new_image.paste(image, (left, top))
                image = new_image
                image.save("input_3D.png")

            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)
            rgb = cv2.resize(image, (self.cfg.data.cond_width, 
                                     self.cfg.data.cond_height)).astype(np.float32) / 255.0
            new_images.append(Image.fromarray(image.astype(np.uint8)).convert("RGB"))
            rgb_cond.append(rgb)
        assert len(rgb_cond) == 4, "Please provide 4 images"

        rgb_cond = torch.from_numpy(np.stack(rgb_cond, axis=0))[None].to(device=self.device, dtype=dtype)

        with torch.no_grad():
            scene_codes = self.system({"rgb_cond": rgb_cond, "c2w_cond": c2w_cond})
            if not torch.isfinite(scene_codes).all().cpu():
                raise RuntimeError("The 3D network produced non-finite values.")
            if is_directml(self.device):
                # Host RAM holds inactive weights; all neural inference remains on the GPU.
                scene_codes = scene_codes.float()
                for name in ("image_tokenizer", "camera_embedder", "tokenizer", "backbone", "post_processor"):
                    getattr(self.system, name).to("cpu")
            exporter_output = self.system.exporter([f"{i:02d}" for i in range(rgb_cond.shape[0])], scene_codes)

        save_dir = os.path.join(os.environ.get("CHARACTERGEN_OUTPUT_DIR", "./3D_Stage/outputs"), datetime.now().strftime("@%Y%m%d-%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)
        self.system.set_save_dir(save_dir)

        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            save_func = getattr(self.system, save_func_name)
            save_func(f"{out.save_name}", **out.params)
        if back_proj:
            refine(save_dir, new_images[1], new_images[0], new_images[3], new_images[2])

        new_obj = traverse(save_dir, back_proj, smooth_iter)
        new_obj.export(f'{save_dir}/output.obj', file_type='obj')

        gltf = GLTF2().load(f'{save_dir}/output.glb')
        for material in gltf.materials:
            if material.pbrMetallicRoughness:
                material.pbrMetallicRoughness.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
                material.pbrMetallicRoughness.metallicFactor = 0.0
                material.pbrMetallicRoughness.roughnessFactor = 1.0
        gltf.save(f'{save_dir}/output.glb')

        return save_dir, f"{save_dir}/output.obj", f"{save_dir}/output.glb"

@torch.no_grad()
def main(
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./2D_Stage/configs/infer.yaml", help='Path to a config yaml file.')
    parser.add_argument("--share", action="store_true", help='Share Gradio as a public URL.')
    parser.add_argument("--skip-download", action="store_true", help='Use already downloaded weights.')
    args = parser.parse_args()

    if not args.skip_download:
        download_models()
    # Only one generation stage occupies VRAM at a time on the RX 6600.
    infer2dapi = None
    infer3dapi = None
    remove_api = None

    def gen4views(image, width, height, seed, timestep, remove_bg):
        nonlocal infer2dapi, infer3dapi, remove_api
        if image is None:
            raise gr.Error("Selecione uma imagem antes de gerar as vistas.")
        import gc
        infer3dapi = None
        gc.collect()
        if remove_bg:
            if remove_api is None:
                remove_api = rm_bg_api()
            image = remove_api.remove_background(
                imgs=[np.array(image)],
                alpha_min=0.1,
                alpha_max=0.9,
            )[0]
        remove_api = None
        gc.collect()
        if infer2dapi is None:
            infer2dapi = Inference2D_API(**OmegaConf.load(args.config))
        images = infer2dapi.inference(
            image, int(width), int(height), crop=True, seed=int(seed), timestep=int(timestep)
        )
        remove_api = rm_bg_api()
        return remove_api.remove_background(
            imgs=images, alpha_min=0.2, alpha_max=0.9)

    def gen3d(*images_and_options):
        nonlocal infer2dapi, infer3dapi, remove_api
        import gc
        infer2dapi = None
        remove_api = None
        gc.collect()
        if infer3dapi is None:
            infer3dapi = Inference3D_API()
        return infer3dapi.process_images(*images_and_options)

    with gr.Blocks() as demo:
        gr.Markdown("# [SIGGRAPH'24] CharacterGen: Efficient 3D Character Generation from Single Images with Multi-View Pose Calibration")
        with gr.Row():
            with gr.Column(variant="panel"):
                img_input = gr.Image(type="pil", label="Upload Image(without background)", image_mode="RGBA", width=768, height=512)
                gr.Examples(
                    label="Example Images",
                    examples=glob.glob("./2D_Stage/material/examples/*.png"),
                    inputs=[img_input]
                )
                with gr.Row():
                    width_input = gr.Number(label="Width", value=512, precision=0)
                    height_input = gr.Number(label="Height", value=768, precision=0)
                    seed_input = gr.Number(label="Seed", value=2333, precision=0)
                    remove_bg = gr.Checkbox(label="Remove Background (with algorithm)", value=True)
            with gr.Column(variant="panel"):
                timestep = gr.Slider(minimum=10, maximum=70, step=1, value=40, label="Timesteps")
                button1 = gr.Button(value="Generate 4 Views")
            with gr.Row():
                    img_input0 = gr.Image(type="pil", label="Back Image", image_mode="RGBA", width=256, height=384)
                    img_input1 = gr.Image(type="pil", label="Front Image", image_mode="RGBA", width=256, height=384)
            with gr.Row():
                    img_input2 = gr.Image(type="pil", label="Right Image", image_mode="RGBA", width=256, height=384)
                    img_input3 = gr.Image(type="pil", label="Left Image", image_mode="RGBA", width=256, height=384)
            with gr.Column(variant="panel"):
                smooth_iter = gr.Slider(minimum=0, maximum=10, step=1, value=5, label="Laplacian Smoothing Iterations")
                with gr.Row():
                    back_proj = gr.Checkbox(label="Back Projection")
                    button2 = gr.Button(value="Generate 3D Mesh")
                output_dir = gr.Textbox(label="Output Directory")
                with gr.Row():
                    with gr.Tab("GLB"):
                        output_model_glb = gr.Model3D( label="Output Model (GLB Format)", height=512)
                        gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")
                    with gr.Tab("OBJ"):
                        output_model_obj = gr.Model3D( label="Output Model (OBJ Format)")
                        gr.Markdown("Note: The model shown here's texture is mapped to vertex. Download to get correct results.")
        button1.click(
            fn=gen4views,
            inputs=[img_input, width_input, height_input, seed_input, timestep, remove_bg],
            outputs=[img_input2, img_input0, img_input3, img_input1], concurrency_id="generation"
        )
        button2.click(
            gen3d,
            inputs=[img_input0, img_input1, img_input2, img_input3, back_proj, smooth_iter],
            outputs=[output_dir, output_model_obj, output_model_glb], concurrency_id="generation"
        )
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", share=args.share,
        allowed_paths=[os.path.abspath(os.environ.get("CHARACTERGEN_OUTPUT_DIR", "./3D_Stage/outputs"))],
    )

if __name__ == "__main__":
    main()
