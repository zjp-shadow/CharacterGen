"""Download inference weights without loading them into RAM or VRAM."""
import os
from pathlib import Path

base = Path(os.environ.get("CHARACTERGEN_DATA_DIR", "D:/Codex/models/CharacterGen"))
os.environ.setdefault("HF_HOME", str(base / "huggingface"))
from huggingface_hub import hf_hub_download, snapshot_download

if __name__ == "__main__":
    snapshot_download("zjpshadow/CharacterGen", local_dir=str(base), max_workers=3,
                      allow_patterns=["2D_Stage/models/checkpoint/pytorch_model*.bin",
                                      "2D_Stage/models/image_encoder/*", "3D_Stage/models/*"])
    snapshot_download("sd2-community/stable-diffusion-2-1", max_workers=3,
                      allow_patterns=["*/config.json", "scheduler/*", "tokenizer/*",
                                      "text_encoder/model.safetensors",
                                      "unet/diffusion_pytorch_model.safetensors",
                                      "vae/diffusion_pytorch_model.safetensors"])
    hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
    print(f"Inference weights ready in {base}")
