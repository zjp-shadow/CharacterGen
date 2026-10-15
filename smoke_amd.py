"""Run a real inference stage on the AMD GPU, using the bundled example images."""
import argparse
import os
from pathlib import Path
import warnings
import time

os.environ.setdefault("CHARACTERGEN_DEVICE", "directml")
os.environ.setdefault("CHARACTERGEN_MODEL_DIR", "D:/Codex/models/CharacterGen")
os.environ.setdefault("HF_HOME", "D:/Codex/models/CharacterGen/huggingface")
os.environ.setdefault("MPLCONFIGDIR", "D:/Codex/models/CharacterGen/matplotlib")
os.environ.setdefault("CHARACTERGEN_OUTPUT_DIR", "D:/Codex/outputs/CharacterGen")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from PIL import Image
from omegaconf import OmegaConf
from webui import Inference2D_API, Inference3D_API

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["2d", "3d"])
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    started = time.monotonic()
    warnings.filterwarnings("error", message=".*fall back to run on the CPU.*")
    with torch.no_grad():
        if args.stage == "3d":
            api = Inference3D_API()
            images = [Image.open(f"3D_Stage/material/examples/1/{i}.png").convert("RGBA") for i in range(1,5)]
            print(api.process_images(*images, True, 1), flush=True)
        else:
            api = Inference2D_API(**OmegaConf.load("2D_Stage/configs/infer.yaml"))
            images = api.inference(Image.open("materials/input/1.png"), 512, 768, seed=2333, timestep=args.steps)
            output = Path(os.environ["CHARACTERGEN_OUTPUT_DIR"]) / "smoke-2d"
            output.mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(images):
                image.save(output / f"view-{i}.png")
            print(f"Four views generated in {output}", flush=True)
    print(f"Completed {args.stage} in {time.monotonic() - started:.1f} seconds", flush=True)
