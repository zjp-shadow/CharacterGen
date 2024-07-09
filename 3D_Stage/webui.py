import os
import json
import tqdm
import cv2
import numpy as np
import torch, lrm
import torch.nn.functional as F
from lrm.utils.config import load_config
from datetime import datetime
import gradio as gr
from pygltflib import GLTF2
from PIL import Image
from huggingface_hub import hf_hub_download

from refine import refine

device = "cuda"

import trimesh
import pymeshlab
import numpy as np

from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "zjpshadow/CharacterGen"
all_files = list_repo_files(repo_id, revision="main")

for file in all_files:
    if os.path.exists("../" + file):
        continue
    if file.startswith("3D_Stage"):
        hf_hub_download(repo_id, file, local_dir="../")

def traverse(path, back_proj):
    mesh = trimesh.load(f"{path}/model-00.obj")
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90.0), [-1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180.0), [0, 1, 0]))

    cmesh = pymeshlab.Mesh(mesh.vertices, mesh.faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(cmesh)
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=4)
    mesh.vertices = ms.current_mesh().vertex_matrix()

    mesh.export(f'{path}/output.glb', file_type='glb')

    image = Image.open(f"{path}/{'refined_texture_kd.jpg' if back_proj else 'texture_kd.jpg'}")
    texture = np.array(image)
    vertex_colors = np.zeros((mesh.vertices.shape[0], 4), dtype=np.uint8)

    for vertex_index in range(len(mesh.visual.uv)):
        uv = mesh.visual.uv[vertex_index]
        x = int(uv[0] * (texture.shape[1] - 1))
        y = int((1 - uv[1]) * (texture.shape[0] - 1))

        color = texture[y, x, :3]
        vertex_colors[vertex_index] = [color[0], color[1], color[2], 255]
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors)

class Inference_API:

    def __init__(self):
        # Load config
        self.cfg = load_config("configs/infer.yaml", makedirs=False)
        # Load system
        print("Loading system")
        self.system = lrm.find(self.cfg.system_cls)(self.cfg.system).to(device)
        self.system.eval()

    def process_images(self, img_input0, img_input1, img_input2, img_input3, back_proj):
        meta = json.load(open("material/meta.json"))
        c2w_cond = [np.array(loc["transform_matrix"]) for loc in meta["locations"]]
        c2w_cond = torch.from_numpy(np.stack(c2w_cond, axis=0)).float()[None].to(device)
        
        # Prepare input data
        rgb_cond = []
        files = [img_input0, img_input1, img_input2, img_input3]
        new_image = []
        for file in files:
            image = np.array(file)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            new_image.append(Image.fromarray(image.astype(np.uint8)).convert("RGB"))
            rgb = cv2.resize(image, (self.cfg.data.cond_width, 
                                     self.cfg.data.cond_height)).astype(np.float32) / 255.0
            rgb_cond.append(rgb)
        assert len(rgb_cond) == 4, "Please provide 4 images"

        rgb_cond = torch.from_numpy(np.stack(rgb_cond, axis=0)).float()[None].to(device)

        # Run inference
        with torch.no_grad():
            scene_codes = self.system({"rgb_cond": rgb_cond, "c2w_cond": c2w_cond})
            exporter_output = self.system.exporter([f"{i:02d}" for i in range(rgb_cond.shape[0])], scene_codes)

        # Save output
        save_dir = os.path.join("./outputs", datetime.now().strftime("@%Y%m%d-%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)
        self.system.set_save_dir(save_dir)

        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            save_func = getattr(self.system, save_func_name)
            save_func(f"{out.save_name}", **out.params)

        if back_proj:
            refine(save_dir, new_image[1], new_image[0], new_image[3], new_image[2])

        new_obj = traverse(save_dir, back_proj)
        new_obj.export(f'{save_dir}/output.obj', file_type='obj')

        gltf = GLTF2().load(f'{save_dir}/output.glb')
        for material in gltf.materials:
            if material.pbrMetallicRoughness:
                material.pbrMetallicRoughness.baseColorFactor = [1.0, 1.0, 1.0, 100.0]
                material.pbrMetallicRoughness.metallicFactor = 0.0
                material.pbrMetallicRoughness.roughnessFactor = 1.0
        gltf.save(f'{save_dir}/output.glb')

        return save_dir, f"{save_dir}/output.obj", f"{save_dir}/output.glb"

inferapi = Inference_API()

# Define the interface
with gr.Blocks() as demo:
    gr.Markdown("# [SIGGRAPH'24] CharacterGen: Efficient 3D Character Generation from Single Images with Multi-View Pose Calibration")
    gr.Markdown("# 3D Stage: Four View Images to 3D Mesh")
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                img_input0 = gr.Image(type="pil", label="Back Image", image_mode="RGBA", width=256, height=384)
                img_input1 = gr.Image(type="pil", label="Front Image", image_mode="RGBA", width=256, height=384)
            with gr.Row():
                img_input2 = gr.Image(type="pil", label="Right Image", image_mode="RGBA", width=256, height=384)
                img_input3 = gr.Image(type="pil", label="Left Image", image_mode="RGBA", width=256, height=384)
            with gr.Row():
                gr.Examples(
                    examples=
                    [["material/examples/1/1.png",
                    "material/examples/1/2.png",
                    "material/examples/1/3.png",
                    "material/examples/1/4.png"]],
                    label="Example Images",
                    inputs=[img_input0, img_input1, img_input2, img_input3]
                )
        with gr.Column():
            with gr.Row():
                back_proj = gr.Checkbox(label="Back Projection")
                submit_button = gr.Button("Process")
            output_dir = gr.Textbox(label="Output Directory")
            with gr.Column():
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D( label="Output Model (GLB Format)", height = 768)
                    gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D( label="Output Model (OBJ Format)", height = 768)
                    gr.Markdown("Note: The model shown here is flipped. Download to get correct results.")

    submit_button.click(inferapi.process_images, inputs=[img_input0, img_input1, img_input2, img_input3, back_proj],
                        outputs=[output_dir, output_model_obj, output_model_glb])

# Run the interface
demo.launch()