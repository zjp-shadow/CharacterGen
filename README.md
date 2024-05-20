# CharacterGen: Efficient 3D Character Generation from Single Images with Multi-View Pose Calibration

This is the official codebase of SIGGRAPH'24 (TOG) [CharacterGen](https://charactergen.github.io/).

![teaser](./materials/teaser.png)

- [x] Rendering Script of VRM model, including blender and three-js.
- [ ] Inference code for 2D generation stage.
- [ ] Inference code for 3D generation stage.

## Quick Start

### 1. Prepare environment

`pip install -r requirements.txt`


### 2. Download the weight

To be uploaded soon.

### 3. Run the script

To be finnished soon.

## Get the Anime3D Dataset

Due to the policy, we cannot redistribute the raw data of VRM format 3D character.
You can download the vroid dataset follow [PAniC-3D](https://github.com/ShuhongChen/panic3d-anime-reconstruction) instruction.
And the you can render the script with blender or three-js with our released rendering script.

### blender

### three-vrm

Much quicker than blender vrm add-on.

Install [Node.js](https://nodejs.org/) first to use the npm environment.

```
cd render_script/three-js
npm install three @pixiv/three-vrm
```

If you want to render depth-map images of vrm, you should replace three-vrm with [my version](/home/zjp/CharacterGen/render_script/three-js/src/three-vrm.js).

Fisrt, run the backend to catch the data from the frontend (default port is `17070`), remember to change the folder path.

```
pip install fastapi uvicorn aiofiles pillow numpy
python up_backend.py
```

Second, run the frontend to render the images.

```
npm run dev
```

The open the website http://localhost:5173/, it use 2 threads to render the image, which costs about 1 day.

## Our Result

| Single Input Image | 2D Multi-View Images | 3D Character |
|-------|-------|-------|
| ![](./materials/input/1.png) | ![](./materials/ours_multiview/1.png) | <img alt="threestudio" src="./materials/videos/1.gif" width="100%"> |
| ![](./materials/input/2.png) | ![](./materials/ours_multiview/2.png) | <img alt="threestudio" src="./materials/videos/2.gif" width="100%"> |
| ![](./materials/input/3.png) | ![](./materials/ours_multiview/3.png) | <img alt="threestudio" src="./materials/videos/3.gif" width="100%"> |