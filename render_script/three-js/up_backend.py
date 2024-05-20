
from fastapi import FastAPI, File, UploadFile
import aiofiles
import uvicorn, os, io
import cv2, json, math
import numpy as np
from PIL import Image
from typing import List
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "*"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vertices_list = ['leftUpperArm',
 'rightUpperArm',
 'leftLowerArm',
 'rightLowerArm',
 'leftHand',
 'rightHand',
 'leftUpperLeg',
 'rightUpperLeg',
 'leftLowerLeg',
 'rightLowerLeg',
 'leftFoot',
 'rightFoot',
 'head',
 'upperChest']

def process_depth_file(file_data):
    image = Image.open(io.BytesIO(file_data))
    if image.mode != 'RGBA':
        raise ValueError("Image is not in RGBA format")

    data = np.array(image)

    R = data[:, :, 0]
    G = data[:, :, 1]
    B = data[:, :, 2]
    A = data[:, :, 3]

    depth = (R + G / 256.0 + B / (256.0 * 256.0))

    depth[A == 0] = 5.0

    return depth

vertices_list = ['leftUpperArm',
 'rightUpperArm',
 'leftLowerArm',
 'rightLowerArm',
 'leftHand',
 'rightHand',
 'leftUpperLeg',
 'rightUpperLeg',
 'leftLowerLeg',
 'rightLowerLeg',
 'leftFoot',
 'rightFoot',
 'head',
 'hips',
 'spine']


def backproject_to_3d(base_dir):
    for fn in os.listdir(base_dir):
        if fn.endswith(".json"):
            pose_file = open(os.path.join(base_dir, fn), "r")
            pose_dict = json.load(pose_file)
            vertices = []
            joints_name = []
            for node_name in vertices_list:
                joints_name.append(node_name)
                v = pose_dict['node_array'][node_name]
                vertices.append([v["world_position"]['x'], v["world_position"]['y'], v["world_position"]['z'], 1.0])
            vertices = np.array(vertices)
            chest_v = vertices[12] * 2 / 3 + vertices[14] / 3
            vertices = np.concatenate([vertices[:13], [chest_v]], axis=0)
            vertices = vertices[[12, 13, 0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11]]
            focal_length = 256 / np.tan(40 / 2 / 180 * np.pi)
            ext = np.array(pose_dict["extrinsicMatrix"]["elements"]).reshape(4,4)
            ext = np.linalg.inv(ext)
            vertices = vertices @ ext
            camera_v = vertices[...,:3] / vertices[...,3:]
            camera_v = camera_v[...,:2] / camera_v[...,2:] * focal_length
            camera_v += 256
            camera_v /= 512
            camera_v[:,0] = 1 - camera_v[:,0]
            canvas = np.zeros((768, 768, 3), dtype=np.uint8)
            #canvas = cv2.imread(os.path.join(base_dir, fn.replace(".json", "_rgb.png")))
            draw_bodypose(canvas, camera_v)
            cv2.imwrite(os.path.join(base_dir, fn.replace(".json", "_pose.png")), cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

def draw_bodypose(canvas: np.ndarray, keypoints) -> np.ndarray:
    H, W, C = canvas.shape
    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], 
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        if k1_index-1 >= len(keypoints) or k2_index -1>= len(keypoints):
            continue
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1[0], keypoint2[0]]) * float(W)
        X = np.array([keypoint1[1], keypoint2[1]]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas

count = 0
@app.post("/upload/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    global count
    count += 1
    print(count)
    for file in files:
        object_name = file.filename.split("_")[0]
        folder = f"E:/new_render/{object_name}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_data = await file.read()
        if(file.filename.split('_')[-1].split('.')[0] == "depth"):
            depth = process_depth_file(file_data)
            cv2.imwrite(f"{folder}/{file.filename}", depth)
            new_path = f"{folder}/{file.filename}"
            numpy_path = new_path.replace(".png", ".npz")
            np.savez_compressed(numpy_path, depth)
        else:
            with open(f"{folder}/{file.filename}", "wb") as buffer:
                buffer.write(file_data)
    backproject_to_3d(folder)
            
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=17070)