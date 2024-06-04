import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
# from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation
from transformers import pipeline

MODEL_SIZE = "large"
VIDEO_PATH = "/app/video/GoPro001.mp4"
OUTPUT_PATH = "/app/data/GoPro001.mp4"

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline(
    task="depth-estimation", 
    model=f"LiheYoung/depth-anything-{MODEL_SIZE}-hf",
    device=device
)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

i = 0
with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 画像フォーマットの変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        pil_image = Image.fromarray(frame_rgb)
        
        # 深度推定
        depth = pipe(pil_image)["depth"]
        
        # ヒートマップの適用
        arr_depth = cv2.applyColorMap(np.array(depth), cv2.COLORMAP_INFERNO)

        # 保存
        out.write(arr_depth)
        
        pbar.update(1)
        i += 1
        
        # if i > 100:
        #     break

cap.release()
out.release()
