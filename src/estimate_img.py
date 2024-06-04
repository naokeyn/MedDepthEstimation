from transformers import pipeline
from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch

MODEL_SIZE = "large" # small or large

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline(
    task="depth-estimation", 
    model=f"LiheYoung/depth-anything-{MODEL_SIZE}-hf",
    device=device
)

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# url = "https://trip.pref.kanagawa.jp/img/spots/photos/hero/o2hPDqtaSoi16nFjKo52Z3aSSOT6MwVd8xRwydrm-1920x1080.jpg"
url = "https://www.benesse.co.jp/gtec/general/lp_a/img/case/post-11/pic01.jpg"
image = Image.open(requests.get(url, stream=True).raw)

depth = pipe(image)["depth"]

fig = plt.figure(figsize=(5, 10))

ax1 = fig.add_subplot(211)
ax1.imshow(image)

ax2 = fig.add_subplot(212)
ax2.imshow(depth, cmap="coolwarm")

plt.savefig(f"../data/output3-{MODEL_SIZE}.png", format="png", dpi=300)
