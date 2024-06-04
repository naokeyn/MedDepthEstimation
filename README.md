# Depth Anythingで深度推定を試す

![](https://img.shields.io/badge/python-v.3.8.10-blue?logo=python)
![](https://img.shields.io/badge/huggingface-transformers-orange?logo=huggingface)

## 環境構築
520のUbuntuでやりました
- Docker
- Ubuntu
- Linux

### Dockerコンテナの作成
```bash
$ docker pull huggingface/transformers-pytorch-gpu:latest bash
$ docker run -it \
	-v $(pwd):/app \
	-w /app \
	-p 8000:8000 \
	--gpus all \
	--name depth_anything \
	huggingface/transformers-pytorch-gpu:latest bash
```

### コンテナ内の設定

```bash
apt-get update -y && apt update -y
```

`cv2` のインストール
```bash
python3 -m pip install opencv-python
```

CUDA使えるか確認
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```
`True`が帰ってくればおっけい


## 深度推定
### 画像の深度推定
`src/estimate_img` のパスを書き換えて実行
```bash
$ cd src
$ python3 estimate_img.py
```

### 動画の深度推定
`src/estimate_video.py` のパスを書き換えて実行
```bash
$ cd src
$ python3 estimate_video.py
```

## 参考
- [Hugging Face | Depth Anything](https://huggingface.co/docs/transformers/model_doc/depth_anything)
- [GitHub | Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
