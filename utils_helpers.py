# utils/helpers.py
import os
import torch
import numpy as np

# 사용 가능 디바이스 확인 (CUDA 우선)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 디렉토리 생성 유틸리티 (존재하지 않으면 생성)
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# 텐서 denormalize ([-1,1] -> [0,255] 이미지 배열)
def denormalize(tensor):
    # 입력: 3채널 텐서 (C,H,W), 출력: 0~255 uint8 numpy 배열 (H,W,C)
    t = tensor.clone().detach().cpu() * 0.5 + 0.5  # [-1,1] -> [0,1]
    t = t.clamp(0, 1)
    arr = (t.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return arr
