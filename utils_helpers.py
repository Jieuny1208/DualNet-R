# utils/helpers.py
import os
import torch
import numpy as np

# ��� ���� ����̽� Ȯ�� (CUDA �켱)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ���丮 ���� ��ƿ��Ƽ (�������� ������ ����)
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# �ټ� denormalize ([-1,1] -> [0,255] �̹��� �迭)
def denormalize(tensor):
    # �Է�: 3ä�� �ټ� (C,H,W), ���: 0~255 uint8 numpy �迭 (H,W,C)
    t = tensor.clone().detach().cpu() * 0.5 + 0.5  # [-1,1] -> [0,1]
    t = t.clamp(0, 1)
    arr = (t.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return arr
