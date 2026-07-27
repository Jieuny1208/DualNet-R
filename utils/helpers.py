# utils/helpers.py
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# 사용 가능 디바이스 확인 (CUDA 우선)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_device(spec="auto"):
    """config 의 device 설정("auto"/"cuda"/"cpu"/"mps")을 torch.device 로 변환."""
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        print("[경고] CUDA 를 사용할 수 없어 CPU 로 대체합니다.")
        return torch.device("cpu")
    return torch.device(spec)


# 디렉토리 생성 유틸리티 (존재하지 않으면 생성)
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def list_images(directory):
    """디렉토리 내 이미지 파일 경로를 정렬된 리스트로 반환."""
    if not directory or not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory) if f.lower().endswith(VALID_IMAGE_EXTS)]
    files.sort()
    return [os.path.join(directory, f) for f in files]


def index_by_stem(directory):
    """디렉토리 내 이미지를 확장자 제외 파일명(stem) 기준으로 색인.

    마스크는 PNG, 원본은 JPG 처럼 확장자가 다른 경우가 많아 stem 으로 대응시킨다.
    """
    index = {}
    for path in list_images(directory):
        index[os.path.splitext(os.path.basename(path))[0]] = path
    return index


def image_transform(image_size=None):
    """RGB 이미지를 [-1,1] 범위 텐서로 변환하는 transform."""
    steps = []
    if image_size:
        size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        steps.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(steps)


def load_mask_tensor(path, image_size=None, threshold=0.5):
    """마스크 이미지를 {0,1} 값의 (1,H,W) 텐서로 로드 (최근접 보간)."""
    mask = Image.open(path).convert("L")
    if image_size:
        size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        mask = mask.resize((size[1], size[0]), resample=Image.NEAREST)
    tensor = transforms.functional.to_tensor(mask)
    return (tensor > threshold).float()


# 텐서 denormalize ([-1,1] -> [0,255] 이미지 배열)
def denormalize(tensor):
    # 입력: 3채널 텐서 (C,H,W), 출력: 0~255 uint8 numpy 배열 (H,W,C)
    t = tensor.clone().detach().cpu().float() * 0.5 + 0.5  # [-1,1] -> [0,1]
    t = t.clamp(0, 1)
    arr = (t.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return arr


def count_parameters(model, trainable_only=True):
    """모델 파라미터 수 계산."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
