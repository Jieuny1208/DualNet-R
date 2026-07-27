# utils/seed.py
"""전역 시드 고정 유틸리티.

논문 스펙: 데이터 분할 / 가중치 초기화 / pseudo-GT 생성 전부 동일 시드(기본 42)를 사용한다.
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed=42, deterministic=False):
    """random / numpy / torch(CPU·CUDA) 시드를 한 번에 고정한다.

    deterministic=True 이면 cuDNN 결정론 모드를 켠다 (속도는 다소 느려진다).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    return seed


def make_generator(seed=42, device="cpu"):
    """diffusers 파이프라인 등에 넘길 torch.Generator 를 시드 고정하여 생성한다."""
    device = torch.device(device)
    # CUDA generator 는 device 별로 생성해야 파이프라인이 그대로 사용할 수 있다.
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id):
    """DataLoader worker 시드 고정 (num_workers > 0 일 때 사용)."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
