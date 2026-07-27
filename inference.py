# inference.py
"""단일 이미지 / 디렉토리 추론.

파이프라인은 학습과 동일하다: 세그멘테이션으로 손상 마스크를 예측한 뒤
(RGB ⊕ mask) 4채널을 복원 네트워크에 넣어 복원 이미지를 얻는다.
정상 원본(--gt)이 주어지면 PSNR/SSIM 과 수리 가능 여부도 함께 출력한다.
"""

import os

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from evaluate import grade_restoration, load_models, restore_image
from utils.helpers import ensure_dir, list_images, resolve_device
from utils.seed import set_global_seed

# 수리 가능 판정 기준
PSNR_THRESHOLD = 20.0
SSIM_THRESHOLD = 0.80


def _collect_inputs(input_path):
    if os.path.isdir(input_path):
        paths = list_images(input_path)
        if not paths:
            raise FileNotFoundError(f"입력 디렉토리에 이미지가 없습니다: {input_path}")
        return paths
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {input_path}")
    return [input_path]


def _resolve_gt(gt_arg, img_path):
    """--gt 가 디렉토리면 같은 stem 의 파일을, 파일이면 그 파일을 사용."""
    if not gt_arg:
        return None
    if os.path.isdir(gt_arg):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        for cand in list_images(gt_arg):
            if os.path.splitext(os.path.basename(cand))[0] == stem:
                return cand
        return None
    return gt_arg if os.path.isfile(gt_arg) else None


def run_inference(cfg, input_path, output_dir=None, gt=None,
                  seg_weights=None, rest_weights=None, save_mask=True):
    set_global_seed(cfg["seed"], deterministic=cfg["deterministic"])
    device = resolve_device(cfg["device"])
    print(f"사용 장치: {device}")

    image_paths = _collect_inputs(input_path)
    output_dir = ensure_dir(output_dir or os.path.join(cfg["paths"]["output_dir"], "inference"))
    seg_model, rest_model = load_models(cfg, device, seg_weights, rest_weights)

    results = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert("RGB")
        restored, mask_pil = restore_image(cfg, seg_model, rest_model, img, device)

        restored_path = os.path.join(output_dir, f"{stem}_restored.png")
        restored.save(restored_path)
        if save_mask:
            mask_pil.save(os.path.join(output_dir, f"{stem}_mask.png"))

        result = {"image": stem, "restored": restored_path,
                  "psnr": None, "ssim": None, "repairable": None}

        gt_path = _resolve_gt(gt, img_path)
        if gt_path:
            gt_img = Image.open(gt_path).convert("RGB")
            if gt_img.size != restored.size:
                gt_img = gt_img.resize(restored.size, resample=Image.BILINEAR)
            gt_arr, out_arr = np.array(gt_img), np.array(restored)
            result["psnr"] = float(peak_signal_noise_ratio(gt_arr, out_arr, data_range=255))
            result["ssim"] = float(structural_similarity(gt_arr, out_arr, channel_axis=-1,
                                                         data_range=255))
            result["repairable"] = (result["psnr"] >= PSNR_THRESHOLD
                                    and result["ssim"] >= SSIM_THRESHOLD)
            grade = grade_restoration(result["psnr"], result["ssim"], 0.0)
            print(f"{stem}: PSNR {result['psnr']:.2f}dB, SSIM {result['ssim']:.4f} → "
                  f"{'수리 가능' if result['repairable'] else '수리 불가'} ({grade})")
        else:
            print(f"{stem}: 복원 결과 저장 → {restored_path} "
                  f"(원본 미제공으로 PSNR/SSIM 평가 생략)")
        results.append(result)

    return results
