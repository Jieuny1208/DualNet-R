# evaluate.py
"""테스트셋 평가: 세그멘테이션 IoU + 복원 PSNR/SSIM + 수리 가능성 등급.

기대하는 test_dir 구조 (original/masks 는 선택):
    test_dir/
      images/    손상 이미지 (없으면 test_dir 자체를 이미지 폴더로 사용)
      original/  정상 상태 원본 (PSNR/SSIM 계산용)
      masks/     정답 마스크 (IoU 계산용)
"""

import csv
import os

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.self_training import SelfTraining
from models.unet import build_unet
from utils.helpers import (
    denormalize,
    ensure_dir,
    image_transform,
    index_by_stem,
    list_images,
    resolve_device,
)
from utils.seed import set_global_seed

# 수리 가능성 판정 기준 (논문 5.x 규칙 기반 모듈의 간이 버전)
GRADE_THRESHOLDS = {
    "irreparable": {"psnr": 20.0, "ssim": 0.80, "damage_ratio": 30.0},
    "excellent": {"psnr": 30.0, "ssim": 0.90, "damage_ratio": 10.0},
}


def load_models(cfg, device, seg_weights=None, rest_weights=None):
    """세그멘테이션 / 복원 모델을 config 스펙대로 만들고 가중치를 로드."""
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    seg_weights = seg_weights or os.path.join(ckpt_dir, "segmentation.pth")
    rest_weights = rest_weights or os.path.join(ckpt_dir, "restoration.pth")
    for path in (seg_weights, rest_weights):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {path}")

    seg_model = build_unet(cfg, "segmentation", device)
    rest_model = build_unet(cfg, "restoration", device)
    seg_model.load_state_dict(torch.load(seg_weights, map_location=device))
    rest_model.load_state_dict(torch.load(rest_weights, map_location=device))
    seg_model.eval()
    rest_model.eval()
    return seg_model, rest_model


@torch.no_grad()
def restore_image(cfg, seg_model, rest_model, img, device):
    """단일 이미지 복원. (복원 PIL 이미지, 예측 마스크 PIL 이미지) 반환."""
    image_size = cfg["data"]["image_size"]
    threshold = cfg["eval"]["mask_threshold"]

    st = SelfTraining(seg_model, device, threshold=threshold, image_size=image_size)
    mask_pil = st.predict_mask(img)  # 원본 해상도 이진 마스크

    transform = image_transform(image_size)
    img_tensor = transform(img).unsqueeze(0).to(device)
    mask_small = mask_pil.resize((image_size, image_size), resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(
        (np.array(mask_small, dtype=np.float32) > 127).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)

    output_tensor = rest_model(torch.cat([img_tensor, mask_tensor], dim=1))
    output_pil = Image.fromarray(denormalize(output_tensor[0]))
    # 지표 계산은 원본 해상도에서 수행
    if output_pil.size != img.size:
        output_pil = output_pil.resize(img.size, resample=Image.BILINEAR)
    return output_pil, mask_pil


def compute_damage_ratio(gt_img, restored_img, mask_arr):
    """예측 마스크 영역 내에서 밝기 차이가 20 이상인 픽셀 비율(%)."""
    gt_gray = np.array(gt_img.convert("L"), dtype=np.int16)
    rest_gray = np.array(restored_img.convert("L"), dtype=np.int16)
    diff_map = np.abs(gt_gray - rest_gray)
    mask_bin = mask_arr > 127
    if mask_bin.shape != diff_map.shape:
        return float("nan")
    damaged = np.logical_and(diff_map > 20, mask_bin).sum()
    total = mask_bin.sum()
    return float(damaged) / float(total) * 100.0 if total else 0.0


def grade_restoration(psnr_val, ssim_val, damage_ratio):
    bad, good = GRADE_THRESHOLDS["irreparable"], GRADE_THRESHOLDS["excellent"]
    dr = 0.0 if damage_ratio != damage_ratio else damage_ratio  # NaN 방어
    if psnr_val < bad["psnr"] or ssim_val < bad["ssim"] or dr > bad["damage_ratio"]:
        return "Irreparable"
    if psnr_val >= good["psnr"] and ssim_val >= good["ssim"] and dr <= good["damage_ratio"]:
        return "Excellent"
    return "Repairable"


def evaluate(cfg, seg_weights=None, rest_weights=None):
    set_global_seed(cfg["seed"], deterministic=cfg["deterministic"])
    device = resolve_device(cfg["device"])
    print(f"사용 장치: {device}")

    test_dir = cfg["paths"]["test_dir"]
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"테스트 데이터 디렉토리가 존재하지 않습니다: {test_dir}")
    image_dir = os.path.join(test_dir, "images")
    if not os.path.isdir(image_dir):
        image_dir = test_dir
    orig_index = index_by_stem(os.path.join(test_dir, "original"))
    gt_mask_index = index_by_stem(os.path.join(test_dir, "masks"))

    image_paths = list_images(image_dir)
    if not image_paths:
        print("테스트할 이미지가 없습니다.")
        return {}
    print(f"테스트 이미지 수: {len(image_paths)} "
          f"(원본 {len(orig_index)}장, 정답 마스크 {len(gt_mask_index)}장)")

    seg_model, rest_model = load_models(cfg, device, seg_weights, rest_weights)

    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    restored_dir = ensure_dir(os.path.join(output_dir, "restored")) if cfg["eval"]["save_outputs"] else None

    # 비교용 teacher 파이프라인 (선택)
    teacher = None
    if cfg["eval"]["compare_teacher"] and cfg["teacher"]["enabled"]:
        from models.teacher import DiffusionTeacher
        teacher = DiffusionTeacher(cfg, device).load()

    rows = []
    try:
        for i, img_path in enumerate(image_paths):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            img = Image.open(img_path).convert("RGB")
            output_pil, mask_pil = restore_image(cfg, seg_model, rest_model, img, device)
            mask_arr = np.array(mask_pil)

            if restored_dir:
                output_pil.save(os.path.join(restored_dir, f"{stem}.png"))

            row = {"image": stem, "psnr": None, "ssim": None, "damage_ratio": None,
                   "grade": None, "iou": None, "teacher_psnr": None, "teacher_ssim": None}

            if stem in orig_index:
                gt_img = Image.open(orig_index[stem]).convert("RGB")
                if gt_img.size != output_pil.size:
                    gt_img = gt_img.resize(output_pil.size, resample=Image.BILINEAR)
                gt_arr, out_arr = np.array(gt_img), np.array(output_pil)
                row["psnr"] = float(peak_signal_noise_ratio(gt_arr, out_arr, data_range=255))
                row["ssim"] = float(structural_similarity(gt_arr, out_arr, channel_axis=-1,
                                                          data_range=255))
                row["damage_ratio"] = compute_damage_ratio(gt_img, output_pil, mask_arr)
                row["grade"] = grade_restoration(row["psnr"], row["ssim"], row["damage_ratio"])

                if teacher is not None:
                    try:
                        teacher_pil = teacher.restore(img, mask_pil, index=i)
                        t_arr = np.array(teacher_pil.resize(gt_img.size, resample=Image.BILINEAR))
                        row["teacher_psnr"] = float(
                            peak_signal_noise_ratio(gt_arr, t_arr, data_range=255))
                        row["teacher_ssim"] = float(
                            structural_similarity(gt_arr, t_arr, channel_axis=-1, data_range=255))
                    except Exception as exc:
                        print(f"teacher 복원 실패 ({stem}):", exc)

            if stem in gt_mask_index:
                gt_mask = Image.open(gt_mask_index[stem]).convert("L")
                if gt_mask.size != mask_pil.size:
                    gt_mask = gt_mask.resize(mask_pil.size, resample=Image.NEAREST)
                gt_bin = np.array(gt_mask) > 127
                pred_bin = mask_arr > 127
                union = np.logical_or(pred_bin, gt_bin).sum()
                inter = np.logical_and(pred_bin, gt_bin).sum()
                row["iou"] = 1.0 if union == 0 else float(inter) / float(union)

            rows.append(row)
            if row["psnr"] is not None:
                print(f"{stem}: PSNR {row['psnr']:.2f}dB, SSIM {row['ssim']:.4f}, "
                      f"Damage {row['damage_ratio']:.1f}%, Grade {row['grade']}")
    finally:
        if teacher is not None:
            teacher.unload()

    # per-image CSV 저장 (통계 검정 스크립트 입력으로 사용)
    csv_path = os.path.join(output_dir, cfg["eval"]["per_image_csv"])
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nper-image 지표 저장: {csv_path}")

    def _mean(key):
        values = [r[key] for r in rows if r[key] is not None]
        return sum(values) / len(values) if values else None

    summary = {
        "count": len(rows),
        "psnr": _mean("psnr"),
        "ssim": _mean("ssim"),
        "iou": _mean("iou"),
        "teacher_psnr": _mean("teacher_psnr"),
        "teacher_ssim": _mean("teacher_ssim"),
    }
    print("\n===== 평가 요약 =====")
    if summary["iou"] is not None:
        print(f"세그멘테이션 평균 IoU: {summary['iou'] * 100:.1f}%")
    else:
        print("정답 마스크가 없어 세그멘테이션 IoU 를 계산할 수 없습니다.")
    if summary["psnr"] is not None:
        print(f"DualNet-R 평균 PSNR: {summary['psnr']:.2f} dB")
        print(f"DualNet-R 평균 SSIM: {summary['ssim']:.4f}")
    else:
        print("원본 이미지가 없어 PSNR/SSIM 을 계산할 수 없습니다.")
    if summary["teacher_psnr"] is not None:
        print(f"Teacher(SD) 평균 PSNR: {summary['teacher_psnr']:.2f} dB")
        print(f"Teacher(SD) 평균 SSIM: {summary['teacher_ssim']:.4f}")
    return summary
