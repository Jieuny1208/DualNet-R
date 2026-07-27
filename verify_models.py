# verify_models.py
"""학습 없이 수행하는 정적 검증.

  1) config.yaml 로드 및 주요 스펙 확인
  2) 세그멘테이션 / 복원 네트워크 인스턴스화 + 파라미터 수 출력 (목표 31.4M)
  3) 더미 텐서 forward 통과 및 출력 shape/range 확인
  4) 마스크 가중 L1 손실의 수치 검증

실행:  python verify_models.py [--size 64] [--config config.yaml]
"""

import argparse

import torch

from models.loss import MaskWeightedL1Loss
from models.unet import build_unet
from utils.config import load_config
from utils.helpers import count_parameters
from utils.seed import set_global_seed

TARGET_PARAMS_M = 31.4
TOLERANCE_M = 0.1


def check(label, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="DualNet-R 정적 검증 (학습 없음)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--size", type=int, default=64, help="더미 입력 해상도 (CPU 검증용)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    passed = []

    print("\n=== 1. config 확인 ===")
    print(f"  config: {cfg['_config_path']}")
    passed.append(check("seed = 42", cfg["seed"] == 42, str(cfg["seed"])))
    passed.append(check("features = [64,128,256,512]",
                        list(cfg["model"]["features"]) == [64, 128, 256, 512],
                        str(cfg["model"]["features"])))
    passed.append(check("Adam lr = 1e-4", float(cfg["train"]["lr"]) == 1e-4,
                        str(cfg["train"]["lr"])))
    passed.append(check("batch size = 8", cfg["train"]["batch_size"] == 8,
                        str(cfg["train"]["batch_size"])))
    passed.append(check("StepLR gamma = 0.5",
                        cfg["train"]["scheduler"]["enabled"]
                        and float(cfg["train"]["scheduler"]["gamma"]) == 0.5,
                        f"step_size={cfg['train']['scheduler']['step_size']}"))
    passed.append(check("teacher: DDIM / 50 steps / guidance 7.5",
                        str(cfg["teacher"]["scheduler"]).lower() == "ddim"
                        and cfg["teacher"]["num_inference_steps"] == 50
                        and float(cfg["teacher"]["guidance_scale"]) == 7.5))
    passed.append(check('negative prompt = "blurry, distorted, unrealistic"',
                        cfg["teacher"]["negative_prompt"] == "blurry, distorted, unrealistic"))
    passed.append(check("self-training K >= 1", int(cfg["self_training"]["rounds"]) >= 1,
                        f"K={cfg['self_training']['rounds']}"))
    passed.append(check("lambda_mask 정의됨", "lambda_mask" in cfg["loss"],
                        f"λ={cfg['loss']['lambda_mask']}"))

    print("\n=== 2. 모델 인스턴스화 & 파라미터 수 ===")
    seg_model = build_unet(cfg, "segmentation")
    rest_model = build_unet(cfg, "restoration")
    seg_params = count_parameters(seg_model)
    rest_params = count_parameters(rest_model)
    print(f"  세그멘테이션 (in=3, out=1) : {seg_params:,} ({seg_params / 1e6:.2f}M)")
    print(f"  복원 student (in=4, out=3) : {rest_params:,} ({rest_params / 1e6:.2f}M)")
    passed.append(check(f"세그멘테이션 ≈ {TARGET_PARAMS_M}M",
                        abs(seg_params / 1e6 - TARGET_PARAMS_M) <= TOLERANCE_M))
    passed.append(check(f"복원 student ≈ {TARGET_PARAMS_M}M",
                        abs(rest_params / 1e6 - TARGET_PARAMS_M) <= TOLERANCE_M))

    no_att = build_unet(cfg.copy().set_path("model.use_attention", False), "segmentation")
    no_att_params = count_parameters(no_att)
    print(f"  w/o attention (ablation)   : {no_att_params:,} ({no_att_params / 1e6:.2f}M)")
    passed.append(check("attention 제거 시 파라미터 감소", no_att_params < seg_params,
                        f"-{(seg_params - no_att_params):,}"))

    print(f"\n=== 3. 더미 텐서 forward ({args.size}x{args.size}) ===")
    size = args.size
    seg_model.eval()
    rest_model.eval()
    with torch.no_grad():
        x_seg = torch.randn(2, 3, size, size)
        y_seg = seg_model(x_seg)
        passed.append(check("세그멘테이션 출력 shape (2,1,H,W)",
                            tuple(y_seg.shape) == (2, 1, size, size), str(tuple(y_seg.shape))))
        prob = torch.sigmoid(y_seg)
        passed.append(check("sigmoid 확률맵 범위 [0,1]",
                            bool((prob >= 0).all() and (prob <= 1).all())))

        # x ⊕ M : RGB 3채널 + 마스크 1채널
        img = torch.randn(2, 3, size, size)
        mask = (torch.rand(2, 1, size, size) > 0.5).float()
        y_rest = rest_model(torch.cat([img, mask], dim=1))
        passed.append(check("복원 출력 shape (2,3,H,W)",
                            tuple(y_rest.shape) == (2, 3, size, size), str(tuple(y_rest.shape))))
        passed.append(check("tanh 출력 범위 [-1,1]",
                            bool((y_rest >= -1).all() and (y_rest <= 1).all())))

        # 홀수 해상도에서도 skip 연결 보간이 동작하는지 확인
        odd = seg_model(torch.randn(1, 3, size + 5, size + 3))
        passed.append(check("비정수배 해상도 forward",
                            tuple(odd.shape) == (1, 1, size + 5, size + 3), str(tuple(odd.shape))))

    print("\n=== 4. 마스크 가중 L1 손실 ===")
    pred = torch.zeros(1, 3, 4, 4)
    target = torch.ones(1, 3, 4, 4)      # |pred - target| = 1 인 상수 오차
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :2, :] = 1.0               # 픽셀의 절반이 손상 영역
    loss_l1 = MaskWeightedL1Loss(lambda_mask=0.0)(pred, target, mask)
    loss_w = MaskWeightedL1Loss(lambda_mask=1.0)(pred, target, mask)
    # λ=1, 마스크 비율 0.5 → 평균 가중치 1.5
    passed.append(check("λ=0 이면 일반 L1 (=1.0)", abs(loss_l1.item() - 1.0) < 1e-6,
                        f"{loss_l1.item():.4f}"))
    passed.append(check("λ=1, 마스크 50% → 1.5", abs(loss_w.item() - 1.5) < 1e-6,
                        f"{loss_w.item():.4f}"))

    total, ok = len(passed), sum(passed)
    print(f"\n===== 검증 결과: {ok}/{total} 통과 =====")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
