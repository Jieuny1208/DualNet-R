# preflight.py
"""Phase 1 실행 전 사전 점검 (수 초 ~ 수십 초).

  A. 데이터셋: 학습/테스트 이미지·마스크 존재 여부와 파일명 대응
  B. teacher : HF 토큰 / 체크포인트 접근 권한(gated 여부) / 캐시 상태
  C. 환경    : device, 라이브러리 버전, 디스크 여유
  D. 예상 시간: --benchmark 로 실제 스텝 시간을 재서 Phase 1 소요를 추정

9일 걸리는 작업에 들어가기 전에 이걸 먼저 통과시킬 것.

실행:  python preflight.py [--benchmark]
"""

import argparse
import math
import os
import shutil
import sys
import time

OK, FAIL, WARN = "OK", "FAIL", "WARN"
_results = []


def report(status, label, detail=""):
    mark = {OK: "OK  ", FAIL: "FAIL", WARN: "WARN"}[status]
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    _results.append(status)
    return status == OK


def section(title):
    print(f"\n=== {title} ===")


def check_dataset(cfg):
    from utils.helpers import index_by_stem, list_images

    section("A. 데이터셋")
    p = cfg["paths"]
    dataset_dir = p["dataset_dir"]
    if not os.path.isdir(dataset_dir):
        report(FAIL, "학습 데이터 디렉토리", f"{dataset_dir} 없음")
        return False

    img_dir = os.path.join(dataset_dir, p["train_image_subdir"])
    mask_dir = os.path.join(dataset_dir, p["train_mask_subdir"])
    images = list_images(img_dir)
    masks = index_by_stem(mask_dir)
    bbox = os.path.join(dataset_dir, p["bbox_file"]) if p["bbox_file"] else None

    ok = report(OK if images else FAIL, "학습 이미지", f"{len(images)}장 @ {img_dir}")
    if masks:
        matched = sum(1 for i in images
                      if os.path.splitext(os.path.basename(i))[0] in masks)
        report(OK if matched else FAIL, "약지도 마스크 대응",
               f"{matched}/{len(images)}장 매칭 (마스크 {len(masks)}개)")
        if matched and matched < len(images):
            report(WARN, "일부 이미지에 마스크 없음", f"{len(images) - matched}장은 학습에서 제외됨")
    elif bbox and os.path.isfile(bbox):
        report(OK, "bbox CSV 라벨", bbox)
    else:
        report(FAIL, "라벨 없음", f"{mask_dir} 도 {bbox} 도 없음")
        ok = False

    test_dir = p["test_dir"]
    test_img = os.path.join(test_dir, "images")
    if not os.path.isdir(test_img):
        test_img = test_dir
    t_images = list_images(test_img)
    t_orig = index_by_stem(os.path.join(test_dir, "original"))
    t_mask = index_by_stem(os.path.join(test_dir, "masks"))
    report(OK if t_images else FAIL, "테스트 이미지", f"{len(t_images)}장 @ {test_img}")
    report(OK if t_orig else FAIL, "테스트 원본(PSNR/SSIM용)",
           f"{len(t_orig)}장" + ("" if t_orig else " — 없으면 Phase 1 게이트 판정 불가"))
    report(OK if t_mask else WARN, "테스트 정답 마스크(IoU용)", f"{len(t_mask)}장")
    return ok and bool(t_images) and bool(t_orig)


def check_teacher(cfg):
    section("B. teacher 체크포인트")
    from models.teacher import resolve_checkpoint

    if not cfg["teacher"]["enabled"]:
        report(WARN, "teacher.enabled=false", "pseudo-GT 생성을 건너뜁니다")
        return True

    repo = resolve_checkpoint(cfg)
    report(OK, "체크포인트", f"{cfg['teacher']['name']} → {repo}")

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    report(OK if token else WARN, "HF 토큰", "설정됨" if token else
           "없음 — gated 저장소(sd21)라면 401 로 실패합니다")

    try:
        from huggingface_hub import model_info
        info = model_info(repo, token=token)
        report(OK, "저장소 접근 권한", f"{repo} 조회 성공 (gated={getattr(info, 'gated', None)})")
        accessible = True
    except Exception as exc:
        name = type(exc).__name__
        hint = ""
        if "401" in str(exc) or "Unauthorized" in name or "GatedRepo" in name:
            hint = (f" → https://huggingface.co/{repo} 에서 라이선스 동의 후 "
                    "HF_TOKEN 환경변수를 설정하세요")
        report(FAIL, "저장소 접근 권한", f"{name}{hint}")
        accessible = False

    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(repo, "model_index.json")
        report(OK if cached else WARN, "로컬 캐시",
               "캐시됨 (재다운로드 불필요)" if cached else
               "미캐시 — 최초 실행 시 약 5~11GB 다운로드")
    except Exception:
        pass

    for key, expect in (("num_inference_steps", 50), ("guidance_scale", 7.5)):
        report(OK if cfg["teacher"][key] == expect else WARN, f"teacher.{key}",
               str(cfg["teacher"][key]))
    report(OK if str(cfg["teacher"]["scheduler"]).lower() == "ddim" else WARN,
           "teacher.scheduler", str(cfg["teacher"]["scheduler"]))
    return accessible


def check_environment(cfg):
    section("C. 실행 환경")
    import torch
    from utils.helpers import resolve_device

    device = resolve_device(cfg["device"])
    report(OK, "torch", torch.__version__)
    try:
        import diffusers
        report(OK, "diffusers", diffusers.__version__)
    except ImportError:
        report(FAIL, "diffusers", "미설치 — pip install -r requirements.txt")

    if device.type == "cuda":
        report(OK, "device", f"cuda ({torch.cuda.get_device_name(0)}) — fp16 teacher 사용")
    elif device.type == "mps":
        report(WARN, "device", "mps (Apple GPU) — teacher 가 fp32 로 동작, CUDA 대비 매우 느림")
    else:
        report(WARN, "device", "cpu — Phase 1 규모에는 현실적으로 부적합")

    free_gb = shutil.disk_usage(".").free / 2 ** 30
    report(OK if free_gb > 30 else WARN, "디스크 여유",
           f"{free_gb:.0f} GB (체크포인트+pseudo-GT+캐시로 30GB 이상 권장)")
    return device


def estimate_time(cfg, device, benchmark=False):
    section("D. Phase 1 예상 소요 시간")
    import torch
    from models.unet import build_unet
    from utils.helpers import list_images

    n_train = len(list_images(os.path.join(cfg["paths"]["dataset_dir"],
                                           cfg["paths"]["train_image_subdir"]))) or 960
    bs = cfg["train"]["batch_size"]
    steps = math.ceil(n_train / bs)

    if not benchmark:
        print("  (--benchmark 를 주면 실제 스텝 시간을 측정해 추정합니다)")
        return

    def bench(role, in_ch, size, iters=2):
        model = build_unet(cfg, role, device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        x = torch.randn(bs, in_ch, size, size, device=device)
        y = torch.randn(bs, 1 if role == "segmentation" else 3, size, size, device=device)
        lossf = (torch.nn.BCEWithLogitsLoss() if role == "segmentation"
                 else torch.nn.L1Loss())
        ts = []
        for i in range(iters + 1):
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            lossf(model(x), y).backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            if i:
                ts.append(time.time() - t0)
        del model, opt, x, y
        return sum(ts) / len(ts)

    size = cfg["data"]["image_size"]
    seg_s = bench("segmentation", 3, size)
    rest_s = bench("restoration", 4, size)
    print(f"  측정: seg {seg_s:.2f}s/step, student {rest_s:.2f}s/step "
          f"({size}px, batch {bs}, {steps} step/epoch)")

    # teacher 생성 시간은 하드웨어별 편차가 커서 대표값을 범위로 제시
    gen_lo, gen_hi = (6, 10) if device.type == "cuda" else (150, 300)
    ep = cfg["train"]["segmentation"]["epochs"]
    for label, e in (("early stop ~100 epoch", min(100, ep)), (f"상한 {ep} epoch", ep)):
        train_h = (steps * seg_s * e * 2 + steps * rest_s * e) / 3600
        pgt_lo, pgt_hi = n_train * gen_lo / 3600, n_train * gen_hi / 3600
        print(f"  {label}: 학습 {train_h:.1f}h + pseudo-GT {pgt_lo:.1f}~{pgt_hi:.1f}h "
              f"= 총 {train_h + pgt_lo:.1f}~{train_h + pgt_hi:.1f}h "
              f"({(train_h + pgt_lo) / 24:.1f}~{(train_h + pgt_hi) / 24:.1f}일)")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 사전 점검")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true",
                        help="실제 학습 스텝 시간을 측정해 소요 시간 추정")
    args = parser.parse_args()

    from utils.config import load_config
    cfg = load_config(args.config)
    print(f"config: {cfg['_config_path']}")

    data_ok = check_dataset(cfg)
    teacher_ok = check_teacher(cfg)
    device = check_environment(cfg)
    estimate_time(cfg, device, args.benchmark)

    n_fail = _results.count(FAIL)
    n_warn = _results.count(WARN)
    print(f"\n===== 사전 점검: {_results.count(OK)} OK / {n_warn} WARN / {n_fail} FAIL =====")
    if n_fail:
        print("FAIL 항목을 해결한 뒤 Phase 1 을 시작하세요.")
    elif not (data_ok and teacher_ok):
        print("일부 항목이 불완전합니다. 위 경고를 확인하세요.")
    else:
        print("Phase 1 실행 가능 상태입니다.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
