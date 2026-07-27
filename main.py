# main.py
"""DualNet-R 약지도 손상 복원 프레임워크 실행 엔트리포인트.

사용 예:
    python main.py --mode train
    python main.py --mode train --teacher sd15 --lambda-mask 0.0 --exp-name lambda0
    python main.py --mode evaluate --test-dir CarDD_release/CarDD_SOD/CarDD-TE
    python main.py --mode inference --input sample.jpg --output-dir outputs/demo

모든 하이퍼파라미터의 기본값은 config.yaml 에 있고, 아래 플래그와
`--set key.path=value` 로 덮어쓴다.
"""

import argparse
import os
import sys

from utils.config import load_config


def str2bool(value):
    """ablation 스크립트가 넘기는 "True"/"False" 문자열도 처리."""
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("true", "t", "yes", "y", "1"):
        return True
    if lowered in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"불리언 값이 아닙니다: {value}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="DualNet-R 약지도 손상 복원 프레임워크 실행",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "evaluate", "inference"], required=True,
                        help="실행 모드: train(학습) / evaluate(평가) / inference(추론)")
    parser.add_argument("--config", type=str, default=None,
                        help="설정 파일 경로 (기본값: 프로젝트 루트의 config.yaml)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="config 값 덮어쓰기 (예: --set train.batch_size=4). 여러 번 사용 가능")
    parser.add_argument("--print-config", action="store_true",
                        help="최종 적용된 설정을 출력하고 종료")

    # --- 자주 바꾸는 하이퍼파라미터 (config 값을 덮어씀) ---
    group = parser.add_argument_group("하이퍼파라미터 오버라이드")
    group.add_argument("--seed", type=int, help="전역 시드")
    group.add_argument("--lr", type=float, help="Adam 학습률")
    group.add_argument("--batch-size", type=int, help="배치 크기")
    group.add_argument("--epochs", type=int, help="세그멘테이션·복원 최대 epoch 수")
    group.add_argument("--lambda-mask", type=float, help="마스크 가중 L1 손실의 λ")
    group.add_argument("--teacher", type=str,
                       help="teacher 체크포인트 (sd15 | sd21 | HuggingFace ID)")
    group.add_argument("--self-training-rounds", type=int, help="self-training 라운드 수 K")

    # --- ablation 플래그 (03_ablation_combinations.py 와 이름 일치) ---
    abl = parser.add_argument_group("ablation 플래그")
    abl.add_argument("--use-pseudo-gt", type=str2bool, help="pseudo-GT 생성(teacher) 사용 여부")
    abl.add_argument("--use-self-training", type=str2bool, help="self-training 사용 여부")
    abl.add_argument("--use-attention", type=str2bool, help="attention 게이트 사용 여부")
    abl.add_argument("--exp-name", type=str,
                     help="실험 이름. checkpoint_dir/output_dir 아래 같은 이름의 하위 폴더를 사용")

    # --- 경로 ---
    paths = parser.add_argument_group("경로")
    paths.add_argument("--dataset-dir", type=str, help="학습 데이터셋 디렉토리")
    paths.add_argument("--test-dir", type=str, help="평가용 테스트 디렉토리")
    paths.add_argument("--output-dir", type=str, help="결과 저장 디렉토리")
    paths.add_argument("--work-dir", type=str, help="예측 마스크·pseudo-GT 중간 산출물 디렉토리")
    paths.add_argument("--input", type=str, help="추론 입력 이미지 또는 디렉토리")
    paths.add_argument("--gt", type=str, help="추론 시 비교할 정상 원본 이미지/디렉토리 (선택)")
    paths.add_argument("--seg-model-path", type=str, help="세그멘테이션 가중치 경로")
    paths.add_argument("--rest-model-path", type=str, help="복원 가중치 경로")
    return parser


# CLI 플래그 → config 키 매핑 (단순 1:1 대응)
FLAG_TO_KEY = {
    "seed": "seed",
    "lr": "train.lr",
    "batch_size": "train.batch_size",
    "lambda_mask": "loss.lambda_mask",
    "teacher": "teacher.name",
    "self_training_rounds": "self_training.rounds",
    "use_pseudo_gt": "teacher.enabled",
    "use_self_training": "self_training.enabled",
    "use_attention": "model.use_attention",
    "dataset_dir": "paths.dataset_dir",
    "test_dir": "paths.test_dir",
    "output_dir": "paths.output_dir",
    "work_dir": "paths.work_dir",
}


def apply_cli_overrides(cfg, args):
    for flag, key in FLAG_TO_KEY.items():
        value = getattr(args, flag, None)
        if value is not None:
            cfg.set_path(key, value)
    if args.epochs is not None:
        cfg.set_path("train.segmentation.epochs", args.epochs)
        cfg.set_path("train.restoration.epochs", args.epochs)
    if args.exp_name:
        # 실험별로 가중치와 결과를 분리 (work_dir 는 공유하므로 pseudo-GT 재생성이
        # 필요한 실험은 --work-dir 로 따로 지정할 것)
        cfg.set_path("paths.checkpoint_dir",
                     os.path.join(cfg["paths"]["checkpoint_dir"], args.exp_name))
        cfg.set_path("paths.output_dir",
                     os.path.join(cfg["paths"]["output_dir"], args.exp_name))
    return cfg


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    cfg = apply_cli_overrides(cfg, args)

    if args.print_config:
        import yaml
        print(yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False))
        return 0

    if args.mode == "train":
        import train
        train.train(cfg)
    elif args.mode == "evaluate":
        import evaluate
        evaluate.evaluate(cfg, seg_weights=args.seg_model_path,
                          rest_weights=args.rest_model_path)
    elif args.mode == "inference":
        if not args.input:
            parser.error("--mode inference 에는 --input (이미지 경로 또는 디렉토리)이 필요합니다")
        import inference
        inference.run_inference(cfg, args.input, output_dir=args.output_dir, gt=args.gt,
                                seg_weights=args.seg_model_path,
                                rest_weights=args.rest_model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
