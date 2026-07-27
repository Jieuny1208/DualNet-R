# train.py
"""DualNet-R 4단계 학습 파이프라인.

  1) 약지도 세그멘테이션 학습 (Attention U-Net, BCEWithLogits)
  2) Self-training refinement (K 라운드, pseudo-label 재학습)
  3) Stable Diffusion inpainting teacher 로 pseudo-GT 생성
  4) Student(복원) 네트워크 증류 학습 (마스크 가중 L1)

모든 하이퍼파라미터는 config.yaml 에서 온다.
"""

import json
import math
import os

import torch
from PIL import Image, ImageDraw
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.loss import LossFunctions
from models.self_training import SelfTraining
from models.teacher import generate_pseudo_gt, resolve_checkpoint
from models.unet import build_unet
from utils.helpers import (
    count_parameters,
    ensure_dir,
    image_transform,
    index_by_stem,
    list_images,
    resolve_device,
)
from utils.seed import seed_worker, set_global_seed


# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------
class SegDataset(Dataset):
    """세그멘테이션용 데이터셋 (약지도 마스크 또는 bbox 라벨)."""

    def __init__(self, image_dir, mask_dir=None, bbox_file=None, image_size=512, threshold=0.5):
        self.image_dir = image_dir
        self.image_size = image_size
        self.threshold = threshold
        self.transform = image_transform(image_size)
        self.image_paths = list_images(image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_dir}")

        # 마스크는 stem 기준으로 대응 (원본 .jpg ↔ 마스크 .png)
        self.mask_paths = index_by_stem(mask_dir) if mask_dir else {}

        # 마스크가 없을 때만 바운딩박스 CSV 사용
        self.bbox_info = {}
        if not self.mask_paths and bbox_file and os.path.isfile(bbox_file):
            self.bbox_info = self._load_bboxes(bbox_file)

        # 레이블이 하나라도 있으면 라벨 없는 이미지는 제외
        if self.mask_paths or self.bbox_info:
            self.image_paths = [
                p for p in self.image_paths
                if os.path.splitext(os.path.basename(p))[0] in self.mask_paths
                or os.path.splitext(os.path.basename(p))[0] in self.bbox_info
            ]
            if not self.image_paths:
                raise FileNotFoundError(
                    f"이미지와 라벨의 파일명이 일치하지 않습니다: {image_dir} / {mask_dir}"
                )

    @staticmethod
    def _load_bboxes(bbox_file):
        """bbox CSV 로드. x,y,w,h 또는 x1,y1,x2,y2 형식 지원."""
        bbox_info = {}
        try:
            import pandas as pd
        except ImportError:
            print("[경고] pandas 가 없어 bbox CSV 를 읽을 수 없습니다.")
            return bbox_info
        try:
            df = pd.read_csv(bbox_file)
            for _, row in df.iterrows():
                fname = str(row.get("filename") or row.get("image") or "")
                if not fname:
                    continue
                x = int(row.get("x", row.get("x1", 0)))
                y = int(row.get("y", row.get("y1", 0)))
                if "w" in row and "h" in row:
                    w, h = int(row["w"]), int(row["h"])
                else:
                    w = int(row.get("x2", 0)) - x
                    h = int(row.get("y2", 0)) - y
                bbox_info.setdefault(os.path.splitext(fname)[0], []).append((x, y, w, h))
        except Exception as exc:  # CSV 형식 문제는 학습을 막지 않고 경고만 출력
            print("바운딩 박스 CSV 파일 로드 오류:", exc)
        return bbox_info

    def __len__(self):
        return len(self.image_paths)

    def _load_label_mask(self, stem, size):
        """stem 에 해당하는 라벨을 PIL 'L' 마스크로 반환 (없으면 빈 마스크)."""
        if stem in self.mask_paths:
            return Image.open(self.mask_paths[stem]).convert("L")
        mask = Image.new("L", size, 0)
        if stem in self.bbox_info:
            draw = ImageDraw.Draw(mask)
            for x, y, w, h in self.bbox_info[stem]:
                draw.rectangle([x, y, x + w, y + h], fill=255)
        return mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert("RGB")
        mask = self._load_label_mask(stem, img.size)

        img_tensor = self.transform(img)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask_tensor = (transforms.functional.to_tensor(mask) > self.threshold).float()
        return img_tensor, mask_tensor


class RestDataset(Dataset):
    """복원(증류)용 데이터셋: (RGB ⊕ mask) → pseudo-GT."""

    def __init__(self, image_dir, mask_dir, target_dir, image_size=512, threshold=0.5):
        self.image_size = image_size
        self.threshold = threshold
        self.transform = image_transform(image_size)

        self.mask_paths = index_by_stem(mask_dir)
        self.target_paths = index_by_stem(target_dir)
        # 마스크와 pseudo-GT 가 모두 있는 이미지만 사용
        self.image_paths = [
            p for p in list_images(image_dir)
            if os.path.splitext(os.path.basename(p))[0] in self.mask_paths
            and os.path.splitext(os.path.basename(p))[0] in self.target_paths
        ]
        if not self.image_paths:
            raise FileNotFoundError(
                "복원 학습용 (이미지, 마스크, pseudo-GT) 쌍을 찾지 못했습니다.\n"
                f"  image_dir : {image_dir}\n  mask_dir  : {mask_dir}\n  target_dir: {target_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]

        img_tensor = self.transform(Image.open(img_path).convert("RGB"))
        target_tensor = self.transform(Image.open(self.target_paths[stem]).convert("RGB"))

        mask = Image.open(self.mask_paths[stem]).convert("L")
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask_tensor = (transforms.functional.to_tensor(mask) > self.threshold).float()

        # x ⊕ M (채널 결합, 4채널 입력)
        input_tensor = torch.cat([img_tensor, mask_tensor], dim=0)
        return input_tensor, target_tensor, mask_tensor


# ---------------------------------------------------------------------------
# 학습 헬퍼
# ---------------------------------------------------------------------------
def make_loader(cfg, dataset, shuffle=None):
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["data"]["shuffle"] if shuffle is None else shuffle,
        num_workers=cfg["data"]["num_workers"],
        drop_last=cfg["data"]["drop_last"],
        worker_init_fn=seed_worker if cfg["data"]["num_workers"] > 0 else None,
    )


def make_optimizer(cfg, model):
    """Adam(lr=1e-4) + StepLR(gamma=0.5) — 논문 스펙."""
    tcfg = cfg["train"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(tcfg["lr"]),
        betas=tuple(tcfg["betas"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    scheduler = None
    scfg = tcfg["scheduler"]
    if scfg["enabled"]:
        if str(scfg["type"]).lower() != "step":
            raise ValueError(f"지원하지 않는 스케줄러 타입: {scfg['type']}")
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=int(scfg["step_size"]), gamma=float(scfg["gamma"])
        )
    return optimizer, scheduler


def run_epochs(model, loader, optimizer, scheduler, step_fn, epochs, tag,
               patience=0, min_delta=0.0, log_interval=50):
    """공통 학습 루프 (early stopping + StepLR). epoch 별 평균 손실 리스트를 반환."""
    history = []
    best_loss = math.inf
    bad_epochs = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss, total_items = 0.0, 0
        lr_used = optimizer.param_groups[0]["lr"]
        for step, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss, batch_size = step_fn(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
            total_items += batch_size
            if log_interval and step % log_interval == 0:
                print(f"  [{tag}] epoch {epoch} step {step}: loss={loss.item():.4f}")
        if scheduler is not None:
            scheduler.step()
        avg_loss = total_loss / max(total_items, 1)
        history.append(avg_loss)
        print(f"[{tag}] Epoch {epoch}/{epochs} 평균 손실: {avg_loss:.4f} (lr={lr_used:.2e})")

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"[{tag}] {patience} epoch 동안 개선이 없어 조기 종료합니다 (epoch {epoch}).")
                break
    return history


@torch.no_grad()
def evaluate_iou(model, image_dir, mask_dir, device, image_size=512, threshold=0.5):
    """검증셋 평균 IoU (self-training 라운드 조기 종료 판정용)."""
    transform = image_transform(image_size)
    mask_index = index_by_stem(mask_dir)
    was_training = model.training
    model.eval()
    scores = []
    for img_path in list_images(image_dir):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if stem not in mask_index:
            continue
        img = Image.open(img_path).convert("RGB")
        pred = torch.sigmoid(model(transform(img).unsqueeze(0).to(device)))
        pred_bin = (pred >= threshold).squeeze(0).squeeze(0).cpu()

        gt = Image.open(mask_index[stem]).convert("L").resize(
            (image_size, image_size), resample=Image.NEAREST)
        gt_bin = (transforms.functional.to_tensor(gt) > 0.5).squeeze(0)

        inter = torch.logical_and(pred_bin, gt_bin).sum().item()
        union = torch.logical_or(pred_bin, gt_bin).sum().item()
        scores.append(1.0 if union == 0 else inter / union)
    model.train(was_training)
    return sum(scores) / len(scores) if scores else float("nan")


# ---------------------------------------------------------------------------
# 파이프라인 단계
# ---------------------------------------------------------------------------
def train_segmentation(cfg, device, paths):
    """1단계(약지도 학습) + 2단계(self-training K 라운드)."""
    scfg = cfg["train"]["segmentation"]
    stcfg = cfg["self_training"]
    image_size = cfg["data"]["image_size"]

    dataset = SegDataset(
        paths["train_image_dir"],
        mask_dir=paths["train_mask_dir"],
        bbox_file=paths["bbox_file"],
        image_size=image_size,
    )
    print(f"세그멘테이션 학습 이미지 수: {len(dataset)}")
    loader = make_loader(cfg, dataset)

    model = build_unet(cfg, "segmentation", device)
    print(f"세그멘테이션 네트워크 파라미터 수: {count_parameters(model):,}")
    loss_funcs = LossFunctions.from_config(cfg)
    optimizer, scheduler = make_optimizer(cfg, model)

    def seg_step(batch):
        imgs, masks = batch
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        return loss_funcs.segmentation_loss(preds, masks), imgs.size(0)

    print("\n[1단계] 약지도 세그멘테이션 학습을 시작합니다...")
    history = {"stage1": run_epochs(
        model, loader, optimizer, scheduler, seg_step,
        epochs=scfg["epochs"], tag="seg-stage1",
        patience=scfg["early_stopping_patience"], min_delta=float(scfg["min_delta"]),
        log_interval=cfg["train"]["log_interval"],
    )}
    seg_path = os.path.join(paths["checkpoint_dir"], "segmentation.pth")
    torch.save(model.state_dict(), seg_path)
    print(f"[1단계] 완료 → {seg_path}")

    # ---- 2단계: self-training K 라운드 ----
    rounds_run = 0
    iou_log = []
    if stcfg["enabled"] and int(stcfg["rounds"]) > 0:
        prev_iou = None
        use_val = bool(stcfg["validate"]) and bool(stcfg["val_image_dir"])
        if use_val:
            prev_iou = evaluate_iou(model, stcfg["val_image_dir"], stcfg["val_mask_dir"],
                                    device, image_size, stcfg["threshold"])
            iou_log.append({"round": 0, "val_iou": prev_iou})
            print(f"[2단계] 라운드 0 (약지도) 검증 IoU: {prev_iou:.4f}")

        for k in range(1, int(stcfg["rounds"]) + 1):
            print(f"\n[2단계] Self-training 라운드 {k}/{stcfg['rounds']}")
            round_mask_dir = os.path.join(paths["work_dir"], f"pred_masks_round{k}")
            st = SelfTraining(model, device, threshold=stcfg["threshold"], image_size=image_size)
            st.generate_masks(dataset.image_paths, round_mask_dir)
            print(f"  pseudo-label 마스크 생성 완료 → {round_mask_dir}")

            round_dataset = SegDataset(paths["train_image_dir"], mask_dir=round_mask_dir,
                                       image_size=image_size)
            round_loader = make_loader(cfg, round_dataset)
            # 라운드마다 옵티마이저·스케줄러를 초기 학습률에서 다시 시작
            optimizer, scheduler = make_optimizer(cfg, model)
            history[f"self_training_round{k}"] = run_epochs(
                model, round_loader, optimizer, scheduler, seg_step,
                epochs=stcfg["epochs_per_round"], tag=f"seg-round{k}",
                patience=scfg["early_stopping_patience"], min_delta=float(scfg["min_delta"]),
                log_interval=cfg["train"]["log_interval"],
            )
            rounds_run = k
            torch.save(model.state_dict(),
                       os.path.join(paths["checkpoint_dir"], f"segmentation_round{k}.pth"))

            if use_val:
                iou = evaluate_iou(model, stcfg["val_image_dir"], stcfg["val_mask_dir"],
                                   device, image_size, stcfg["threshold"])
                iou_log.append({"round": k, "val_iou": iou})
                print(f"[2단계] 라운드 {k} 검증 IoU: {iou:.4f}")
                if prev_iou is not None and (iou - prev_iou) < float(stcfg["min_iou_gain"]):
                    print(f"[2단계] IoU 개선폭 {iou - prev_iou:+.4f} < {stcfg['min_iou_gain']} "
                          f"→ 라운드 {k} 에서 종료합니다.")
                    prev_iou = iou
                    break
                prev_iou = iou

        ft_path = os.path.join(paths["checkpoint_dir"], "segmentation_ft.pth")
        torch.save(model.state_dict(), ft_path)
        print(f"[2단계] Self-training 완료 ({rounds_run} 라운드) → {ft_path}")
    else:
        print("\n[2단계] self_training.enabled=false → self-training 을 건너뜁니다.")

    return model, dataset, {"loss_history": history, "iou_log": iou_log, "rounds_run": rounds_run}


def generate_final_masks(cfg, device, model, dataset, paths):
    """3단계 입력이 될 최종 예측 마스크 생성."""
    print("\n[3단계-준비] 최종 세그멘테이션 모델로 전체 학습 이미지의 마스크를 예측합니다...")
    st = SelfTraining(model, device, threshold=cfg["self_training"]["threshold"],
                      image_size=cfg["data"]["image_size"])
    mask_index = st.generate_masks(dataset.image_paths, paths["pred_mask_dir"])
    print(f"  마스크 {len(mask_index)}장 생성 완료 → {paths['pred_mask_dir']}")
    return mask_index


def generate_targets(cfg, device, dataset, mask_index, paths):
    """3단계: Stable Diffusion teacher 로 pseudo-GT 생성."""
    if not cfg["teacher"]["enabled"]:
        existing = index_by_stem(paths["pseudo_gt_dir"])
        if not existing:
            raise FileNotFoundError(
                "teacher.enabled=false 인데 pseudo-GT 가 없습니다.\n"
                f"복원 학습 타겟을 {paths['pseudo_gt_dir']} 에 미리 준비하거나 "
                "teacher.enabled=true 로 두세요."
            )
        print(f"\n[3단계] teacher.enabled=false → 기존 pseudo-GT {len(existing)}장을 사용합니다.")
        return existing

    print(f"\n[3단계] Stable Diffusion inpainting 으로 pseudo-GT 생성 "
          f"({resolve_checkpoint(cfg)}, {cfg['teacher']['num_inference_steps']} steps, "
          f"guidance {cfg['teacher']['guidance_scale']})")
    return generate_pseudo_gt(
        cfg, device, dataset.image_paths, mask_index, paths["pseudo_gt_dir"],
        skip_existing=cfg["teacher"]["skip_existing"],
    )


def train_restoration(cfg, device, paths):
    """4단계: student 복원 네트워크 증류 학습 (마스크 가중 L1)."""
    rcfg = cfg["train"]["restoration"]
    dataset = RestDataset(
        paths["train_image_dir"], paths["pred_mask_dir"], paths["pseudo_gt_dir"],
        image_size=cfg["data"]["image_size"],
    )
    print(f"\n[4단계] 복원 네트워크 학습 시작 (학습 쌍 {len(dataset)}개, "
          f"λ={cfg['loss']['lambda_mask']})")
    loader = make_loader(cfg, dataset)

    model = build_unet(cfg, "restoration", device)
    print(f"복원 네트워크 파라미터 수: {count_parameters(model):,}")
    loss_funcs = LossFunctions.from_config(cfg)
    optimizer, scheduler = make_optimizer(cfg, model)

    def rest_step(batch):
        inputs, targets, masks = batch
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
        outputs = model(inputs)
        return loss_funcs.restoration_loss(outputs, targets, masks), inputs.size(0)

    history = run_epochs(
        model, loader, optimizer, scheduler, rest_step,
        epochs=rcfg["epochs"], tag="restoration",
        patience=rcfg["early_stopping_patience"], min_delta=float(rcfg["min_delta"]),
        log_interval=cfg["train"]["log_interval"],
    )
    rest_path = os.path.join(paths["checkpoint_dir"], "restoration.pth")
    torch.save(model.state_dict(), rest_path)
    print(f"[4단계] 완료 → {rest_path}")
    return model, {"loss_history": history}


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------
def resolve_paths(cfg):
    p = cfg["paths"]
    dataset_dir = p["dataset_dir"]
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"데이터셋 디렉토리가 존재하지 않습니다: {dataset_dir}")

    train_image_dir = os.path.join(dataset_dir, p["train_image_subdir"])
    train_mask_dir = os.path.join(dataset_dir, p["train_mask_subdir"])
    bbox_file = os.path.join(dataset_dir, p["bbox_file"]) if p["bbox_file"] else None

    paths = {
        "dataset_dir": dataset_dir,
        "train_image_dir": train_image_dir,
        "train_mask_dir": train_mask_dir if os.path.isdir(train_mask_dir) else None,
        "bbox_file": bbox_file if bbox_file and os.path.isfile(bbox_file) else None,
        "checkpoint_dir": ensure_dir(p["checkpoint_dir"]),
        "work_dir": ensure_dir(p["work_dir"]),
        "output_dir": ensure_dir(p["output_dir"]),
    }
    paths["pred_mask_dir"] = ensure_dir(os.path.join(paths["work_dir"], "pred_masks"))
    paths["pseudo_gt_dir"] = ensure_dir(os.path.join(paths["work_dir"], "pseudo_gt"))
    return paths


def train(cfg):
    """전체 파이프라인 실행."""
    set_global_seed(cfg["seed"], deterministic=cfg["deterministic"])
    device = resolve_device(cfg["device"])
    print(f"사용 장치: {device} / 시드: {cfg['seed']}")

    paths = resolve_paths(cfg)
    if paths["train_mask_dir"] is None and paths["bbox_file"] is None:
        print("[경고] 마스크 디렉토리와 bbox 파일이 모두 없습니다. 라벨이 빈 마스크로 채워집니다.")

    seg_model, seg_dataset, seg_stats = train_segmentation(cfg, device, paths)
    mask_index = generate_final_masks(cfg, device, seg_model, seg_dataset, paths)

    # teacher 로드 전에 세그멘테이션 모델의 GPU 메모리를 해제
    seg_model.to("cpu")
    del seg_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generate_targets(cfg, device, seg_dataset, mask_index, paths)
    _, rest_stats = train_restoration(cfg, device, paths)

    # 실행에 사용된 설정과 요약을 함께 저장 (재현성 기록)
    cfg.dump(os.path.join(paths["output_dir"], "used_config.yaml"))
    summary = {
        "seed": cfg["seed"],
        "teacher": resolve_checkpoint(cfg) if cfg["teacher"]["enabled"] else None,
        "lambda_mask": cfg["loss"]["lambda_mask"],
        "use_attention": cfg["model"]["use_attention"],
        "self_training_rounds_run": seg_stats["rounds_run"],
        "val_iou_log": seg_stats["iou_log"],
        "segmentation_loss_history": seg_stats["loss_history"],
        "restoration_loss_history": rest_stats["loss_history"],
    }
    summary_path = os.path.join(paths["output_dir"], "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n학습 완료. 요약: {summary_path}, 가중치: {paths['checkpoint_dir']}/")
    return summary
