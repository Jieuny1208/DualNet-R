# models/loss.py
import torch
import torch.nn as nn


class MaskWeightedL1Loss(nn.Module):
    r"""마스크 가중 L1 복원(증류) 손실.

        L_distill(θ) = || (1 + λM) ⊙ ( S_θ(x ⊕ M) − ŷ ) ||_1

    - M: 손상 영역 이진 마스크 (N,1,H,W), 채널 방향으로 브로드캐스팅된다.
    - λ(lambda_mask): 손상 영역 내부의 오차 가중치. λ=0 이면 일반 L1 과 동일하다.
    - reduction="mean" 이 기본값이며, 배치/픽셀 수에 무관한 스케일을 유지한다.
    """

    def __init__(self, lambda_mask=1.0, reduction="mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"지원하지 않는 reduction: {reduction}")
        self.lambda_mask = float(lambda_mask)
        self.reduction = reduction

    def forward(self, pred_img, target_img, mask=None):
        diff = torch.abs(pred_img - target_img)
        if mask is not None and self.lambda_mask != 0.0:
            if mask.dim() != diff.dim():
                raise ValueError(
                    f"마스크 차원이 예측값과 다릅니다: mask={tuple(mask.shape)}, pred={tuple(diff.shape)}"
                )
            weight = 1.0 + self.lambda_mask * mask.to(diff.dtype)
            diff = weight * diff
        if self.reduction == "mean":
            return diff.mean()
        if self.reduction == "sum":
            return diff.sum()
        return diff


# 손실 함수 모음 (세그멘테이션 + 복원)
class LossFunctions:
    def __init__(self, lambda_seg=1.0, lambda_rest=1.0, lambda_mask=1.0):
        self.lambda_seg = lambda_seg
        self.lambda_rest = lambda_rest
        self.lambda_mask = lambda_mask
        # BCEWithLogitsLoss: 세그멘테이션 손실 (로짓 입력)
        self.seg_loss_fn = nn.BCEWithLogitsLoss()
        # 마스크 가중 L1: 복원(증류) 손실
        self.rest_loss_fn = MaskWeightedL1Loss(lambda_mask=lambda_mask)

    @classmethod
    def from_config(cls, cfg):
        return cls(lambda_mask=cfg["loss"]["lambda_mask"])

    def segmentation_loss(self, pred_mask, gt_mask):
        # 세그멘테이션 손실 계산
        return self.seg_loss_fn(pred_mask, gt_mask)

    def restoration_loss(self, pred_img, gt_img, mask=None):
        # 복원 손실 계산 (mask 전달 시 손상 영역 가중)
        return self.rest_loss_fn(pred_img, gt_img, mask)

    def total_loss(self, pred_mask, gt_mask, pred_img, gt_img, mask=None):
        # 필요 시 두 가지 손실을 결합한 총 손실 계산
        seg_loss = self.segmentation_loss(pred_mask, gt_mask)
        rest_loss = self.restoration_loss(pred_img, gt_img, mask)
        return self.lambda_seg * seg_loss + self.lambda_rest * rest_loss
