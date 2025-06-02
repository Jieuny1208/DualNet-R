# models/loss.py
import torch
import torch.nn as nn

# 손실 함수 클래스 정의 (세그멘테이션 + 복원)
class LossFunctions:
    def __init__(self, lambda_seg=1.0, lambda_rest=1.0):
        self.lambda_seg = lambda_seg
        self.lambda_rest = lambda_rest
        # BCEWithLogitsLoss: 세그멘테이션 손실
        self.seg_loss_fn = nn.BCEWithLogitsLoss()
        # L1 Loss: 복원 손실
        self.rest_loss_fn = nn.L1Loss()

    def segmentation_loss(self, pred_mask, gt_mask):
        # 세그멘테이션 손실 계산
        return self.seg_loss_fn(pred_mask, gt_mask)

    def restoration_loss(self, pred_img, gt_img):
        # 복원 손실 계산
        return self.rest_loss_fn(pred_img, gt_img)

    def total_loss(self, pred_mask, gt_mask, pred_img, gt_img):
        # 필요 시 두 가지 손실을 결합한 총 손실 계산
        seg_loss = self.seg_loss_fn(pred_mask, gt_mask)
        rest_loss = self.rest_loss_fn(pred_img, gt_img)
        return self.lambda_seg * seg_loss + self.lambda_rest * rest_loss
