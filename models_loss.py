# models/loss.py
import torch
import torch.nn as nn

# �ս� �Լ� Ŭ���� ���� (���׸����̼� + ����)
class LossFunctions:
    def __init__(self, lambda_seg=1.0, lambda_rest=1.0):
        self.lambda_seg = lambda_seg
        self.lambda_rest = lambda_rest
        # BCEWithLogitsLoss: ���׸����̼� �ս�
        self.seg_loss_fn = nn.BCEWithLogitsLoss()
        # L1 Loss: ���� �ս�
        self.rest_loss_fn = nn.L1Loss()

    def segmentation_loss(self, pred_mask, gt_mask):
        # ���׸����̼� �ս� ���
        return self.seg_loss_fn(pred_mask, gt_mask)

    def restoration_loss(self, pred_img, gt_img):
        # ���� �ս� ���
        return self.rest_loss_fn(pred_img, gt_img)

    def total_loss(self, pred_mask, gt_mask, pred_img, gt_img):
        # �ʿ� �� �� ���� �ս��� ������ �� �ս� ���
        seg_loss = self.seg_loss_fn(pred_mask, gt_mask)
        rest_loss = self.rest_loss_fn(pred_img, gt_img)
        return self.lambda_seg * seg_loss + self.lambda_rest * rest_loss
