# models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention ����Ʈ ��� ���� (Attention U-Net ���)
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: ���ڴ� gating ��ȣ, x: ���ڴ� skip Ư¡��
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # psi ũ�Ⱑ x�� �ٸ��� ����
        if psi.shape[2:] != x.shape[2:]:
            psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=True)
        # Attention ����� skip ���
        return x * psi
