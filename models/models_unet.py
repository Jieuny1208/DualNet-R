# models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import AttentionBlock

# U-Net 건축 정의 (Down/Up 컨볼루션 블록 포함, Attention 게이트 사용)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 인코더 (다운샘플링 경로)
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = in_channels
        for feat in features:
            self.enc_blocks.append(ConvBlock(prev_channels, feat))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = feat
        # 보틀넥 (최심층)
        self.bottleneck = ConvBlock(prev_channels, prev_channels * 2)
        # 디코더 (업샘플링 경로)
        self.upconvs = nn.ModuleList()
        self.att_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        curr_channels = prev_channels * 2  # bottleneck output channels
        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(curr_channels, feat, kernel_size=2, stride=2))
            self.att_blocks.append(AttentionBlock(F_g=feat, F_l=feat, F_int=max(feat//2, 1)))
            self.dec_blocks.append(ConvBlock(feat * 2, feat))
            curr_channels = feat
        self.final_conv = nn.Conv2d(curr_channels, out_channels, kernel_size=1)
        # 복원 모델의 경우 출력에 tanh 활성화 적용
        self.use_tanh = (out_channels == 3)

    def forward(self, x):
        skips = []
        # 인코더: 컨볼루션 + 풀링, skip 연결 저장
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        # 보틀넥
        x = self.bottleneck(x)
        # 디코더: 업컨볼루션 + Attention 게이트 + 병합 + 컨볼루션
        for i in range(len(self.upconvs)):
            skip_idx = len(skips) - 1 - i
            x_up = self.upconvs[i](x)
            # 크기 불일치 시 보간 조정
            if x_up.shape[2:] != skips[skip_idx].shape[2:]:
                x_up = F.interpolate(x_up, size=skips[skip_idx].shape[2:], mode='bilinear', align_corners=True)
            x_att = self.att_blocks[i](x_up, skips[skip_idx])
            x = torch.cat((x_up, x_att), dim=1)
            x = self.dec_blocks[i](x)
        x = self.final_conv(x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x
