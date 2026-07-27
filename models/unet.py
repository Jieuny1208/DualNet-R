# models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionBlock


# U-Net 구조 정의 (Down/Up 컨볼루션 블록 포함, Attention 게이트 사용)
class ConvBlock(nn.Module):
    """3x3 Conv + BatchNorm + ReLU 를 2회 반복하는 기본 블록."""

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
    """Attention U-Net.

    - 세그멘테이션 네트워크: in_channels=3, out_channels=1 (BCEWithLogits 용 로짓 출력)
    - 복원(student) 네트워크: in_channels=4 (RGB ⊕ mask), out_channels=3 (tanh 출력)

    features=[64,128,256,512] + bottleneck 1024 기준 파라미터 수는 약 31.4M.
    use_attention=False 로 두면 attention 게이트를 우회한 순수 U-Net이 되며,
    ablation 실험(w/o attention)에 사용한다.
    """

    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512),
                 use_attention=True):
        super().__init__()
        features = list(features)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.use_attention = use_attention

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
        self.att_blocks = nn.ModuleList() if use_attention else None
        self.dec_blocks = nn.ModuleList()
        curr_channels = prev_channels * 2  # bottleneck output channels
        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(curr_channels, feat, kernel_size=2, stride=2))
            if use_attention:
                # additive attention gate, F_int = F/2
                self.att_blocks.append(AttentionBlock(F_g=feat, F_l=feat, F_int=max(feat // 2, 1)))
            self.dec_blocks.append(ConvBlock(feat * 2, feat))
            curr_channels = feat

        self.final_conv = nn.Conv2d(curr_channels, out_channels, kernel_size=1)
        # 복원 모델(3채널 출력)의 경우에만 출력에 tanh 활성화 적용 ([-1,1] 범위)
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
            skip = skips[skip_idx]
            x_up = self.upconvs[i](x)
            # 크기 불일치 시 보간 조정
            if x_up.shape[2:] != skip.shape[2:]:
                x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x_skip = self.att_blocks[i](x_up, skip) if self.use_attention else skip
            x = torch.cat((x_up, x_skip), dim=1)
            x = self.dec_blocks[i](x)
        x = self.final_conv(x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x


def build_unet(cfg, role, device=None):
    """config 기반 U-Net 생성 헬퍼.

    role: "segmentation" (3ch → 1ch) 또는 "restoration" (4ch → 3ch)
    """
    if role == "segmentation":
        in_channels, out_channels = 3, 1
    elif role == "restoration":
        in_channels, out_channels = 4, 3
    else:
        raise ValueError(f"알 수 없는 role: {role} (segmentation | restoration)")

    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=cfg["model"]["features"],
        use_attention=cfg["model"]["use_attention"],
    )
    if device is not None:
        model = model.to(device)
    return model
