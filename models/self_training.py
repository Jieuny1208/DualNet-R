# models/self_training.py
import os

import torch
import torch.nn.functional as F
from PIL import Image

from utils.helpers import ensure_dir, image_transform


# 세그멘테이션 모델 Self-Training 보조 클래스 (예측 마스크 = pseudo-label 생성)
class SelfTraining:
    """현재 세그멘테이션 모델로 pseudo-label 마스크를 생성한다.

    확률맵 sigmoid → threshold(기본 0.5) 이진화. 마스크는 원본 해상도로 되돌린 뒤
    PNG(무손실)로 저장하므로, 파일명은 원본 이미지의 stem 을 그대로 따른다.
    """

    def __init__(self, model, device, threshold=0.5, image_size=512):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.image_size = image_size
        # 학습 시와 동일한 전처리 ([-1,1] 정규화 + 지정 해상도 리사이즈)
        self.transform = image_transform(image_size)

    @torch.no_grad()
    def predict_prob(self, img):
        """PIL RGB 이미지 → 원본 해상도의 확률맵 텐서 (1,1,H,W)."""
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        pred = self.model(img_tensor)
        pred_prob = torch.sigmoid(pred)
        # 원본 이미지 크기로 복원 (PIL size 는 (W,H))
        target_hw = (img.size[1], img.size[0])
        if pred_prob.shape[2:] != target_hw:
            pred_prob = F.interpolate(pred_prob, size=target_hw, mode="bilinear", align_corners=True)
        return pred_prob

    @torch.no_grad()
    def predict_mask(self, img):
        """PIL RGB 이미지 → 이진 마스크 PIL 이미지 (mode='L', 값 {0,255})."""
        pred_prob = self.predict_prob(img)
        mask = (pred_prob >= self.threshold).squeeze().cpu().numpy().astype("uint8") * 255
        return Image.fromarray(mask, mode="L")

    def generate_masks(self, image_paths, output_dir):
        """이미지 목록에 대해 pseudo-label 마스크를 생성·저장하고 {stem: 경로} 를 반환."""
        ensure_dir(output_dir)
        was_training = self.model.training
        self.model.eval()
        saved = {}
        try:
            for img_path in image_paths:
                img = Image.open(img_path).convert("RGB")
                mask_img = self.predict_mask(img)
                stem = os.path.splitext(os.path.basename(img_path))[0]
                out_path = os.path.join(output_dir, f"{stem}.png")
                mask_img.save(out_path)
                saved[stem] = out_path
        finally:
            # 호출 전 모드로 복귀 (이어서 학습하는 경우를 위해)
            self.model.train(was_training)
        return saved
