# models/self_training.py
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.helpers import ensure_dir

# 세그멘테이션 모델 Self-Training 보조 클래스 (예측 마스크 생성)
class SelfTraining:
    def __init__(self, model, device, threshold=0.5):
        self.model = model
        self.device = device
        self.threshold = threshold
        # 입력 이미지에 대한 변환 (모델 학습 시 사용된 정규화로 설정)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])

    def generate_masks(self, image_paths, output_dir):
        ensure_dir(output_dir)
        self.model.eval()
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(img_tensor)
                # 예측 결과에 Sigmoid 적용 후 이진화
                pred_prob = torch.sigmoid(pred)
                # 필요 시 원본 이미지 크기에 맞게 보간
                if pred_prob.shape[2:] != img_tensor.shape[2:]:
                    pred_prob = F.interpolate(pred_prob, size=img_tensor.shape[2:], mode='bilinear', align_corners=True)
                mask = (pred_prob >= self.threshold).cpu().numpy().astype('uint8')
            mask = mask[0, 0] * 255  # 2D array (H,W) 값 {0,255}
            mask_img = Image.fromarray(mask, mode='L')
            # 입력 이미지와 동일한 이름으로 마스크 저장
            mask_img.save(f"{output_dir}/{os.path.basename(img_path)}")
        # (필요 시) self.model.train() 호출하여 학습 모드로 복귀 가능
