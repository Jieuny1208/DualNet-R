# models/self_training.py
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.helpers import ensure_dir

# ���׸����̼� �� Self-Training ���� Ŭ���� (���� ����ũ ����)
class SelfTraining:
    def __init__(self, model, device, threshold=0.5):
        self.model = model
        self.device = device
        self.threshold = threshold
        # �Է� �̹����� ���� ��ȯ (�� �н� �� ���� ����ȭ�� ����)
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
                # ���� ����� Sigmoid ���� �� ����ȭ
                pred_prob = torch.sigmoid(pred)
                # �ʿ� �� ���� �̹��� ũ�⿡ �°� ����
                if pred_prob.shape[2:] != img_tensor.shape[2:]:
                    pred_prob = F.interpolate(pred_prob, size=img_tensor.shape[2:], mode='bilinear', align_corners=True)
                mask = (pred_prob >= self.threshold).cpu().numpy().astype('uint8')
            mask = mask[0, 0] * 255  # 2D array (H,W) �� {0,255}
            mask_img = Image.fromarray(mask, mode='L')
            # �Է� �̹����� ������ �̸����� ����ũ ����
            mask_img.save(f"{output_dir}/{os.path.basename(img_path)}")
        # (�ʿ� ��) self.model.train() ȣ���Ͽ� �н� ���� ���� ����
