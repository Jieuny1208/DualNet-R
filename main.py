# # main.py
# import argparse

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="DualNet-R 약지도 손상 복원 프레임워크 실행")
#     parser.add_argument("--mode", choices=["train", "evaluate", "inference"], required=True, help="실행 모드 선택: train(훈련), evaluate(평가), inference(단일 추론)")
#     parser.add_argument("--dataset_dir", type=str, help="훈련 시 사용할 데이터셋 디렉토리 경로")
#     parser.add_argument("--test_dir", type=str, help="평가 시 사용할 테스트 데이터 디렉토리 경로")
#     parser.add_argument("--input", type=str, help="단일 이미지 추론 시 입력 이미지 경로")
#     parser.add_argument("--output_dir", type=str, help="단일 이미지 추론 결과를 저장할 디렉토리 (기본값: output)")
#     parser.add_argument("--seg_model_path", type=str, help="세그멘테이션 모델 가중치 파일 경로 (기본값: checkpoints/segmentation.pth)")
#     parser.add_argument("--rest_model_path", type=str, help="복원 모델 가중치 파일 경로 (기본값: checkpoints/restoration.pth)")
#     parser.add_argument("--epochs_seg", type=int, default=5, help="세그멘테이션 1차 학습 에포크 수 (기본값: 5)")
#     parser.add_argument("--epochs_seg2", type=int, default=2, help="세그멘테이션 Self-Training 2차 학습 에포크 수 (기본값: 2, 0이면 생략)")
#     parser.add_argument("--epochs_rest", type=int, default=10, help="복원 네트워크 학습 에포크 수 (기본값: 10)")
#     parser.add_argument("--batch_size", type=int, default=4, help="훈련 배치 크기 (기본값: 4)")
#     parser.add_argument("--lr", type=float, default=1e-3, help="학습률 (기본값: 1e-3)")
#     args = parser.parse_args()

#     if args.mode == "train":
#         if not args.dataset_dir:
#             parser.error("--mode train requires --dataset_dir")
#         # train.py 모듈 임포트 및 실행
#         import train
#         train.train(args)
#     elif args.mode == "evaluate":
#         if not args.test_dir:
#             parser.error("--mode evaluate requires --test_dir")
#         import evaluate
#         evaluate.evaluate(args)
#     elif args.mode == "inference":
#         if not args.input:
#             parser.error("--mode inference requires --input (image path)")
#         import inference
#         inference.run_inference(args)


# pip install torch torchvision matplotlib scikit-image 해주기
# requirement.txt 파일에도 작성해놨어요~
import os
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

# *** 경로 설정 (Windows 경로 형식 사용 - 필요시 실제 폴더명에 맞게 조정) ***
BASE_DIR = r"C:\Users\lu387\Desktop\Dualnet-R\CarDD_release"
COCO_DIR = os.path.join(BASE_DIR, "CarDD_COCO")        # COCO 형식 데이터 (사용하지 않더라도 경로 정의)
SOD_DIR = os.path.join(BASE_DIR, "CarDD_SOD")          # SOD 데이터 셋 기본 경로
TRAIN_DIR = os.path.join(SOD_DIR, "CarDD-TR")          # 학습용 디렉토리
VAL_DIR   = os.path.join(SOD_DIR, "CarDD-VAL")         # 검증용 디렉토리
TEST_DIR  = os.path.join(SOD_DIR, "CarDD-TE")          # 테스트용 디렉토리

# 하위 폴더 경로 (이미지, 마스크, 엣지) - 실제 폴더명이 대소문자 등 다를 경우 수정
TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, "CarDD-TR-Image")
TRAIN_MASK_DIR = os.path.join(TRAIN_DIR, "CarDD-TR-Mask")
TRAIN_EDGE_DIR = os.path.join(TRAIN_DIR, "CarDD-TR-Edge")

VAL_IMG_DIR = os.path.join(VAL_DIR, "CarDD-VAL-Image")
VAL_MASK_DIR = os.path.join(VAL_DIR, "CarDD-VAL-Mask")
VAL_EDGE_DIR = os.path.join(VAL_DIR, "CarDD-VAL-Edge")

TEST_IMG_DIR = os.path.join(TEST_DIR, "CarDD-TE-Image")
# 테스트 GT 경로 (평가에는 사용 않음)
TEST_MASK_DIR = os.path.join(TEST_DIR, "CarDD-TE-Mask")
TEST_EDGE_DIR = os.path.join(TEST_DIR, "CarDD-TE-Edge")

# *** 하이퍼파라미터 및 기본 설정 ***
IMG_SIZE = (320, 320)       # 학습에 사용할 이미지 해상도 (고정 크기로 조정)
BATCH_SIZE = 4              # 학습 배치 크기
EPOCHS = 10                 # 학습 epoch 횟수 (필요시 증가 가능)
LEARNING_RATE = 1e-4        # 최적화 초기 학습률
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# *** 데이터셋 클래스 정의 ***
class CarDDDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir=None, edge_dir=None, transform=True):
        """
        img_dir: 이미지 폴더 경로
        mask_dir: 마스크(ground truth) 폴더 경로 (없으면 추론용 데이터셋으로 처리)
        edge_dir: 엣지(ground truth) 폴더 경로
        transform: 이미지와 라벨에 대한 전처리 적용 여부
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.edge_dir = edge_dir
        self.transform = transform
        # 지원하는 이미지 확장자 목록
        exts = (".jpg", ".jpeg", ".png", ".bmp") 
        # 이미지 파일 목록 로드
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(exts)]
        self.images.sort()  # 정렬 (선택 사항: 일정한 순서 보장)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        # 이미지 로드 (RGB로 변환)
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        # 마스크/엣지 경로 결정 (마스크/엣지가 제공되지 않은 경우 None 처리)
        mask = None
        edge = None
        if self.mask_dir is not None:
            base_name = os.path.splitext(img_name)[0]  # 확장자 제외한 파일명
            mask_path = os.path.join(self.mask_dir, base_name + ".png")
            if not os.path.isfile(mask_path):
                # 만약 .png 마스크가 없으면 (예: 확장자가 동일한 경우), 이미지와 동일 확장자 시도
                mask_path = os.path.join(self.mask_dir, base_name + os.path.splitext(img_name)[1])
            # 마스크 이미지 로드 (그레이스케일)
            mask = Image.open(mask_path).convert('L')
        if self.edge_dir is not None:
            base_name = os.path.splitext(img_name)[0]
            edge_path = os.path.join(self.edge_dir, base_name + ".png")
            if not os.path.isfile(edge_path):
                edge_path = os.path.join(self.edge_dir, base_name + os.path.splitext(img_name)[1])
            edge = Image.open(edge_path).convert('L')
        # 전처리(transform): 이미지 리사이즈 및 Tensor 변환
        if self.transform:
            # 이미지 리사이즈 (양선형 보간)
            image = image.resize(IMG_SIZE, resample=Image.BILINEAR)
            if mask is not None:
                # 마스크와 엣지는 최근접 보간으로 크기 변경 (픽셀 값 유지)
                mask = mask.resize(IMG_SIZE, resample=Image.NEAREST)
            if edge is not None:
                edge = edge.resize(IMG_SIZE, resample=Image.NEAREST)
        # Tensor 변환
        image_np = np.array(image, dtype=np.float32)  # shape: (H, W, 3)
        # 이미지 픽셀값 [0,1] 정규화
        image_np /= 255.0
        # (H,W,3) -> (3,H,W)로 축변경 후 Tensor로 변환
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))
        # 마스크 및 엣지 Tensor 변환 (있을 경우)
        if mask is not None:
            mask_np = np.array(mask, dtype=np.float32)  # shape: (H, W)
            mask_np /= 255.0  # 0 또는 1로 변환
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (1,H,W)
        else:
            mask_tensor = torch.zeros((1, *IMG_SIZE), dtype=torch.float32)  # dummy (not used)
        if edge is not None:
            edge_np = np.array(edge, dtype=np.float32)
            edge_np /= 255.0
            edge_tensor = torch.from_numpy(edge_np).unsqueeze(0)  # (1,H,W)
        else:
            edge_tensor = torch.zeros((1, *IMG_SIZE), dtype=torch.float32)
        return image_tensor, mask_tensor, edge_tensor

# *** 모델 정의 (Dual U-Net 구조: 마스크 및 엣지 2개 출력) ***
def conv_block(in_channels, out_channels):
    """합성곱 블록: Conv-BN-ReLU 두 번 수행"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class DualNet(nn.Module):
    def __init__(self):
        super(DualNet, self).__init__()
        # 인코더 (다운샘플 경로)
        self.enc_conv0 = conv_block(3, 64)
        self.enc_conv1 = conv_block(64, 128)
        self.enc_conv2 = conv_block(128, 256)
        self.enc_conv3 = conv_block(256, 512)
        self.enc_conv4 = conv_block(512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 디코더 (업샘플 경로)
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv3 = conv_block(1024, 512)   # upconv3 출력(512) + enc_conv3 출력(512) = 입력 1024
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv2 = conv_block(512, 256)    # upconv2 출력(256) + enc_conv2 출력(256) = 512
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv1 = conv_block(256, 128)    # 128+128
        self.upconv0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv0 = conv_block(128, 64)     # 64+64
        # 출력 레이어 (1채널 마스크와 1채널 엣지 맵)
        self.out_seg = nn.Conv2d(64, 1, kernel_size=1)   # 손상 부위 마스크 출력
        self.out_edge = nn.Conv2d(64, 1, kernel_size=1)  # 손상 경계선 출력

    def forward(self, x):
        # 인코더 단계
        x0 = self.enc_conv0(x)              # 크기: [N,64,H, W]
        x1 = self.enc_conv1(self.pool(x0))  # [N,128,H/2,W/2]
        x2 = self.enc_conv2(self.pool(x1))  # [N,256,H/4,W/4]
        x3 = self.enc_conv3(self.pool(x2))  # [N,512,H/8,W/8]
        x4 = self.enc_conv4(self.pool(x3))  # [N,1024,H/16,W/16]
        # 디코더 단계 (업샘플 + 스킵 연결)
        d3 = self.upconv3(x4)               # [N,512, H/8, W/8] (x3 크기로 업샘플)
        # 만약 크기가 맞지 않는다면(홀수 해상도 등) 보간으로 맞춤
        if d3.size(-1) != x3.size(-1) or d3.size(-2) != x3.size(-2):
            d3 = F.interpolate(d3, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = torch.cat((d3, x3), dim=1)     # 채널 concat: [N,1024,H/8,W/8]
        d3 = self.dec_conv3(d3)             # [N,512,H/8,W/8]
        d2 = self.upconv2(d3)               # [N,256,H/4,W/4]
        if d2.size(-1) != x2.size(-1) or d2.size(-2) != x2.size(-2):
            d2 = F.interpolate(d2, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = torch.cat((d2, x2), dim=1)     # [N,512,H/4,W/4]
        d2 = self.dec_conv2(d2)             # [N,256,H/4,W/4]
        d1 = self.upconv1(d2)               # [N,128,H/2,W/2]
        if d1.size(-1) != x1.size(-1) or d1.size(-2) != x1.size(-2):
            d1 = F.interpolate(d1, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = torch.cat((d1, x1), dim=1)     # [N,256,H/2,W/2]
        d1 = self.dec_conv1(d1)             # [N,128,H/2,W/2]
        d0 = self.upconv0(d1)               # [N,64,H,W]
        if d0.size(-1) != x0.size(-1) or d0.size(-2) != x0.size(-2):
            d0 = F.interpolate(d0, size=x0.shape[-2:], mode='bilinear', align_corners=False)
        d0 = torch.cat((d0, x0), dim=1)     # [N,128,H,W]
        d0 = self.dec_conv0(d0)             # [N,64,H,W]
        # 출력 계산 (sigmoid는 적용하지 않고 logits 반환; 손실 계산시 BCEWithLogitsLoss 사용)
        seg_out = self.out_seg(d0)          # [N,1,H,W] - 손상 영역 분할 출력 (로짓값)
        edge_out = self.out_edge(d0)        # [N,1,H,W] - 경계선 출력 (로짓값)
        return seg_out, edge_out

# *** 학습, 평가, 추론 함수 정의 ***
def train_model():
    """모델 학습 실행"""
    # 데이터 로더 준비 (학습용)
    train_dataset = CarDDDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TRAIN_EDGE_DIR, transform=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                              shuffle=True, num_workers=0)
    # 모델 및 최적화 설정
    model = DualNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 손실 함수 (이진 분류이므로 BCEWithLogitsLoss 사용)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    print(f"[학습 시작] 총 Epoch: {EPOCHS}, 배치당 {BATCH_SIZE}개, 학습 이미지 수: {len(train_dataset)}")
    for epoch in range(1, EPOCHS+1):
        running_loss = 0.0
        for i, (images, masks, edges) in enumerate(train_loader, start=1):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            edges = edges.to(DEVICE)
            # 순전파
            seg_out, edge_out = model(images)
            # 손실 계산 (마스크 예측과 엣지 예측 각각 BCE 손실)
            loss_seg = criterion(seg_out, masks)     # 손상 영역 마스크 손실
            loss_edge = criterion(edge_out, edges)   # 경계선 맵 손실
            loss = loss_seg + loss_edge
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # 일정 간격으로 중간 손실 출력
            if i % 100 == 0:
                print(f" - Epoch [{epoch}/{EPOCHS}] Step {i}: total_loss={loss.item():.4f} "
                      f"(seg_loss={loss_seg.item():.4f}, edge_loss={loss_edge.item():.4f})")
        # Epoch 종료 후 평균 손실 출력
        epoch_loss = running_loss / len(train_loader)
        print(f"==> Epoch {epoch} 완료: 평균 손실 = {epoch_loss:.4f}")
    # 학습 완료 후 모델 저장
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    model_path = os.path.join(BASE_DIR, "outputs", "model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[학습 종료] 모델 가중치를 '{model_path}' 경로에 저장했습니다.")

def evaluate_model():
    # """검증 데이터셋에 대한 모델 성능 평가 (IoU 계산)"""
    # 데이터 로더 준비 (검증용)
    val_dataset = CarDDDataset(VAL_IMG_DIR, VAL_MASK_DIR, VAL_EDGE_DIR, transform=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    # 저장된 모델 로드
    model = DualNet().to(DEVICE)
    model_path = os.path.join(BASE_DIR, "outputs", "model_final.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"[평가 시작] 검증 이미지 수: {len(val_dataset)}, 불러온 모델: {model_path}")
    iou_scores = []
    with torch.no_grad():
        for images, masks, edges in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            # 예측 수행
            seg_out, edge_out = model(images)
            # 출력 확률로 변환 및 이진화 (임계값 0.5)
            pred_mask = (torch.sigmoid(seg_out) >= 0.5).cpu().numpy().astype(np.uint8)  # [N,1,H,W] -> 0/1
            gt_mask = masks.cpu().numpy().astype(np.uint8)  # [N,1,H,W] (이미 0/1 값)
            # 배치 차원 N에 대해 IoU 계산
            for j in range(pred_mask.shape[0]):
                pred = pred_mask[j, 0]  # (H,W) 예측 이진 맵
                gt = gt_mask[j, 0]      # (H,W) 실제 이진 맵
                # 교집합 및 합집합 계산
                inter = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                if union == 0:
                    iou = 1.0  # GT도 없고 예측도 없는 경우 IoU=1로 처리
                else:
                    iou = inter / union
                iou_scores.append(iou)
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    print(f"[평가 완료] 평균 IoU = {mean_iou:.4f} (총 {len(iou_scores)}개 이미지에 대해 계산)")

def inference_model():
    """테스트 데이터셋에 대한 추론 실행 (예측 마스크/엣지 이미지 출력)"""
    # 테스트용 모델 로드
    model = DualNet().to(DEVICE)
    model_path = os.path.join(BASE_DIR, "outputs", "model_final.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    # 결과 저장 디렉토리 생성
    output_mask_dir = os.path.join(BASE_DIR, "outputs", "pred_masks")
    output_edge_dir = os.path.join(BASE_DIR, "outputs", "pred_edges")
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_edge_dir, exist_ok=True)
    # 테스트 이미지 목록 확보
    test_images = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    test_images.sort()
    print(f"[추론 시작] 테스트 이미지 수: {len(test_images)}, 모델: {model_path}")
    with torch.no_grad():
        for img_name in test_images:
            # 이미지 로드 및 전처리
            img_path = os.path.join(TEST_IMG_DIR, img_name)
            image = Image.open(img_path).convert('RGB')
            image_resized = image.resize(IMG_SIZE, resample=Image.BILINEAR)
            img_np = np.array(image_resized, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)  # 배치 차원 추가
            # 추론
            seg_out, edge_out = model(img_tensor)
            # 결과 이진화
            seg_prob = torch.sigmoid(seg_out)[0, 0].cpu().numpy()  # (H,W) 확률맵
            edge_prob = torch.sigmoid(edge_out)[0, 0].cpu().numpy()  # (H,W) 확률맵
            seg_pred = (seg_prob >= 0.5).astype(np.uint8) * 255  # 0 또는 255
            edge_pred = (edge_prob >= 0.5).astype(np.uint8) * 255
            # 결과 이미지 저장 (PNG 형식)
            base_name = os.path.splitext(img_name)[0]
            mask_save_path = os.path.join(output_mask_dir, f"{base_name}_pred_mask.png")
            edge_save_path = os.path.join(output_edge_dir, f"{base_name}_pred_edge.png")
            mask_img = Image.fromarray(seg_pred.astype(np.uint8), mode='L')
            edge_img = Image.fromarray(edge_pred.astype(np.uint8), mode='L')
            mask_img.save(mask_save_path)
            edge_img.save(edge_save_path)
    print(f"[추론 완료] 예측 결과 마스크를 '{output_mask_dir}' 폴더에 저장했습니다 (경계선 결과는 '{output_edge_dir}' 폴더).")

# *** 메인 실행부: MODE에 따라 동작 ***
if __name__ == "__main__":
    # 기본 모드 설정 ('train', 'evaluate', 'inference' 중 선택)
    MODE = "train"  # ← 디버깅 시 원하는 모드로 변경 가능
    if MODE == "train":
        train_model()
    elif MODE == "evaluate":
        evaluate_model()
    elif MODE == "inference":
        inference_model()
    else:
        print(f"지원하지 않는 MODE 입니다: {MODE}")
