import os
import argparse
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms

from models_unet import UNet  # 필요에 따라 사용되는 모델 클래스 (예: U-Net)

# PSNR 및 SSIM 기준값 (필요시 변경 가능)
PSNR_THRESHOLD = 20.0
SSIM_THRESHOLD = 0.80

def load_model(checkpoint_path, device):
    """모델을 생성하고 체크포인트를 불러오는 함수."""
    # 모델 인스턴스 생성 (입력 채널 3, 출력 채널 3인 U-Net 예시)
    model = UNet() if hasattr(UNet, "__call__") else UNet
    try:
        model = UNet(in_channels=3, out_channels=3)
    except Exception:
        # 인자없이 생성 가능한 경우
        model = UNet()
    model = model.to(device)
    model.eval()
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # state_dict 키 확인 후 로드
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        # 체크포인트 자체가 state_dict인 경우
        model.load_state_dict(checkpoint)
    return model

def process_image(image_path, model, device):
    """단일 이미지에 대해 모델 추론을 수행하고 복원된 이미지를 PIL 형태로 반환."""
    # 이미지 불러오기 및 텐서로 변환
    img = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()  # [0, 255] -> [0.0, 1.0] 범위로 변환
    input_tensor = to_tensor(img).unsqueeze(0).to(device)  # 배치 차원 추가 후 장치로 이동

    # 모델 추론 (gradient 계산 비활성화)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # 출력 텐서 후처리: CPU로 이동 및 이미지 범위로 변환
    output_tensor = output_tensor.squeeze(0).cpu()  # [C, H, W]
    # 출력이 이미 [0.0,1.0] 범위라고 가정하고 0~255로 변환
    output_np = output_tensor.numpy()
    # 값 범위를 [0,255]로 클램프하고 uint8 형 변환
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    # [C, H, W] -> [H, W, C]로 축 변환하여 PIL 이미지 생성
    output_img = Image.fromarray(np.transpose(output_np, (1, 2, 0)))
    return output_img

def main():
    parser = argparse.ArgumentParser(description="차량 파손 이미지 복원 및 수리 가능 여부 판단")
    parser.add_argument("--input", type=str, required=True, help="입력 이미지 경로나 디렉토리 (손상된 이미지)")
    parser.add_argument("--output", type=str, help="복원된 이미지를 저장할 경로나 디렉토리")
    parser.add_argument("--model", type=str, required=True, help="학습된 모델 가중치 파일 경로")
    parser.add_argument("--gt", type=str, help="원본 이미지(정상 상태) 경로 또는 디렉토리 (선택 사항)")
    parser.add_argument("--gpu", action="store_true", help="GPU 사용 여부 (지정 시 가능할 경우 GPU 사용)")
    args = parser.parse_args()

    # 장치 설정 (GPU 사용 옵션에 따라)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    # 모델 로드
    model = load_model(args.model, device)

    # 입력 경로가 디렉토리인지 파일인지 확인
    input_paths = []
    if os.path.isdir(args.input):
        # 디렉토리 내의 이미지 파일 목록 추출 (jpg, png 등 확장자 필터링)
        for fname in os.listdir(args.input):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                input_paths.append(os.path.join(args.input, fname))
        input_paths.sort()
    else:
        input_paths.append(args.input)

    # 출력 경로 준비
    save_single_file = False
    output_dir = None
    if args.output:
        # 출력이 지정된 경우
        if len(input_paths) > 1:
            # 입력이 여러 개인 경우 output을 디렉토리로 간주
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)
        else:
            # 한 개의 입력에 대해 단일 파일 경로로 저장
            if args.output.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                save_single_file = True
            else:
                # 파일명이 아니면 디렉토리로 간주하여 생성
                output_dir = args.output
                os.makedirs(output_dir, exist_ok=True)

    # 원본 이미지 경로가 디렉토리인지 확인 (복수 입력 대응)
    gt_dir = None
    single_gt_path = None
    if args.gt:
        if os.path.isdir(args.gt):
            gt_dir = args.gt
        else:
            single_gt_path = args.gt

    # 각 입력 이미지에 대해 추론 및 결과 처리
    for img_path in input_paths:
        # 모델로 복원된 이미지 얻기
        restored_img = process_image(img_path, model, device)

        # 복원 이미지 저장 (디렉토리 또는 파일로)
        if output_dir:
            # 출력 디렉토리가 주어졌을 경우, 원본 파일명에 "_restored" 접미사를 붙여 저장
            base_name, ext = os.path.splitext(os.path.basename(img_path))
            save_path = os.path.join(output_dir, f"{base_name}_restored{ext if ext else '.png'}")
            restored_img.save(save_path)
        elif save_single_file:
            # 출력이 단일 파일로 지정된 경우
            restored_img.save(args.output)
        else:
            # 출력 경로 미지정 시, 입력 파일명에 "_restored"를 붙여 동일 위치에 저장
            base_name, ext = os.path.splitext(img_path)
            save_path = f"{base_name}_restored{ext if ext else '.png'}"
            restored_img.save(save_path)

        # 원본 이미지(정상 이미지)가 제공된 경우 PSNR 및 SSIM 계산
        if args.gt:
            original_img = None
            if gt_dir:
                # GT 디렉토리가 있는 경우 입력 파일명에 해당하는 GT 이미지 사용
                gt_path = os.path.join(gt_dir, os.path.basename(img_path))
                if os.path.exists(gt_path):
                    original_img = Image.open(gt_path).convert("RGB")
                else:
                    # 만약 정확한 이름으로 GT가 없으면, 확장자 제외 비교 등 시도
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    # 디렉토리 내에서 동일한 이름을 가진 파일 검색
                    candidates = [f for f in os.listdir(gt_dir) if f.startswith(base_name)]
                    if candidates:
                        original_img = Image.open(os.path.join(gt_dir, candidates[0])).convert("RGB")
            else:
                # 단일 GT 경로가 주어진 경우 (입력이 하나일 때만 유의미)
                original_img = Image.open(single_gt_path).convert("RGB")

            if original_img is not None:
                # numpy 배열로 변환
                orig_np = np.array(original_img)
                restored_np = np.array(restored_img)
                # PSNR, SSIM 계산 (컬러 이미지이므로 multichannel=True)
                psnr_val = peak_signal_noise_ratio(orig_np, restored_np, data_range=255)
                ssim_val = structural_similarity(orig_np, restored_np, multichannel=True, data_range=255)
                # 기준에 따라 결과 출력
                result_text = "수리 가능" if (psnr_val >= PSNR_THRESHOLD and ssim_val >= SSIM_THRESHOLD) else "수리 불가"
                if len(input_paths) > 1:
                    print(f"{os.path.basename(img_path)}: {result_text}")
                else:
                    print(result_text)
            else:
                # 원본 이미지가 존재하지 않아 평가하지 못한 경우
                if len(input_paths) > 1:
                    print(f"{os.path.basename(img_path)}: 원본 이미지 없음 (평가 생략)")
                else:
                    print("원본 이미지가 제공되지 않아 PSNR/SSIM 평가를 생략합니다.")

if __name__ == "__main__":
    main()
