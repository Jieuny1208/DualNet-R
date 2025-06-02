import os
import argparse
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms

from models_unet import UNet  # �ʿ信 ���� ���Ǵ� �� Ŭ���� (��: U-Net)

# PSNR �� SSIM ���ذ� (�ʿ�� ���� ����)
PSNR_THRESHOLD = 20.0
SSIM_THRESHOLD = 0.80

def load_model(checkpoint_path, device):
    """���� �����ϰ� üũ����Ʈ�� �ҷ����� �Լ�."""
    # �� �ν��Ͻ� ���� (�Է� ä�� 3, ��� ä�� 3�� U-Net ����)
    model = UNet() if hasattr(UNet, "__call__") else UNet
    try:
        model = UNet(in_channels=3, out_channels=3)
    except Exception:
        # ���ھ��� ���� ������ ���
        model = UNet()
    model = model.to(device)
    model.eval()
    # üũ����Ʈ �ε�
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # state_dict Ű Ȯ�� �� �ε�
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        # üũ����Ʈ ��ü�� state_dict�� ���
        model.load_state_dict(checkpoint)
    return model

def process_image(image_path, model, device):
    """���� �̹����� ���� �� �߷��� �����ϰ� ������ �̹����� PIL ���·� ��ȯ."""
    # �̹��� �ҷ����� �� �ټ��� ��ȯ
    img = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()  # [0, 255] -> [0.0, 1.0] ������ ��ȯ
    input_tensor = to_tensor(img).unsqueeze(0).to(device)  # ��ġ ���� �߰� �� ��ġ�� �̵�

    # �� �߷� (gradient ��� ��Ȱ��ȭ)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # ��� �ټ� ��ó��: CPU�� �̵� �� �̹��� ������ ��ȯ
    output_tensor = output_tensor.squeeze(0).cpu()  # [C, H, W]
    # ����� �̹� [0.0,1.0] ������� �����ϰ� 0~255�� ��ȯ
    output_np = output_tensor.numpy()
    # �� ������ [0,255]�� Ŭ�����ϰ� uint8 �� ��ȯ
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    # [C, H, W] -> [H, W, C]�� �� ��ȯ�Ͽ� PIL �̹��� ����
    output_img = Image.fromarray(np.transpose(output_np, (1, 2, 0)))
    return output_img

def main():
    parser = argparse.ArgumentParser(description="���� �ļ� �̹��� ���� �� ���� ���� ���� �Ǵ�")
    parser.add_argument("--input", type=str, required=True, help="�Է� �̹��� ��γ� ���丮 (�ջ�� �̹���)")
    parser.add_argument("--output", type=str, help="������ �̹����� ������ ��γ� ���丮")
    parser.add_argument("--model", type=str, required=True, help="�н��� �� ����ġ ���� ���")
    parser.add_argument("--gt", type=str, help="���� �̹���(���� ����) ��� �Ǵ� ���丮 (���� ����)")
    parser.add_argument("--gpu", action="store_true", help="GPU ��� ���� (���� �� ������ ��� GPU ���)")
    args = parser.parse_args()

    # ��ġ ���� (GPU ��� �ɼǿ� ����)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    # �� �ε�
    model = load_model(args.model, device)

    # �Է� ��ΰ� ���丮���� �������� Ȯ��
    input_paths = []
    if os.path.isdir(args.input):
        # ���丮 ���� �̹��� ���� ��� ���� (jpg, png �� Ȯ���� ���͸�)
        for fname in os.listdir(args.input):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                input_paths.append(os.path.join(args.input, fname))
        input_paths.sort()
    else:
        input_paths.append(args.input)

    # ��� ��� �غ�
    save_single_file = False
    output_dir = None
    if args.output:
        # ����� ������ ���
        if len(input_paths) > 1:
            # �Է��� ���� ���� ��� output�� ���丮�� ����
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)
        else:
            # �� ���� �Է¿� ���� ���� ���� ��η� ����
            if args.output.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                save_single_file = True
            else:
                # ���ϸ��� �ƴϸ� ���丮�� �����Ͽ� ����
                output_dir = args.output
                os.makedirs(output_dir, exist_ok=True)

    # ���� �̹��� ��ΰ� ���丮���� Ȯ�� (���� �Է� ����)
    gt_dir = None
    single_gt_path = None
    if args.gt:
        if os.path.isdir(args.gt):
            gt_dir = args.gt
        else:
            single_gt_path = args.gt

    # �� �Է� �̹����� ���� �߷� �� ��� ó��
    for img_path in input_paths:
        # �𵨷� ������ �̹��� ���
        restored_img = process_image(img_path, model, device)

        # ���� �̹��� ���� (���丮 �Ǵ� ���Ϸ�)
        if output_dir:
            # ��� ���丮�� �־����� ���, ���� ���ϸ� "_restored" ���̻縦 �ٿ� ����
            base_name, ext = os.path.splitext(os.path.basename(img_path))
            save_path = os.path.join(output_dir, f"{base_name}_restored{ext if ext else '.png'}")
            restored_img.save(save_path)
        elif save_single_file:
            # ����� ���� ���Ϸ� ������ ���
            restored_img.save(args.output)
        else:
            # ��� ��� ������ ��, �Է� ���ϸ� "_restored"�� �ٿ� ���� ��ġ�� ����
            base_name, ext = os.path.splitext(img_path)
            save_path = f"{base_name}_restored{ext if ext else '.png'}"
            restored_img.save(save_path)

        # ���� �̹���(���� �̹���)�� ������ ��� PSNR �� SSIM ���
        if args.gt:
            original_img = None
            if gt_dir:
                # GT ���丮�� �ִ� ��� �Է� ���ϸ� �ش��ϴ� GT �̹��� ���
                gt_path = os.path.join(gt_dir, os.path.basename(img_path))
                if os.path.exists(gt_path):
                    original_img = Image.open(gt_path).convert("RGB")
                else:
                    # ���� ��Ȯ�� �̸����� GT�� ������, Ȯ���� ���� �� �� �õ�
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    # ���丮 ������ ������ �̸��� ���� ���� �˻�
                    candidates = [f for f in os.listdir(gt_dir) if f.startswith(base_name)]
                    if candidates:
                        original_img = Image.open(os.path.join(gt_dir, candidates[0])).convert("RGB")
            else:
                # ���� GT ��ΰ� �־��� ��� (�Է��� �ϳ��� ���� ���ǹ�)
                original_img = Image.open(single_gt_path).convert("RGB")

            if original_img is not None:
                # numpy �迭�� ��ȯ
                orig_np = np.array(original_img)
                restored_np = np.array(restored_img)
                # PSNR, SSIM ��� (�÷� �̹����̹Ƿ� multichannel=True)
                psnr_val = peak_signal_noise_ratio(orig_np, restored_np, data_range=255)
                ssim_val = structural_similarity(orig_np, restored_np, multichannel=True, data_range=255)
                # ���ؿ� ���� ��� ���
                result_text = "���� ����" if (psnr_val >= PSNR_THRESHOLD and ssim_val >= SSIM_THRESHOLD) else "���� �Ұ�"
                if len(input_paths) > 1:
                    print(f"{os.path.basename(img_path)}: {result_text}")
                else:
                    print(result_text)
            else:
                # ���� �̹����� �������� �ʾ� ������ ���� ���
                if len(input_paths) > 1:
                    print(f"{os.path.basename(img_path)}: ���� �̹��� ���� (�� ����)")
                else:
                    print("���� �̹����� �������� �ʾ� PSNR/SSIM �򰡸� �����մϴ�.")

if __name__ == "__main__":
    main()
