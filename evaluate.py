# evaluate.py
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from models.unet import UNet
from utils.helpers import get_device, denormalize

def evaluate(args):
    device = get_device()
    print(f"��� ��ġ: {device}")

    test_dir = args.test_dir
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"�׽�Ʈ ������ ���丮 {test_dir} �� �������� �ʽ��ϴ�.")
    # �׽�Ʈ ���丮 �� ���� ���� Ȯ��
    if os.path.isdir(os.path.join(test_dir, "images")):
        image_dir = os.path.join(test_dir, "images")
    else:
        image_dir = test_dir
    orig_dir = os.path.join(test_dir, "original")
    mask_dir = os.path.join(test_dir, "masks")
    have_orig = os.path.isdir(orig_dir)
    have_mask = os.path.isdir(mask_dir)

    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if os.path.splitext(f)[1].lower() in valid_exts]
    image_paths.sort()

    # �� �ε�
    seg_model = UNet(in_channels=3, out_channels=1).to(device)
    rest_model = UNet(in_channels=4, out_channels=3).to(device)
    seg_weights = args.seg_model_path or os.path.join("checkpoints", "segmentation.pth")
    rest_weights = args.rest_model_path or os.path.join("checkpoints", "restoration.pth")
    seg_model.load_state_dict(torch.load(seg_weights, map_location=device))
    rest_model.load_state_dict(torch.load(rest_weights, map_location=device))
    seg_model.eval()
    rest_model.eval()

    # Stable Diffusion ���������� �ε� (�񱳿�)
    pipe = None
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
        pipe = pipe.to(device)
        if pipe.safety_checker is not None:
            pipe.safety_checker = lambda images, **kwargs: (images, False)
    except Exception as e:
        print("Stable Diffusion ���������� �ε� �� ����:", e)
        pipe = None

    total_psnr = 0.0
    total_ssim = 0.0
    total_psnr_sd = 0.0
    total_ssim_sd = 0.0
    total_iou = 0.0
    count = 0

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        gt_orig = None
        gt_mask = None
        if have_orig:
            orig_path = os.path.join(orig_dir, base_name)
            if os.path.isfile(orig_path):
                gt_orig = Image.open(orig_path).convert("RGB")
        if have_mask:
            mask_path = os.path.join(mask_dir, base_name)
            if os.path.isfile(mask_path):
                gt_mask = Image.open(mask_path).convert("L")

        # ���׸����̼� ����
        img_tensor = img_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask_logits = seg_model(img_tensor)
            pred_mask_prob = torch.sigmoid(pred_mask_logits)
        pred_mask = (pred_mask_prob >= 0.5).cpu().numpy().astype(np.uint8)
        pred_mask = pred_mask[0, 0] * 255  # 2D array (H,W)
        mask_pil = Image.fromarray(pred_mask, mode='L')

        # �츮 �� ���� �߷�
        mask_tensor = torch.from_numpy((pred_mask // 255)).unsqueeze(0).unsqueeze(0).float().to(device)
        input_tensor = torch.cat([img_tensor, mask_tensor], dim=1)
        with torch.no_grad():
            output_tensor = rest_model(input_tensor)
        output_np = denormalize(output_tensor[0])
        output_pil = Image.fromarray(output_np)

        # Stable Diffusion ���� �߷�
        stable_pil = None
        if pipe is not None:
            try:
                result = pipe(prompt="a car", image=img, mask_image=mask_pil)
                stable_pil = result.images[0]
            except Exception as e:
                print(f"Stable Diffusion ���� ���� ({base_name}):", e)

        # ��ǥ ���
        if gt_orig is not None:
            psnr_val = peak_signal_noise_ratio(np.array(gt_orig), np.array(output_pil), data_range=255)
            ssim_val = structural_similarity(np.array(gt_orig), np.array(output_pil), multichannel=True)
            total_psnr += psnr_val
            total_ssim += ssim_val
            
            # ���� �ջ�� ��� (���� ����ũ ���� �� ���� 20 �̻��� �ȼ� ����)
            orig_np = np.array(gt_orig.convert("L"))  # �׷��̽�����
            rest_np = np.array(output_pil.convert("L"))
            diff_map = np.abs(orig_np - rest_np)
            mask_bin = pred_mask > 127  # ���� ����ũ ���̳ʸ�

            damaged_pixels = np.logical_and(diff_map > 20, mask_bin).sum()
            total_mask_pixels = mask_bin.sum() or 1
            damage_ratio = (damaged_pixels / total_mask_pixels) * 100.0

            # ���� ��� �Ǵ�
            if psnr_val < 20 or ssim_val < 0.80 or damage_ratio > 30:
                grade = "Irreparable"
            elif psnr_val >= 30 and ssim_val >= 0.90 and damage_ratio <= 10:
                grade = "Excellent"
            else:
                grade = "Repairable"

            print(f"{base_name} ? PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, "
                  f"Damage: {damage_ratio:.1f}%, Restoration Grade: {grade}")
            
            if stable_pil is not None:
                psnr_sd_val = peak_signal_noise_ratio(np.array(gt_orig), np.array(stable_pil), data_range=255)
                ssim_sd_val = structural_similarity(np.array(gt_orig), np.array(stable_pil), multichannel=True)
            else:
                psnr_sd_val = 0.0
                ssim_sd_val = 0.0
            total_psnr_sd += psnr_sd_val
            total_ssim_sd += ssim_sd_val
        if gt_mask is not None:
            gt_mask_arr = np.array(gt_mask)
            gt_mask_bin = gt_mask_arr > 127
            pred_mask_bin = pred_mask > 127
            union = np.logical_or(pred_mask_bin, gt_mask_bin).sum()
            inter = np.logical_and(pred_mask_bin, gt_mask_bin).sum()
            iou_val = 1.0 if union == 0 else inter / union
            total_iou += iou_val
        count += 1

    if count == 0:
        print("�׽�Ʈ�� �̹����� �����ϴ�.")
        return
    if have_mask:
        avg_iou = total_iou / count
        print(f"���׸����̼� IoU: {avg_iou*100:.1f}%")
    else:
        print("���� ����ũ�� ���� ���׸����̼� IoU�� ����� �� �����ϴ�.")
    if have_orig:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_psnr_sd = total_psnr_sd / count if pipe is not None else 0.0
        avg_ssim_sd = total_ssim_sd / count if pipe is not None else 0.0
        print(f"�츮 �� ��� PSNR: {avg_psnr:.2f} dB")
        print(f"�츮 �� ��� SSIM: {avg_ssim:.4f}")
        if pipe is not None:
            print(f"Stable Diffusion ��� PSNR: {avg_psnr_sd:.2f} dB")
            print(f"Stable Diffusion ��� SSIM: {avg_ssim_sd:.4f}")
    else:
        print("���� �̹����� ���� PSNR/SSIM�� ����� �� �����ϴ�.")
