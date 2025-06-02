# train.py
import os
import torch
import numpy as np
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline
from models.unet import UNet
from models.loss import LossFunctions
from models.self_training import SelfTraining
from utils.helpers import get_device, ensure_dir

# ���׸����̼ǿ� �����ͼ� Ŭ���� ���� (������ �н���)
class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, bbox_file=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # �̹��� ���� ��� ��������
        valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                             if os.path.splitext(fname)[1].lower() in valid_exts]
        self.image_paths.sort()
        # ����ũ ��� �Ǵ� �ٿ���ڽ� ���� ����
        self.mask_dir = mask_dir if mask_dir and os.path.isdir(mask_dir) else None
        self.mask_paths = {}
        if self.mask_dir:
            for fname in os.listdir(self.mask_dir):
                if os.path.splitext(fname)[1].lower() in valid_exts:
                    self.mask_paths[fname] = os.path.join(self.mask_dir, fname)
        # �ٿ���ڽ� CSV ���� �ε�
        self.bbox_info = {}
        if not self.mask_paths and bbox_file and os.path.isfile(bbox_file):
            try:
                import pandas as pd
                df = pd.read_csv(bbox_file)
                for _, row in df.iterrows():
                    fname = str(row.get("filename") or row.get("image") or "")
                    if fname:
                        # x,y,w,h �Ǵ� x1,y1,x2,y2 ���� ����
                        x = int(row.get("x") or row.get("x1", 0))
                        y = int(row.get("y") or row.get("y1", 0))
                        if "w" in row and "h" in row:
                            w = int(row["w"]); h = int(row["h"])
                        else:
                            x2 = int(row.get("x2", 0)); y2 = int(row.get("y2", 0))
                            w = x2 - x; h = y2 - y
                        self.bbox_info[fname] = (x, y, w, h)
            except Exception as e:
                print("�ٿ�� �ڽ� CSV ���� �ε� ����:", e)
        # ���̺� ���� �̹����� ���� (����ũ/�ڽ� ���� ���� ���)
        if self.mask_paths or self.bbox_info:
            self.image_paths = [p for p in self.image_paths 
                                 if os.path.basename(p) in self.mask_paths or os.path.basename(p) in self.bbox_info]
        # (���̺� ������ ���� ������ ��� �̹��� ���)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)
        fname = os.path.basename(img_path)
        # ����ũ ����
        mask_tensor = None
        if fname in self.mask_paths:
            mask_img = Image.open(self.mask_paths[fname]).convert("L")
            mask_tensor = transforms.ToTensor()(mask_img)
            mask_tensor = (mask_tensor > 0.5).float()  # 0 �Ǵ� 1�� ����ȭ
        elif fname in self.bbox_info:
            # �ٿ���ڽ� ������ �̿��� �簢�� ����ũ ����
            x, y, w, h = self.bbox_info[fname]
            mask_img = Image.new("L", img.size, 0)
            # �簢 ������ ���(255)���� ä��
            for yy in range(y, min(y+h, mask_img.height)):
                for xx in range(x, min(x+w, mask_img.width)):
                    mask_img.putpixel((xx, yy), 255)
            mask_tensor = transforms.ToTensor()(mask_img)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            # ���̺� ������ �� ����ũ (0)
            mask_tensor = torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float)
        return img_tensor, mask_tensor

# ����(��������)�� �����ͼ� Ŭ���� ����
class RestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_dir = target_dir
        self.transform = transform
        # �ջ� �̹��� ���
        valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                             if os.path.splitext(fname)[1].lower() in valid_exts]
        self.image_paths.sort()
        # �����Ǵ� ����ũ �� Ÿ�� ��� �غ�
        self.mask_paths = {}
        self.target_paths = {}
        if os.path.isdir(mask_dir):
            for fname in os.listdir(mask_dir):
                if os.path.splitext(fname)[1].lower() in valid_exts:
                    self.mask_paths[fname] = os.path.join(mask_dir, fname)
        if os.path.isdir(target_dir):
            for fname in os.listdir(target_dir):
                if os.path.splitext(fname)[1].lower() in valid_exts:
                    self.target_paths[fname] = os.path.join(target_dir, fname)
        # ����ũ�� Ÿ���� ��� �ִ� �̹����� ���
        self.image_paths = [p for p in self.image_paths 
                             if os.path.basename(p) in self.mask_paths and os.path.basename(p) in self.target_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base = os.path.basename(img_path)
        mask_path = self.mask_paths[base]
        target_path = self.target_paths[base]
        # �̹���, ����ũ, Ÿ�� �ε�
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        target = Image.open(target_path).convert("RGB")
        if self.transform:
            img_tensor = self.transform(img)
            target_tensor = self.transform(target)
        else:
            img_tensor = transforms.ToTensor()(img)
            target_tensor = transforms.ToTensor()(target)
            img_tensor = transforms.Normalize((0.5,)*3, (0.5,)*3)(img_tensor)
            target_tensor = transforms.Normalize((0.5,)*3, (0.5,)*3)(target_tensor)
        # ����ũ �ټ� (0 �Ǵ� 1)
        mask_tensor = transforms.ToTensor()(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        # �̹����� ����ũ ä�� ����
        input_tensor = torch.cat([img_tensor, mask_tensor], dim=0)
        return input_tensor, target_tensor

def train(args):
    device = get_device()
    print(f"��� ��ġ: {device}")

    dataset_dir = args.dataset_dir
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"�����ͼ� ���丮 {dataset_dir} �� �������� �ʽ��ϴ�.")
    train_img_dir = os.path.join(dataset_dir, "CarDD-TR-Image")
    train_mask_dir = os.path.join(dataset_dir, "CarDD-TR-Mask")
    train_edge_dir = os.path.join(dataset_dir, "CarDD-TR-Edge")


    if not os.path.isdir(train_mask_dir):
        train_mask_dir = None
    bbox_file = os.path.join(dataset_dir, "bboxes.csv")
    if not os.path.isfile(bbox_file):
        bbox_file = None

    # ��� ���丮 ����
    ensure_dir("checkpoints")
    pred_mask_dir_stage1 = os.path.join(dataset_dir, "pred_masks_stage1")
    pred_mask_dir_final = os.path.join(dataset_dir, "pred_masks")
    pseudo_gt_dir = os.path.join(dataset_dir, "pseudo_gt")
    ensure_dir(pred_mask_dir_stage1)
    ensure_dir(pred_mask_dir_final)
    ensure_dir(pseudo_gt_dir)

    # ��ȯ ����
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # �����ͼ� �� �����ͷδ� �غ�
    seg_dataset = SegDataset(train_img_dir, mask_dir=train_mask_dir, bbox_file=bbox_file, transform=img_transform)
    seg_loader = DataLoader(seg_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    # �� �ʱ�ȭ
    model_seg = UNet(in_channels=3, out_channels=1).to(device)
    loss_funcs = LossFunctions()
    seg_optimizer = optim.Adam(model_seg.parameters(), lr=args.lr)
    epochs_seg1 = args.epochs_seg
    epochs_seg2 = args.epochs_seg2

    # 1) ���׸����̼� �� 1�� �н�
    print("���׸����̼� ��Ʈ��ũ 1�� �н��� �����մϴ�...")
    model_seg.train()
    for epoch in range(epochs_seg1):
        total_loss = 0.0
        for imgs, masks in seg_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            seg_optimizer.zero_grad()
            preds = model_seg(imgs)
            loss = loss_funcs.segmentation_loss(preds, masks)
            loss.backward()
            seg_optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(seg_loader.dataset)
        print(f"[Epoch {epoch+1}/{epochs_seg1}] ���׸����̼� �ս�: {avg_loss:.4f}")
    print("���׸����̼� 1�� �н� �Ϸ�!")
    # ���׸����̼� �� ����ġ ����
    torch.save(model_seg.state_dict(), os.path.join("checkpoints", "segmentation.pth"))

    # 2) Self-Training: ���� ����ũ ���� �� ���׸����̼� 2�� �н�
    if epochs_seg2 > 0:
        print("���� ����ũ�� �����Ͽ� ���׸����̼� Self-Training(2�� �н�)�� �����մϴ�...")
        st = SelfTraining(model_seg, device, threshold=0.5)
        st.generate_masks(seg_dataset.image_paths, pred_mask_dir_stage1)
        seg_dataset_stage2 = SegDataset(train_img_dir, mask_dir=pred_mask_dir_stage1, transform=img_transform)
        seg_loader_stage2 = DataLoader(seg_dataset_stage2, batch_size=args.batch_size, shuffle=True, drop_last=False)
        model_seg.train()
        for epoch in range(epochs_seg2):
            total_loss = 0.0
            for imgs, masks in seg_loader_stage2:
                imgs = imgs.to(device)
                masks = masks.to(device)
                seg_optimizer.zero_grad()
                preds = model_seg(imgs)
                loss = loss_funcs.segmentation_loss(preds, masks)
                loss.backward()
                seg_optimizer.step()
                total_loss += loss.item() * imgs.size(0)
            avg_loss = total_loss / len(seg_loader_stage2.dataset)
            print(f"[Epoch {epoch+1}/{epochs_seg2}] ���׸����̼� Self-Training �ս�: {avg_loss:.4f}")
        print("���׸����̼� Self-Training(2��) �Ϸ�!")
        torch.save(model_seg.state_dict(), os.path.join("checkpoints", "segmentation_ft.pth"))

    # 3) ���� ���׸����̼� �𵨷� ��� �Ʒ� �̹����� ���� �ջ� ���� ����ũ ����
    print("���� ���׸����̼� �𵨷� ����ũ�� �����մϴ�...")
    st_final = SelfTraining(model_seg, device, threshold=0.5)
    st_final.generate_masks(seg_dataset.image_paths, pred_mask_dir_final)
    print("����ũ ���� �Ϸ�.")
    # ���׸����̼� �� GPU �޸� ����
    model_seg.to("cpu")
    del model_seg
    torch.cuda.empty_cache()

    # 4) Stable Diffusion ������������ ���� �̹��� ����
    print("Stable Diffusion ������������ ���� �̹��� ���� ��...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    pipe = pipe.to(device)
    if pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, False)
    for img_path in seg_dataset.image_paths:
        img = Image.open(img_path).convert("RGB")
        base_name = os.path.basename(img_path)
        mask_path = os.path.join(pred_mask_dir_final, base_name)
        mask_img = Image.open(mask_path).convert("L")
        result = pipe(prompt="a car", image=img, mask_image=mask_img)
        inpainted_image = result.images[0]
        inpainted_image.save(os.path.join(pseudo_gt_dir, base_name))
    del pipe
    torch.cuda.empty_cache()
    print("���� �̹��� ���� �Ϸ�.")

    # 5) ���� ��Ʈ��ũ �н�
    print("���� ��Ʈ��ũ �н��� �����մϴ�...")
    rest_dataset = RestDataset(train_img_dir, mask_dir=pred_mask_dir_final, target_dir=pseudo_gt_dir, transform=img_transform)
    rest_loader = DataLoader(rest_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model_rest = UNet(in_channels=4, out_channels=3).to(device)
    rest_optimizer = optim.Adam(model_rest.parameters(), lr=args.lr)
    for epoch in range(args.epochs_rest):
        total_loss = 0.0
        model_rest.train()
        for inputs, targets in rest_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            rest_optimizer.zero_grad()
            outputs = model_rest(inputs)
            loss = loss_funcs.restoration_loss(outputs, targets)
            loss.backward()
            rest_optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(rest_loader.dataset)
        print(f"[Epoch {epoch+1}/{args.epochs_rest}] ���� ��Ʈ��ũ �ս�: {avg_loss:.4f}")
    print("���� ��Ʈ��ũ �н� �Ϸ�!")
    torch.save(model_rest.state_dict(), os.path.join("checkpoints", "restoration.pth"))
    print("�� ����ġ�� 'checkpoints' ���丮�� ����Ǿ����ϴ�.")
