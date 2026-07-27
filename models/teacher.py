# models/teacher.py
"""Stable Diffusion inpainting teacher (pseudo-GT 생성기).

논문 스펙 (3단계):
  - 체크포인트: config 로 선택 (sd15 = runwayml/stable-diffusion-inpainting,
    sd21 = stabilityai/stable-diffusion-2-inpainting)
  - prompt "a car", negative prompt "blurry, distorted, unrealistic"
  - DDIMScheduler 명시 지정, 50 steps, classifier-free guidance 7.5, eta=0
  - 입력 512x512 리사이즈, 마스크 이진화 + dilation
  - 시드 고정된 generator 를 파이프라인에 전달
  - 생성 영역만 마스크로 원본에 합성 (비손상 영역은 픽셀 단위로 원본 유지)
"""

import os

import torch
from PIL import Image, ImageFilter

from utils.helpers import ensure_dir
from utils.seed import make_generator


def resolve_checkpoint(cfg):
    """config 의 teacher.name 을 실제 HuggingFace 체크포인트 ID 로 변환."""
    name = cfg["teacher"]["name"]
    aliases = cfg["teacher"].get("checkpoints", {}) or {}
    checkpoint = aliases.get(name, name)
    if "/" not in checkpoint:
        raise ValueError(
            f"알 수 없는 teacher 입니다: {name!r}. "
            f"사용 가능한 별칭: {sorted(aliases)} 또는 전체 체크포인트 ID를 지정하세요."
        )
    return checkpoint


class DiffusionTeacher:
    """Stable Diffusion inpainting 파이프라인 래퍼."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.tcfg = cfg["teacher"]
        self.device = torch.device(device)
        self.checkpoint = resolve_checkpoint(cfg)
        self.resolution = int(self.tcfg["resolution"])
        self.seed = int(cfg["seed"])
        self.pipe = None

    def load(self):
        # diffusers 는 pseudo-GT 생성 단계에서만 필요하므로 지연 임포트한다.
        from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline

        use_fp16 = bool(self.tcfg.get("fp16", True)) and self.device.type == "cuda"
        dtype = torch.float16 if use_fp16 else torch.float32
        print(f"teacher 체크포인트 로드: {self.checkpoint} (dtype={dtype})")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(self.checkpoint, torch_dtype=dtype)

        scheduler_name = str(self.tcfg.get("scheduler", "ddim")).lower()
        if scheduler_name == "ddim":
            # 논문 스펙: DDIM 샘플러 명시 지정
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        else:
            print(f"[경고] scheduler={scheduler_name} → 체크포인트 기본 스케줄러를 사용합니다.")

        pipe = pipe.to(self.device)
        if self.tcfg.get("disable_safety_checker", True) and getattr(pipe, "safety_checker", None):
            pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        return self

    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_mask(self, mask_img):
        """마스크 이진화 + dilation (손상 경계까지 확실히 덮도록)."""
        mask = mask_img.convert("L").point(lambda v: 255 if v > 127 else 0)
        dilation = int(self.tcfg.get("mask_dilation", 0) or 0)
        if dilation > 0:
            # MaxFilter 커널 크기는 홀수여야 한다 (반경 d → 2d+1)
            mask = mask.filter(ImageFilter.MaxFilter(2 * dilation + 1))
        return mask

    @torch.no_grad()
    def restore(self, image, mask_img, index=0):
        """단일 이미지 인페인팅. index 는 이미지별 결정론적 시드 오프셋."""
        if self.pipe is None:
            raise RuntimeError("teacher 파이프라인이 로드되지 않았습니다. load() 를 먼저 호출하세요.")

        orig_size = image.size  # (W, H)
        res = self.resolution
        img_in = image.convert("RGB").resize((res, res), resample=Image.BILINEAR)
        mask_in = self._prepare_mask(mask_img).resize((res, res), resample=Image.NEAREST)

        # 이미지마다 다른 노이즈를 쓰되 전체적으로는 재현 가능하도록 seed + index 사용
        generator = make_generator(self.seed + index, device=self.device)

        result = self.pipe(
            prompt=self.tcfg["prompt"],
            negative_prompt=self.tcfg.get("negative_prompt") or None,
            image=img_in,
            mask_image=mask_in,
            num_inference_steps=int(self.tcfg["num_inference_steps"]),
            guidance_scale=float(self.tcfg["guidance_scale"]),
            eta=float(self.tcfg.get("eta", 0.0)),
            generator=generator,
        )
        out = result.images[0].resize(orig_size, resample=Image.BILINEAR)

        if self.tcfg.get("composite", True):
            # 마스크 외부(비손상 영역)는 원본 픽셀을 그대로 유지
            mask_full = self._prepare_mask(mask_img).resize(orig_size, resample=Image.NEAREST)
            out = Image.composite(out, image.convert("RGB"), mask_full)
        return out


def generate_pseudo_gt(cfg, device, image_paths, mask_index, output_dir, skip_existing=True):
    """훈련 이미지 전체에 대해 pseudo-GT 를 생성·저장하고 {stem: 경로} 를 반환.

    mask_index: {stem: 마스크 경로} (SelfTraining.generate_masks 의 반환값)
    """
    ensure_dir(output_dir)
    teacher = DiffusionTeacher(cfg, device).load()
    saved = {}
    try:
        total = len(image_paths)
        for i, img_path in enumerate(image_paths):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(output_dir, f"{stem}.png")
            saved[stem] = out_path
            if skip_existing and os.path.isfile(out_path):
                continue
            mask_path = mask_index.get(stem)
            if mask_path is None:
                print(f"[경고] 마스크가 없어 건너뜁니다: {stem}")
                saved.pop(stem, None)
                continue
            image = Image.open(img_path).convert("RGB")
            mask_img = Image.open(mask_path)
            restored = teacher.restore(image, mask_img, index=i)
            restored.save(out_path)
            if (i + 1) % 25 == 0 or (i + 1) == total:
                print(f"  pseudo-GT 생성 {i + 1}/{total}")
    finally:
        teacher.unload()
    return saved
