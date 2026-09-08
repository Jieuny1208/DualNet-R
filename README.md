# DualNet-R: Dual-Network Surface Restoration with Diffusion-Based Pseudo-Ground Truth Generation

Official implementation of **"DualNet-R: Dual-Network Surface Restoration with Diffusion-Based Pseudo-Ground Truth Generation"** (submitted to *Scientific Reports*, under revision).

<!-- Zenodo DOI가 나오면 아래 줄의 XXXXXXX를 교체하세요 -->
[![DOI](https://zenodo.org/badge/1360413277.svg)](https://doi.org/10.5281/zenodo.22660522)

DualNet-R restores surface damage in vehicle images **without any paired ground truth**, by distilling the generative capability of a frozen Stable Diffusion inpainting teacher into a lightweight single-pass student network, coupled with weakly-supervised damage segmentation and an interpretable rule-based irreparability assessment.

## Highlights

- **Weak supervision only** — no paired before/after images are required; supervision comes from diffusion-generated pseudo-ground truth (pseudo-GT).
- **Single-pass inference** — the deployed student (31.4M parameters) removes the diffusion model from the inference path entirely (859.5M → 31.4M), yielding a **3.8× latency advantage** over DiffIR under identical hardware/software settings.
- **Human-preferred restorations** — in a blinded 3-alternative forced-choice study (15 raters, 50 pairs, Fleiss' κ = 0.908), DualNet-R was preferred over DiffIR in **96.5% of decisive judgments**, despite a small deficit on distortion metrics (perception–distortion trade-off).
- **Irreparability assessment** — five interpretable, auditable criteria (damage area, center invasion, patch count, edge contact, dark-gray coverage) decide whether restoration should be attempted at all.

## Pipeline

Four sequential phases (see Figure 1 of the paper):

1. **Weakly-Supervised Segmentation** — attention U-Net (4-stage encoder–decoder, 64–128–256–512 channels, 1,024-channel bottleneck; additive attention gates on all skip connections; 31.4M parameters).
2. **Self-Training Refinement** — sigmoid probability maps binarized at threshold 0.5; one refinement round over all 2,816 training images.
3. **Teacher Pseudo-GT Generation** — Stable Diffusion v2.1 inpainting (`stabilityai/stable-diffusion-2-inpainting`), DDIM sampler, 50 steps, guidance 7.5, η = 0, per-image deterministic seeding (seed + image index), negative prompt `"blurry, distorted, unrealistic"`, mask-based compositing so undamaged regions remain pixel-identical to the input.
4. **Student Training via Knowledge Distillation** — 4-channel input (RGB + binary mask), tanh output head, mask-weighted L1 objective (λ = 1.0, 2× weight inside the mask); Adam, lr 1e-4, batch 8, StepLR (×0.5 every 40 epochs).

All experiments use the official **CarDD-SOD** splits (2,816 / 810 / 374) and a fixed random seed of **42**.

## Installation

```bash
git clone https://github.com/Jieuny1208/DualNet-R.git
cd DualNet-R
pip install -r requirements.txt
```

Main dependencies: PyTorch, diffusers, transformers, opencv-python, scikit-image, numpy.

## Dataset

Download **CarDD** (Car Damage Detection dataset) from the official source and place the SOD subset under `data/` following the official train/val/test splits (2,816 / 810 / 374). We do not redistribute the dataset; please follow the original license.

## Usage

```bash
# Phase 1–2: segmentation + self-training
python <segmentation_training_script>.py

# Phase 3: pseudo-GT generation with the SD teacher
python <pseudo_gt_generation_script>.py

# Phase 4: student training
python <student_training_script>.py

# Inference (segmentation → irreparability check → restoration)
python <inference_script>.py --input path/to/image.jpg
```

*(Replace the script names above with the actual file names in this repository.)*

## Results (test split, n = 374)

| Method | PSNR ↑ | SSIM ↑ | Relative latency |
|---|---|---|---|
| DiffIR | 24.2 | 0.85 | 3.8× |
| **DualNet-R (ours)** | 22.8 | 0.84 | **1×** |

DiffIR retains a small, statistically significant edge on distortion metrics (paired Wilcoxon on per-image metrics; see the paper's Supplementary Table S1), while **human evaluators preferred DualNet-R in 96.5% of decisive blinded judgments** (719/745; 49 of 50 pairs by majority; Fleiss' κ = 0.908) — a dissociation consistent with the perception–distortion trade-off (Blau & Michaeli, CVPR 2018).

## Citation

```bibtex
@article{dualnetr2026,
  title   = {DualNet-R: Dual-Network Surface Restoration with Diffusion-Based Pseudo-Ground Truth Generation},
  author  = {Lee, Jieun and Kim, Doohong and Jeong, Jongpil},
  journal = {Scientific Reports (under revision)},
  year    = {2026}
}
```

## License

Code released for academic research. The CarDD dataset and Stable Diffusion weights are governed by their own licenses.
