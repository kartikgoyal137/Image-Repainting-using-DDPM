"""
Inpainting evaluation metrics used in the RePaint paper.

Three metrics:
  1. LPIPS  — Learned Perceptual Image Patch Similarity (lower = better)
  2. SSIM   — Structural Similarity Index (higher = better)
  3. FID    — Fréchet Inception Distance (lower = better)

All three are computed only on the *inpainted (unknown) region* of the image,
consistent with the evaluation protocol in the RePaint paper.

Usage:
    from metrics import InpaintingMetrics

    evaluator = InpaintingMetrics(device='cuda')
    evaluator.update(pred, gt, mask)          # call for each batch
    results = evaluator.compute()             # call once at the end
    print(results)
    # {'LPIPS': 0.123, 'SSIM': 0.876, 'FID': 42.1}

Inputs to update():
    pred  — (B, C, H, W) float tensor in [-1, 1], the inpainted image
    gt    — (B, C, H, W) float tensor in [-1, 1], the ground truth image
    mask  — (B, 1, H, W) or (B, C, H, W) float tensor, 0 = unknown (inpainted) region

Install requirements:
    pip install lpips torchmetrics[image] torch torchvision
"""

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Install check helpers
# ---------------------------------------------------------------------------

def _require(pkg, install):
    try:
        return __import__(pkg)
    except ImportError:
        raise ImportError(f"Run: pip install {install}")


# ---------------------------------------------------------------------------
# Utility: mask a batch to the unknown region
# ---------------------------------------------------------------------------

def _mask_to_unknown(imgs, mask):
    """
    Zero-out the known region so metrics focus on the inpainted area.
    mask: 1 = known, 0 = unknown. We keep only the unknown pixels.
    """
    unknown_mask = (1.0 - mask[:, :1])          # (B, 1, H, W), 1 = unknown
    unknown_mask = unknown_mask.expand_as(imgs)  # broadcast to (B, C, H, W)
    return imgs * unknown_mask


def _to_uint8(imgs):
    """Convert [-1, 1] float tensor to [0, 255] uint8 numpy array (B, H, W, C)."""
    imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    return imgs.permute(0, 2, 3, 1).cpu().numpy()


def _to_01(imgs):
    """Convert [-1, 1] float tensor to [0, 1] float tensor."""
    return (imgs.clamp(-1, 1) + 1) / 2.0


# ---------------------------------------------------------------------------
# 1. LPIPS — Perceptual similarity
# ---------------------------------------------------------------------------
# The RePaint paper uses LPIPS (AlexNet backbone) to measure how perceptually
# similar the inpainted region is to the ground truth. It captures high-level
# feature differences that pixel metrics miss (textures, structures).

class LPIPSMetric:
    """
    Computes mean LPIPS over the unknown (inpainted) region.
    Lower is better. Typical range: 0.0 – 0.7.
    """

    def __init__(self, device='cpu'):
        lpips = _require('lpips', 'lpips')
        self.device = device
        self.fn = lpips.LPIPS(net='alex').to(device)
        self.fn.eval()
        self.scores = []

    @torch.no_grad()
    def update(self, pred, gt, mask):
        """
        Args:
            pred, gt: (B, C, H, W) tensors in [-1, 1]
            mask:     (B, 1, H, W) or (B, C, H, W) tensor, 1=known, 0=unknown
        """
        pred = pred.to(self.device)
        gt   = gt.to(self.device)
        mask = mask.to(self.device)

        # Mask to unknown region (LPIPS expects [-1, 1] inputs)
        pred_m = _mask_to_unknown(pred, mask)
        gt_m   = _mask_to_unknown(gt,   mask)

        score = self.fn(pred_m, gt_m)   # (B, 1, 1, 1)
        self.scores.extend(score.view(-1).cpu().tolist())

    def compute(self):
        return float(np.mean(self.scores)) if self.scores else float('nan')


# ---------------------------------------------------------------------------
# 2. SSIM — Structural Similarity Index
# ---------------------------------------------------------------------------
# SSIM measures luminance, contrast, and structure similarity. The RePaint
# paper uses it on the inpainted region to verify structural coherence.

class SSIMMetric:
    """
    Computes mean SSIM over the unknown (inpainted) region using a sliding
    Gaussian window, consistent with the standard torchmetrics implementation.
    Higher is better. Range: [-1, 1], practically [0, 1].
    """

    def __init__(self, device='cpu'):
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        self.device = device
        # data_range=2.0 because we work in [-1, 1]
        self.fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.scores = []

    @torch.no_grad()
    def update(self, pred, gt, mask):
        pred = pred.to(self.device)
        gt   = gt.to(self.device)
        mask = mask.to(self.device)

        pred_m = _mask_to_unknown(pred, mask)
        gt_m   = _mask_to_unknown(gt,   mask)

        score = self.fn(pred_m, gt_m)
        self.scores.append(score.item())

    def compute(self):
        return float(np.mean(self.scores)) if self.scores else float('nan')


# ---------------------------------------------------------------------------
# 3. FID — Fréchet Inception Distance
# ---------------------------------------------------------------------------
# FID measures the distance between the distribution of inpainted patches
# and real patches in Inception feature space. It captures whether the model
# produces realistic outputs in aggregate across the dataset.
# Lower is better. The RePaint paper uses it to validate perceptual realism.

class FIDMetric:
    """
    Computes FID between the inpainted regions of predictions and ground truths.
    Requires enough samples (~1000+) to be statistically meaningful.
    Lower is better.
    """

    def __init__(self, device='cpu'):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.device = device
        # normalize=True tells FID to expect [0, 1] float inputs
        self.fn = FrechetInceptionDistance(normalize=True).to(device)

    @torch.no_grad()
    def update(self, pred, gt, mask):
        pred = pred.to(self.device)
        gt   = gt.to(self.device)
        mask = mask.to(self.device)

        pred_m = _mask_to_unknown(pred, mask)
        gt_m   = _mask_to_unknown(gt,   mask)

        # FID expects [0, 1] float
        self.fn.update(_to_01(pred_m), real=False)
        self.fn.update(_to_01(gt_m),   real=True)

    def compute(self):
        return self.fn.compute().item()


# ---------------------------------------------------------------------------
# Combined evaluator
# ---------------------------------------------------------------------------

class InpaintingMetrics:
    """
    Unified evaluator. Call update() for each batch, compute() at the end.

    Example:
        evaluator = InpaintingMetrics(device='cuda')

        for pred, gt, mask in results:
            evaluator.update(pred, gt, mask)

        print(evaluator.compute())
        # {'LPIPS': 0.12, 'SSIM': 0.88, 'FID': 38.4}

    To compare dynamic U modes:
        for mode in ['fixed', 'A', 'D', 'AD']:
            evaluator = InpaintingMetrics(device='cuda')
            for pred, gt, mask in run_inference(mode):
                evaluator.update(pred, gt, mask)
            print(mode, evaluator.compute())
    """

    def __init__(self, device='cpu'):
        self.lpips = LPIPSMetric(device)
        self.ssim  = SSIMMetric(device)
        self.fid   = FIDMetric(device)

    def update(self, pred, gt, mask):
        """
        Args:
            pred: (B, C, H, W) tensor in [-1, 1] — model output
            gt:   (B, C, H, W) tensor in [-1, 1] — ground truth
            mask: (B, 1, H, W) or (B, C, H, W)  — 1=known, 0=unknown/inpainted
        """
        self.lpips.update(pred, gt, mask)
        self.ssim.update(pred, gt, mask)
        self.fid.update(pred, gt, mask)

    def compute(self):
        """Returns a dict with final metric values."""
        return {
            'LPIPS': round(self.lpips.compute(), 4),   # lower is better
            'SSIM':  round(self.ssim.compute(),  4),   # higher is better
            'FID':   round(self.fid.compute(),   2),   # lower is better
        }


# ---------------------------------------------------------------------------
# Quick sanity check (runs without a real model)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Running sanity check with random tensors...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, C, H, W = 4, 3, 256, 256

    # Simulate a batch: pred close to gt in the unknown region
    gt   = torch.rand(B, C, H, W) * 2 - 1
    pred = gt + 0.05 * torch.randn_like(gt)    # nearly perfect prediction
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, H//4:3*H//4, W//4:3*W//4] = 1  # center = known; edges = unknown

    evaluator = InpaintingMetrics(device=device)
    evaluator.update(pred.to(device), gt.to(device), mask.to(device))

    results = evaluator.compute()
    print("Metrics (near-perfect pred):", results)
    print("Expected: LPIPS~low, SSIM~high, FID~low")
