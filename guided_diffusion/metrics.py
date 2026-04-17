
import torch
import torch.nn.functional as F
import numpy as np



def _require(pkg, install):
    try:
        return __import__(pkg)
    except ImportError:
        raise ImportError(f"Run: pip install {install}")



def _mask_to_unknown(imgs, mask):
    unknown_mask = (1.0 - mask[:, :1])          
    unknown_mask = unknown_mask.expand_as(imgs)  
    return imgs * unknown_mask


def _to_uint8(imgs):
    imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    return imgs.permute(0, 2, 3, 1).cpu().numpy()


def _to_01(imgs):
    return (imgs.clamp(-1, 1) + 1) / 2.0



class LPIPSMetric:

    def __init__(self, device='cpu'):
        lpips = _require('lpips', 'lpips')
        self.device = device
        self.fn = lpips.LPIPS(net='alex').to(device)
        self.fn.eval()
        self.scores = []

    @torch.no_grad()
    def update(self, pred, gt, mask):
        pred = pred.to(self.device)
        gt   = gt.to(self.device)
        mask = mask.to(self.device)

        pred_m = _mask_to_unknown(pred, mask)
        gt_m   = _mask_to_unknown(gt,   mask)

        score = self.fn(pred_m, gt_m)   
        self.scores.extend(score.view(-1).cpu().tolist())

    def compute(self):
        return float(np.mean(self.scores)) if self.scores else float('nan')



class SSIMMetric:

    def __init__(self, device='cpu'):
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        self.device = device
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



class FIDMetric:

    def __init__(self, device='cpu'):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.device = device
        self.fn = FrechetInceptionDistance(normalize=True).to(device)

    @torch.no_grad()
    def update(self, pred, gt, mask):
        pred = pred.to(self.device)
        gt   = gt.to(self.device)
        mask = mask.to(self.device)

        pred_m = _mask_to_unknown(pred, mask)
        gt_m   = _mask_to_unknown(gt,   mask)

        self.fn.update(_to_01(pred_m), real=False)
        self.fn.update(_to_01(gt_m),   real=True)

    def compute(self):
        return self.fn.compute().item()



class InpaintingMetrics:

    def __init__(self, device='cpu'):
        self.lpips = LPIPSMetric(device)
        self.ssim  = SSIMMetric(device)
        self.fid   = FIDMetric(device)

    def update(self, pred, gt, mask):
        self.lpips.update(pred, gt, mask)
        self.ssim.update(pred, gt, mask)
        self.fid.update(pred, gt, mask)

    def compute(self):
        return {
            'LPIPS': round(self.lpips.compute(), 4),   
            'SSIM':  round(self.ssim.compute(),  4),   
            'FID':   round(self.fid.compute(),   2),   
        }



if __name__ == '__main__':
    print("Running sanity check with random tensors...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, C, H, W = 4, 3, 256, 256

    gt   = torch.rand(B, C, H, W) * 2 - 1
    pred = gt + 0.05 * torch.randn_like(gt)    
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, H//4:3*H//4, W//4:3*W//4] = 1  

    evaluator = InpaintingMetrics(device=device)
    evaluator.update(pred.to(device), gt.to(device), mask.to(device))

    results = evaluator.compute()
    print("Metrics (near-perfect pred):", results)
    print("Expected: LPIPS~low, SSIM~high, FID~low")
