import os
import numpy as np
from PIL import Image
import blobfile as bf
from torch.utils.data import Dataset

def list_images(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        if "." in entry and entry.split(".")[-1].lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(list_images(full_path))
    return results

def center_crop(pil_image, size):
    while min(*pil_image.size) >= 2 * size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy, cx = (arr.shape[0] - size) // 2, (arr.shape[1] - size) // 2
    return arr[cy:cy + size, cx:cx + size]

class InpaintDataset(Dataset):
    def __init__(self, gt_path, mask_path, image_size, max_len=None, offset=0, **kw):
        gts = sorted(list_images(os.path.expanduser(gt_path)))[offset:]
        masks = sorted(list_images(os.path.expanduser(mask_path)))[offset:]
        assert len(gts) == len(masks)
        self.gts, self.masks = gts, masks
        self.size = image_size
        self.max_len = max_len

    def __len__(self):
        return self.max_len if self.max_len else len(self.gts)

    def __getitem__(self, idx):
        gt = center_crop(self._load(self.gts[idx]), self.size).astype(np.float32) / 127.5 - 1
        mask = center_crop(self._load(self.masks[idx]), self.size).astype(np.float32) / 255.0
        return {
            'GT': np.transpose(gt, [2, 0, 1]),
            'GT_name': os.path.basename(self.gts[idx]),
            'gt_keep_mask': np.transpose(mask, [2, 0, 1]),
        }

    def _load(self, path):
        with bf.BlobFile(path, "rb") as f:
            img = Image.open(f); img.load()
        return img.convert("RGB")
