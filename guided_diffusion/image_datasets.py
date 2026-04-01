"""Image dataset for inpainting."""

import random
import os
from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data_yield(loader):
    while True:
        yield from loader


def load_data_inpa(*, gt_path=None, mask_path=None, batch_size, image_size,
                   deterministic=False, random_crop=False, random_flip=True,
                   return_dataloader=False, return_dict=False, max_len=None,
                   drop_last=True, offset=0, **kwargs):
    gt_dir = os.path.expanduser(gt_path)
    mask_dir = os.path.expanduser(mask_path)

    gt_paths = _list_image_files_recursively(gt_dir)
    mask_paths = _list_image_files_recursively(mask_dir)
    assert len(gt_paths) == len(mask_paths)

    dataset = ImageDatasetInpa(
        image_size, gt_paths=gt_paths, mask_paths=mask_paths,
        random_crop=random_crop, random_flip=random_flip,
        return_dict=return_dict, max_len=max_len, offset=offset
    )

    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=not deterministic, num_workers=1, drop_last=drop_last
    )

    if return_dataloader:
        return loader
    return load_data_yield(loader)


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDatasetInpa(Dataset):
    def __init__(self, resolution, gt_paths, mask_paths,
                 random_crop=False, random_flip=True, return_dict=False,
                 max_len=None, offset=0):
        super().__init__()
        self.resolution = resolution
        gt_paths = sorted(gt_paths)[offset:]
        mask_paths = sorted(mask_paths)[offset:]
        self.local_gts = gt_paths
        self.local_masks = mask_paths
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.return_dict = return_dict
        self.max_len = max_len

    def __len__(self):
        if self.max_len is not None:
            return self.max_len
        return len(self.local_gts)

    def __getitem__(self, idx):
        gt_path = self.local_gts[idx]
        mask_path = self.local_masks[idx]

        pil_gt = self.imread(gt_path)
        pil_mask = self.imread(mask_path)

        arr_gt = center_crop_arr(pil_gt, self.resolution)
        arr_mask = center_crop_arr(pil_mask, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr_gt = arr_gt[:, ::-1]
            arr_mask = arr_mask[:, ::-1]

        arr_gt = arr_gt.astype(np.float32) / 127.5 - 1
        arr_mask = arr_mask.astype(np.float32) / 255.0

        name = os.path.basename(gt_path)
        return {
            'GT': np.transpose(arr_gt, [2, 0, 1]),
            'GT_name': name,
            'gt_keep_mask': np.transpose(arr_mask, [2, 0, 1]),
        }

    def imread(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        return pil_image.convert("RGB")


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
