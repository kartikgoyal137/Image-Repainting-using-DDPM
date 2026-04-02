"""RePaint: Inpainting using Denoising Diffusion Probabilistic Models (DDPM)."""

import os, io, argparse, yaml, random
import numpy as np
import torch as th
import blobfile as bf
from PIL import Image
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

from guided_diffusion.unet import UNetModel
from guided_diffusion.diffusion import (
    GaussianDiffusion, SpacedDiffusion, space_timesteps, get_named_beta_schedule,
)

NUM_CLASSES = 1000


# --- Config ---

class Conf(defaultdict):
    def __init__(self):
        super().__init__(lambda: None)

    def __getattr__(self, attr):
        return self.get(attr)

    def pget(self, name, default=None):
        d = self
        for n in (name.split('.') if '.' in name else [name]):
            d = d.get(n, default) if isinstance(d, dict) else default
            if d is None:
                return default
        return d


def load_conf(path):
    with open(os.path.expanduser(path), 'r') as f:
        data = yaml.safe_load(f.read())
    conf = Conf()
    conf.update(data)
    return conf


# --- Dataset ---

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


# --- Model creation ---

def create_model(conf):
    ch_mult = conf.channel_mult
    if ch_mult == "":
        ch_mult = {512: (0.5,1,1,2,2,4,4), 256: (1,1,2,2,4,4),
                    128: (1,1,2,3,4), 64: (1,2,3,4)}[conf.image_size]
    elif not isinstance(ch_mult, tuple):
        ch_mult = tuple(int(x) for x in ch_mult.split(","))

    att_ds = [conf.image_size // int(r) for r in conf.attention_resolutions.split(",")]

    return UNetModel(
        image_size=conf.image_size, in_channels=3, model_channels=conf.num_channels,
        out_channels=6 if conf.learn_sigma else 3,
        num_res_blocks=conf.num_res_blocks, attention_resolutions=tuple(att_ds),
        dropout=conf.dropout or 0, channel_mult=ch_mult,
        num_classes=NUM_CLASSES if conf.class_cond else None,
        use_checkpoint=conf.use_checkpoint or False, use_fp16=conf.use_fp16 or False,
        num_heads=conf.num_heads or 1, num_head_channels=conf.num_head_channels or -1,
        num_heads_upsample=conf.num_heads_upsample or -1,
        use_scale_shift_norm=conf.use_scale_shift_norm or False,
        resblock_updown=conf.resblock_updown or False, conf=conf,
    )


def create_diffusion(conf):
    betas = get_named_beta_schedule(conf.diffusion_steps or 1000)
    ts_respace = conf.timestep_respacing or [conf.diffusion_steps or 1000]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(conf.diffusion_steps or 1000, ts_respace),
        betas=betas, learn_sigma=conf.learn_sigma or False,
        rescale_timesteps=conf.rescale_timesteps or False, conf=conf,
    )


# --- Image saving ---

def to_u8(sample):
    if sample is None:
        return None
    return ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).cpu().numpy()


def save_images(imgs, names, dir_path, ext='png'):
    os.makedirs(dir_path, exist_ok=True)
    for name, img in zip(names, imgs):
        name = name.rsplit('.', 1)[0] + '.' + ext
        Image.fromarray(img).save(os.path.join(dir_path, name))


# --- Main ---

def main(conf):
    print("Start", conf['name'])
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    model = create_model(conf)
    with bf.BlobFile(os.path.expanduser(conf.model_path), "rb") as f:
        model.load_state_dict(th.load(io.BytesIO(f.read()), map_location="cpu"))
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    diffusion = create_diffusion(conf)

    def model_fn(x, t, y=None, gt=None, **kw):
        return model(x, t, y if conf.class_cond else None, gt=gt)

    # Get eval dataset config
    eval_key = list(conf['data']['eval'].keys())[0]
    ds_conf = conf['data']['eval'][eval_key]
    ds = InpaintDataset(**ds_conf)
    dl = DataLoader(ds, batch_size=ds_conf.get('batch_size', 1),
                    shuffle=False, num_workers=1, drop_last=ds_conf.get('drop_last', False))

    print("sampling...")
    for batch in dl:
        for k in batch:
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        bs = batch['GT'].shape[0]
        model_kwargs = {"gt": batch['GT'], "gt_keep_mask": batch.get('gt_keep_mask')}
        model_kwargs["y"] = (
            th.ones(bs, dtype=th.long, device=device) * conf.cond_y
            if conf.cond_y is not None
            else th.randint(0, NUM_CLASSES, (bs,), device=device)
        )

        result = diffusion.p_sample_loop(
            model_fn, (bs, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised, model_kwargs=model_kwargs,
            device=device, progress=conf.show_progress, return_all=True, conf=conf)

        srs = to_u8(result['sample'])
        gts = to_u8(result['gt'])
        mask = model_kwargs['gt_keep_mask']
        lrs = to_u8(result['gt'] * mask + (-1) * th.ones_like(result['gt']) * (1 - mask))
        gt_masks = to_u8(mask * 2 - 1)

        paths = ds_conf.get('paths', {})
        names = batch['GT_name']
        if paths.get('srs'):
            save_images(srs, names, os.path.expanduser(paths['srs']))
        if paths.get('gts'):
            save_images(gts, names, os.path.expanduser(paths['gts']))
        if paths.get('lrs'):
            save_images(lrs, names, os.path.expanduser(paths['lrs']))
        if paths.get('gt_keep_masks'):
            save_images(gt_masks, names, os.path.expanduser(paths['gt_keep_masks']))

    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=True)
    main(load_conf(parser.parse_args().conf_path))
