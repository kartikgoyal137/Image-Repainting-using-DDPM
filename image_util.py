import os
import torch as th
from PIL import Image

def to_u8(sample):
    if sample is None:
        return None
    return ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).cpu().numpy()

def save_images(imgs, names, dir_path, ext='png'):
    os.makedirs(dir_path, exist_ok=True)
    for name, img in zip(names, imgs):
        name = name.rsplit('.', 1)[0] + '.' + ext
        Image.fromarray(img).save(os.path.join(dir_path, name))
