import sys
import time
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import display
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TVF
from torchvision.utils import make_grid
from tqdm import tqdm
from torch import Tensor
import cv2
from tap import Tap

from rembg import session_factory, session_simple
import onnxruntime as ort


def get_session(u2net_path):
    session_class = session_simple.SimpleSession
    
    model_name = 'u2net'
    sess_opts = ort.SessionOptions()
    
    return session_class(
        model_name,
        ort.InferenceSession(
            str(u2net_path),
            providers=ort.get_available_providers(),
            sess_options=sess_opts,
        ),
    ) 

# Set autograd and device
torch.set_grad_enabled(False)
device = 'cuda'

def get_depth(midas_repo_path, midas_weights_path, image):
    # image is [h,w,3] in [0, 255]
    assert image.dtype == np.uint8
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"))
    
    
    midas = torch.hub.load(
        repo_or_dir=midas_repo_path,
        model='DPT_BEiT_L_512',
        source='local',
        force_reload=True,
        pretrained=False,
    )
    midas.load_state_dict(torch.load(midas_weights_path))
    midas = midas.to(device).eval()
    
    transforms = torch.hub.load(
        repo_or_dir=midas_repo_path,
        model='transforms',
        source='local',
        force_reload=True,
    )
    transform = transforms.beit512_transform

    
    input_batch = transform(image).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output


class Args(Tap):
    image_path: str
    output_dir: str
    u2net_path: str
    midas_repo_path: str
    midas_weights_path: str
    mask_path: Optional[str] = None
    size: int = 512
    overwrite: bool = False

def center_crop(image):
    h,w = image.size
    s = min(h,w)
    oh = (h - s) // 2
    ow = (w - s) // 2
    return image.crop([oh,ow,oh+s, ow+s])

def main(args: Args):

    # Output paths
    output_dir: Path = Path(args.output_dir)
    image_path: Path = output_dir / 'image.png'
    mask_path: Path = output_dir / 'mask.png'
    rgba_path: Path = output_dir / 'rgba.png'
    idepth_path: Path = output_dir / 'idepth.npy'
    output_dir.mkdir(exist_ok=True, parents=True)
    if not args.overwrite:
        if image_path.is_file() or mask_path.is_file() or rgba_path.is_file():
            print('Will not overwrite. Exiting. Pass in --overwrite if desired.')
            sys.exit()
    
    # Load image and mask
    image = Image.open(args.image_path).convert('RGB')
    image = center_crop(image)
    
    image_orig = image
    if args.mask_path is None:
        rgba = get_rgba(image, u2net_path=args.u2net_path)
        rgba = TVF.to_tensor(rgba).unsqueeze(0)  # (1, 4, H, W)
        image = TVF.to_tensor(image).unsqueeze(0)  # (1, 3, H, W)
        mask = rgba[:, 3:]  # (1, 1, H, W)
    else:
        mask = Image.open(args.mask_path).convert('L')
        image = TVF.to_tensor(image).unsqueeze(0)  # (1, 3, H, W)
        mask = TVF.to_tensor(mask).unsqueeze(0)  # (1, 1, H, W)
        rgba = torch.cat((image, mask), dim=1)  # (1, 4, H, W)

    # Expand
    image, mask = expand_to_size(image, mask, size=args.size)

    # Save
    TVF.to_pil_image(image.squeeze(0)).save(image_path)
    TVF.to_pil_image(mask.squeeze(0)).save(mask_path)
    TVF.to_pil_image(rgba.squeeze(0)).save(rgba_path)
    
    idepth = get_depth(
        args.midas_repo_path, args.midas_weights_path, np.array(image_orig))
    with open(idepth_path, 'wb') as fp:
        np.save(fp, idepth)
    print(f'Saved files to {output_dir}')
    

def expand_to_size(image: Tensor, mask: Tensor, size: int):
    assert image.shape[-2:] == mask.shape[-2:], f'{image.shape=} and {mask.shape=}'
    H, W = image.shape[-2:]
    if H == W == size: 
        return image, mask
    
    # Resize image so the longest size is the given size
    if H > W:
        new_image = torch.zeros(1, 3, H, H)
        new_mask = torch.zeros(1, 1, H, H)
        padding = (H - W) / 2
        new_image[:, :, :, math.floor(padding):H-math.ceil(padding)] = image
        new_mask[:, :, :, math.floor(padding):H-math.ceil(padding)] = mask
    elif W > H:
        new_image = torch.zeros(1, 3, W, W)
        new_mask = torch.zeros(1, 1, W, W)
        padding = (W - H) / 2
        new_image[:, :, math.floor(padding):W-math.ceil(padding)] = image
        new_mask[:, :, math.floor(padding):W-math.ceil(padding)] = mask
    else:
        new_image, new_mask = image, mask
    
    # Resize
    new_image: Tensor = F.interpolate(new_image, size=(size, size), mode='bilinear')
    new_mask: Tensor = F.interpolate(new_mask, size=(size, size), mode='nearest')
    print(f'Resized to size {new_image.shape}')

    return new_image, new_mask


def get_rgba(image, u2net_path):
    try:
        from rembg import remove
    except ImportError:
        print('Please install rembg with "pip install rembg"')
        sys.exit()
    return remove(image, alpha_matting=False, session=get_session(u2net_path=u2net_path))


if __name__ == "__main__":
    args: Args = Args().parse_args()
    main(args)
    
