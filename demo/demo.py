import os 
import os.path as osp
import sys
sys.path.append('/home/phatnt21/language-driven/GLIP')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cv2 
from tqdm import tqdm

import requests
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo



DATA_DIR = '/cm/shared/kimth1/Tracking/GMOT40/GenericMOT_JPEG_Sequence'
SAVE_DIR = '/home/kimth1/'

config_file = "configs/pretrain/glip_Swin_L.yaml"
weight_file = "MODEL/glip_large_model.pth"

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def load_jpg(fpath):
    pil_image = Image.open(fpath).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def imshow(img, caption, save_path: str):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig(save_path)



def main():
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    import torch 
    print(f'n devices: {torch.cuda.device_count()}')

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )
    print('Create model successfully')

    caption = 'black '
    for category in os.listdir(DATA_DIR):
        print(f'Run {category}')
        save_cat = osp.join(SAVE_DIR, category)
        os.makedirs(save_cat, exist_ok=True)
        img_folder = osp.join(DATA_DIR, category, 'img1')

        for fname in tqdm(os.listdir(img_folder)):
            fpath = osp.join(img_folder, fname)
            save_path = osp.join(save_cat, fname)
            caption = category

            img = load_jpg(fpath)
            result, _ = glip_demo.run_on_web_image(img, caption.split('-')[0], 0.5)
            cv2.imwrite(save_path, result[:, :, [2, 1, 0]])
            
            # import pdb; pdb.set_trace()

    pass

if __name__ == '__main__':
    main()
