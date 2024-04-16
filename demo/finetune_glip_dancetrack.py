# from genericpath import isfile
import os 
import os.path as osp
import sys
sys.path.append('/home/kimth1/GLIP')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cv2 
from tqdm import tqdm
import torch
torch.set_num_threads(14)
from torch import nn
import statistics

# import requests
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from torchvision.ops import nms

split = 'val'
caption = 'dancer . athlete . human . person'
threshold = 0.45
DATA_DIR = f'/cm/shared/kimth1/Tracking/DanceTrack/{split}'
SAVE_DIR = f'./dancetrack_{threshold}/{split}/vis_before'
DROP_SAVE_DIR = f'./dancetrack_{threshold}/{split}/vis_after'
BOX_SAVE_DIR = f'./dancetrack_{threshold}/{split}/boxes'
os.makedirs(DROP_SAVE_DIR, exist_ok=True)
os.makedirs(BOX_SAVE_DIR, exist_ok=True)

config_file = "configs/pretrain/glip_Swin_L.yaml" 
weight_file = "MODEL/glip_large_model.pth"

def load_jpg(frame_path):
    pil_image = Image.open(frame_path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, caption, save_path: str):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig(save_path)

def draw_box(img, bbox, color=(255, 0, 0)):
    # bbox is mode xyxy
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    color = color
    thickness = 2
    return cv2.rectangle(img, start_point, end_point, color, thickness)

def get_object_in_image(glip_demo, caption, threshold, frame_name, img_folder, save_vis_before, save_vis_after):
    frame_path = osp.join(img_folder, frame_name)
    # save_path = osp.join(save_vis_before, frame_name)
    img = load_jpg(frame_path)

    result, pred_boxes, proposal_visual_features, proposal_fused_visual_features = glip_demo.run_on_web_image(img, caption, threshold)
    assert pred_boxes.bbox.shape[0] == proposal_visual_features.shape[0]
    assert proposal_visual_features.shape == proposal_fused_visual_features.shape

    # before post-proc 
    # cv2.imwrite(save_path, result)
    
    # after post-proc
    nms_output = nms(boxes=pred_boxes.bbox, scores=pred_boxes.get_field("scores"), iou_threshold=0.7)
    bboxes = pred_boxes.bbox[nms_output]
    scores = pred_boxes.get_field("scores")[nms_output]
    proposal_visual_features = proposal_visual_features[nms_output]
    proposal_fused_visual_features = proposal_fused_visual_features[nms_output]

    # img = cv2.imread(frame_path)
    keep = []
    for box_id in range(bboxes.shape[0]):
        xo1, yo1, xo2, yo2 = bboxes[box_id] 
        cnt = 0
        for box in bboxes:
            xi1, yi1, xi2, yi2 = box
            if cnt > 1:
                break
            if xi1 >= xo1 and yi1 >= yo1 and xi2 <= xo2 and yi2 <= yo2:
                cnt += 1
        if cnt <= 1:
            # img = draw_box(img, bboxes[box_id])
            keep.append(box_id)
        # else:
            # img = draw_box(img, bboxes[box_id], (0,255,0))
    
    # drop_save_path = osp.join(save_vis_after, frame_name)
    # cv2.imwrite(drop_save_path, img)
    assert bboxes.shape[0] == proposal_visual_features.shape[0] and bboxes.shape[0] == scores.shape[0]
    assert proposal_visual_features.shape == proposal_fused_visual_features.shape
    return bboxes[keep], proposal_visual_features[keep], scores[keep], proposal_fused_visual_features[keep]

def main():
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    
    cfg.MODEL.ROI_HEADS.NMS = 0.7
    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.2,
        show_mask_heatmaps=False
    )
    print('Create model successfully')

    for category in tqdm(os.listdir(DATA_DIR)):
        print(f'>>> Run {category}')
        img_folder = osp.join(DATA_DIR, category, 'img1')
        list_frame_names = os.listdir(img_folder)
        list_frame_names.sort()  
        choice_fanmes = list_frame_names

        print('caption, threshold', caption, threshold)
        
        save_vis_before = osp.join(SAVE_DIR, category)
        save_vis_after = osp.join(DROP_SAVE_DIR, category)
        os.makedirs(save_vis_after, exist_ok=True)
        os.makedirs(save_vis_before, exist_ok=True)

        final_bboxes = []
        for frame_id, frame_name in enumerate(choice_fanmes):
            # get features
            high_visual_bboxes, high_visual_features, high_scores, _ = get_object_in_image(glip_demo, caption, threshold, frame_name, img_folder, save_vis_before, save_vis_after)
            for id, bbox in enumerate(high_visual_bboxes):
                x1, y1, x2, y2 = bbox
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                final_bboxes.append([frame_id, -1, x1, y1, w, h, 1, 1, 1])

        np.savetxt(osp.join(BOX_SAVE_DIR, f'{category}.txt'), np.array(final_bboxes), fmt='%.6f', delimiter=',')

if __name__ == '__main__':
    main()
