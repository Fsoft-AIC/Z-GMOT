# from genericpath import isfile
import os 
import os.path as osp

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cv2 
from tqdm import tqdm
import torch
from torch import nn
torch.set_num_threads(14)
import statistics
import json
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from torchvision.ops import nms

box_thres = 0.35
sim_thres = 0.85
topk = 5

DATA_DIR = '/cm/shared/kimth1/Tracking/G2MOT/frames/GMOT40/test'
BOX_SAVE_DIR = f'./iGLIP_openvoc/GMOT_40_boxes'
FINAL_VIS_DIR = f'./iGLIP_openvoc/visualize'
os.makedirs(BOX_SAVE_DIR, exist_ok=True)

config_file = "configs/pretrain/glip_Swin_L.yaml" 
weight_file = "/home/kimth1/iGLIP/MODEL/glip_large_model.pth" # "MODEL/glip_large_model.pth"

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

def draw_box(img, bbox):
    # bbox is mode xyxy
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    color = (0, 255, 0)
    thickness = 5
    return cv2.rectangle(img, start_point, end_point, color, thickness)

# CAPTION_MAP for specific object
with open('prompt/gmot40/specific.json') as f:
    CAPTION_MAP = json.load(f)

with open('prompt/gmot40/general.json') as f:
    CAPTION_MAP2 = json.load(f)

def get_object_in_image(glip_demo, caption, threshold, frame_name, img_folder):
    frame_path = osp.join(img_folder, frame_name)
    img = load_jpg(frame_path)
    result, pred_boxes, proposal_visual_features, \
        proposal_fused_visual_features = glip_demo.run_on_web_image(img, caption, threshold)

    # after post-proc
    nms_output = nms(boxes=pred_boxes.bbox, scores=pred_boxes.get_field("scores"), iou_threshold=0.5)
    bboxes = pred_boxes.bbox[nms_output]
    scores = pred_boxes.get_field("scores")[nms_output]
    proposal_visual_features = proposal_visual_features[nms_output]
    proposal_fused_visual_features = proposal_fused_visual_features[nms_output]

    keep = []
    for box_id in range(bboxes.shape[0]):
        xo1, yo1, xo2, yo2 = bboxes[box_id] 
        cnt = 0
        for box in bboxes:
            xi1, yi1, xi2, yi2 = box
            if cnt > 3:
                break
            if xi1 >= xo1 and yi1 >= yo1 and xi2 <= xo2 and yi2 <= yo2:
                cnt += 1
        if cnt > 2:
            continue
        keep.append(box_id)        
        
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

    all_vids = os.listdir(DATA_DIR)
    all_vids.sort()
    for category in all_vids:
        print(f'>>> Run {category}')
        img_folder = osp.join(DATA_DIR, category, 'img1')
        list_frame_names = sorted(os.listdir(img_folder)) 

        caption = CAPTION_MAP[category] if CAPTION_MAP.get(category) is not None \
                    else category.split('-')[0]
        print('caption: ', caption)

        caption2 = CAPTION_MAP2[category] if CAPTION_MAP2.get(category) is not None \
                    else category.split('-')[0]
        print('caption2: ', caption2)

        save_vis_final = osp.join(FINAL_VIS_DIR, category)
        os.makedirs(save_vis_final, exist_ok=True)

        final_bboxes = []
        
        for frame_id, frame_name in enumerate(list_frame_names):
            frame_id += 1
            gen_bboxes, gen_features, low_scores, _ = get_object_in_image(glip_demo, caption2, box_thres, frame_name, img_folder)    
            if caption != caption2:
                spe_bboxes, spe_features, high_scores, _ = get_object_in_image(glip_demo, caption, box_thres, frame_name, img_folder)                                                      
                try:
                    _, indices = torch.topk(high_scores, topk)
                except:
                    _, indices = torch.topk(high_scores, high_scores.shape[0])

                spe_bboxes = spe_bboxes[indices]
                spe_features = spe_features[indices]
                
                # calculate similarities between specific objects and general objects
                spe_features_norm = spe_features / spe_features.norm(dim=1)[:, None]
                gen_features_norm = gen_features / gen_features.norm(dim=1)[:, None]
                cosine_scores = torch.mm(spe_features_norm, gen_features_norm.transpose(0,1))
                if len(cosine_scores) != 0:
                    cosine_scores, _ = torch.max(cosine_scores, dim=0)
                else:
                    continue

                keep = torch.nonzero(cosine_scores > sim_thres).squeeze(1)
                selected_gen_bboxes = gen_bboxes[keep]
                selected_scores = cosine_scores[keep]
            else:
                selected_gen_bboxes = gen_bboxes

            # visualize
            img = cv2.imread(osp.join(img_folder, frame_name))
            for id, bbox in enumerate(selected_gen_bboxes):
                img = draw_box(img, bbox)
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                if caption != caption2:
                    score = str(round(float(selected_scores[id]), 2))
                    cv2.putText(
                    img, score, (int(x1), int(y1)-4), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                final_bboxes.append([frame_id, -1, x1, y1, w, h, 1, 1, 1])
                
            vis_final_path = osp.join(save_vis_final, frame_name)
            cv2.imwrite(vis_final_path, img)

        np.savetxt(osp.join(BOX_SAVE_DIR, f'{category}.txt'), \
                   np.array(final_bboxes), fmt='%.6f', delimiter=',')

    print('Done!')
if __name__ == '__main__':
    main()