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
from torch import nn
torch.set_num_threads(14)
import statistics
# import requests
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from torchvision.ops import nms

high_threshold = 0.4
# mid_threshold = 0.4
low_threshold = 0.3
sim_threshold = 0.75
topk = 5

DATA_DIR = '/cm/shared/kimth1/Tracking/ByteTrack/datasets/GMOT_40/test'
SAVE_DIR = f'./iGLIP_openVC_{low_threshold}/vis_before'
DROP_SAVE_DIR = f'./iGLIP_openVC_{low_threshold}/vis_after'
BOX_SAVE_DIR = f'./iGLIP_openVC_{low_threshold}/GMOT_40_boxes'
FINAL_VIS_DIR = f'./iGLIP_openVC_{low_threshold}/vis_final'
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

def draw_box(img, bbox):
    # bbox is mode xyxy
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    color = (0, 255, 0)
    thickness = 5
    return cv2.rectangle(img, start_point, end_point, color, thickness)

# CAPTION_MAP for high visual feature, CAPTION_MAP2 for low visual feature
CAPTION_MAP = {
    #'person-2': 'green uniform', # v
    'stock-3': 'big buffalo', # v
}

CAPTION_MAP2 = {
    #'person-2': 'person',
    'stock-3': 'animal',
}

# specific_videos = ['balloon-1', 'bird-3', 'boat-0', 'boat-1', 'car-0', 'fish-2', 'fish-3', 'person-3'] # 'person-1', 'person-2', 'stock-3'
# run_videos = ['balloon-1', 'car-0', 'fish-2', 'fish-3', 'person-3']
# specific_videos = ['balloon-3', 'bird-1', 'bird-2', 'bird-3', 'person-2', 'stock-3'] 
# run_videos = ['balloon-3', 'bird-1', 'bird-2', 'bird-3', 'person-2', 'stock-3']
# specific_videos = ['car-1', 'person-2']

specific_videos = CAPTION_MAP.keys()

def get_object_in_image(glip_demo, caption, threshold, frame_name, img_folder, save_vis_before, save_vis_after, use_topk, category):
    frame_path = osp.join(img_folder, frame_name)
    save_path = osp.join(save_vis_before, frame_name)
    img = load_jpg(frame_path)

    result, pred_boxes, proposal_visual_features, proposal_fused_visual_features = glip_demo.run_on_web_image(img, caption, threshold)
    assert pred_boxes.bbox.shape[0] == proposal_visual_features.shape[0]
    assert proposal_visual_features.shape == proposal_fused_visual_features.shape

    # before post-proc 
    cv2.imwrite(save_path, result)
    
    # after post-proc
    nms_output = nms(boxes=pred_boxes.bbox, scores=pred_boxes.get_field("scores"), iou_threshold=0.5)
    bboxes = pred_boxes.bbox[nms_output]
    scores = pred_boxes.get_field("scores")[nms_output]
    proposal_visual_features = proposal_visual_features[nms_output]
    proposal_fused_visual_features = proposal_fused_visual_features[nms_output]

    img = cv2.imread(frame_path)
    keep = []
    assert bboxes.shape[0] == proposal_visual_features.shape[0]
    for box_id in range(bboxes.shape[0]):
        xo1, yo1, xo2, yo2 = bboxes[box_id] 
        cnt = 0
        for box in bboxes:
            xi1, yi1, xi2, yi2 = box
            if cnt > 1:
                break
            if xi1 >= xo1 and yi1 >= yo1 and xi2 <= xo2 and yi2 <= yo2:
                cnt += 1
        if cnt > 1:
            continue
        
        img = draw_box(img, bboxes[box_id])
        keep.append(box_id)
        # max_score = str(round(float(max_score), 2))
        # cv2.putText(
        # img, max_score, (int(xo1), int(yo1)-4), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0)
        # )        
        
    drop_save_path = osp.join(save_vis_after, frame_name)
    cv2.imwrite(drop_save_path, img)
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

    all_vids = os.listdir(DATA_DIR)
    all_vids.sort()
    for category in all_vids:
        if category != 'stock-3':
            continue
        print(f'>>> Run {category}')
        img_folder = osp.join(DATA_DIR, category, 'img1')
        list_frame_names = os.listdir(img_folder)
        list_frame_names.sort()  
        choice_fanmes = list_frame_names # list_frame_names[0]] #random.choices(list_frame_names, k=10)

        caption = CAPTION_MAP[category] if CAPTION_MAP.get(category) is not None \
                    else category.split('-')[0]
        threshold = high_threshold # if category in specific_videos else mid_threshold
        print('caption, threshold', caption, threshold)
        
        caption2 = CAPTION_MAP2[category] if CAPTION_MAP2.get(category) is not None \
                    else category.split('-')[0]
        threshold2 =  low_threshold # if category in specific_videos else mid_threshold
        print('caption2, threshold2', caption2, threshold2)
        # import pdb;pdb.set_trace()

        save_vis_before = osp.join(SAVE_DIR, category + f'-high')
        save_vis_after = osp.join(DROP_SAVE_DIR, category + f'-high')
        save_vis_before2 = osp.join(SAVE_DIR, category + f'-low')
        save_vis_after2 = osp.join(DROP_SAVE_DIR, category + f'-low')
        save_vis_final = osp.join(FINAL_VIS_DIR, category)
        os.makedirs(save_vis_after, exist_ok=True)
        os.makedirs(save_vis_before, exist_ok=True)
        os.makedirs(save_vis_after2, exist_ok=True)
        os.makedirs(save_vis_before2, exist_ok=True)
        os.makedirs(save_vis_final, exist_ok=True)

        final_bboxes = []
        
        for frame_id, frame_name in enumerate(choice_fanmes):
            # get features
            # if category == 'person-2':
            #     topk = None
            #     threshold = 0.5
            # elif category == 'stock-3':
            #     topk = 5

            low_visual_bboxes, low_visual_features, low_scores, _ = get_object_in_image(glip_demo, caption2, threshold2, frame_name, img_folder, save_vis_before2, save_vis_after2, 0, category)
                        
            if category in specific_videos:
                high_visual_bboxes, high_visual_features, high_scores, _ = get_object_in_image(glip_demo, caption, threshold, frame_name, img_folder, save_vis_before, save_vis_after, 1, category)
            else:
                high_visual_bboxes, high_visual_features, high_scores = low_visual_bboxes, low_visual_features, low_scores
            

            if (topk is not None) and (category in specific_videos):
                try:
                    _, indices = torch.topk(high_scores, topk)
                except:
                    _, indices = torch.topk(high_scores, high_scores.shape[0])
            else:
                _, indices = torch.topk(high_scores, high_scores.shape[0])

            high_visual_bboxes = high_visual_bboxes[indices]
            high_visual_features = high_visual_features[indices]
            
            # calculate similarities between high_visual and low_visual
            high_visual_features_norm = high_visual_features / high_visual_features.norm(dim=1)[:, None]
            low_visual_features_norm = low_visual_features / low_visual_features.norm(dim=1)[:, None]
            cosine_scores = torch.mm(high_visual_features_norm, low_visual_features_norm.transpose(0,1))
            if len(cosine_scores) != 0:
                cosine_scores, _ = torch.max(cosine_scores, dim=0)
            else:
                continue

            threshold3 = sim_threshold
            # if category == 'person-2':
            #     threshold3 = 0.65
            # elif category == 'stock-3':
            #     threshold3 = 0.8
            keep = torch.nonzero(cosine_scores > threshold3).squeeze(1)
            selected_low_visual_bboxes = low_visual_bboxes[keep]
            selected_scores = cosine_scores[keep]

            # visualize
            img = cv2.imread(osp.join(img_folder, frame_name))
            for id, bbox in enumerate(selected_low_visual_bboxes):
                img = draw_box(img, bbox)
                x1, y1, x2, y2 = bbox
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                if category in specific_videos:
                    score = str(round(float(selected_scores[id]), 2))
                    cv2.putText(
                    img, score, (int(x1), int(y1)-4), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                final_bboxes.append([frame_id, -1, x1, y1, w, h, 1, 1, 1])
                
            vis_final_path = osp.join(save_vis_final, frame_name)
            try:
                cv2.imwrite(vis_final_path, img)
            except: 
                print('img is empty or wrong path')

        np.savetxt(osp.join(BOX_SAVE_DIR, f'{category}.txt'), np.array(final_bboxes), fmt='%.6f', delimiter=',')

    print('Done!')
if __name__ == '__main__':
    main()