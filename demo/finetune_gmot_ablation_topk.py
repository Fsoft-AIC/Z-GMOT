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
import statistics

# import requests
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
import sys
sys.path.append('/cm/shared/kimth1/GLIP')
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from torchvision.ops import nms

DATA_DIR = '/cm/shared/kimth1/Tracking/ByteTrack/datasets/GMOT_40/test'
topk = 8
SAVE_DIR = f'./ablation/ablation_top_{topk}/vis_before'
DROP_SAVE_DIR = f'./ablation/ablation_top_{topk}/vis_after'
BOX_SAVE_DIR = f'./ablation/ablation_top_{topk}/GMOT_40_boxes'
FINAL_VIS_DIR = f'./ablation/ablation_top_{topk}/vis_final'

os.makedirs(DROP_SAVE_DIR, exist_ok=True)
os.makedirs(BOX_SAVE_DIR, exist_ok=True)

config_file = "configs/pretrain/glip_Swin_L.yaml" #"configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
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
    color = (255, 0, 0)
    thickness = 2
    return cv2.rectangle(img, start_point, end_point, color, thickness)


FINETUNE_ONLY=True
# CAPTION_MAP for high visual feature, CAPTION_MAP2 for low visual feature

CAPTION_MAP = {
    'ball-0': 'red object',
    'car-1': 'white car',
    'person-2': 'red uniform',
    'person-1': 'black uniform'
}
THRES_MAP = {
    'ball-0': 0.3,
    'car-1': 0.3,
    'person-2': 0.3,
    'person-1': 0.3
}
COND_MAP = {
    'ball-0': False,
    'car-1': False,
    'person-2': False,
    'person-1': False
}

CAPTION_MAP2 = {
    'ball-0': 'ball',
    'car-1': 'car',
    'person-2': 'person',
    'person-1': 'person'
}
THRES_MAP2 = {
    'ball-0': 0.3,
    'car-1': 0.3,
    'person-2': 0.3,
    'person-1': 0.3
}
COND_MAP2 = {
    'ball-0': False, 
    'car-1': False,
    'person-2': False,
    'person-1': False
}

THRES_MAP3 = {
    'ball-0': 0.92,
    'car-1': 0.87,
    'person-2': 0.75,
    'person-1': 0.75
}

TOPK = {
    # 'ball-0': 5,
    # 'car-1': 5,
    # 'person-2': 5,
    # 'person-1': 5
}

result = {}
def get_object_in_image(glip_demo, caption, threshold, condition, frame_name, img_folder, save_vis_before, save_vis_after):
    frame_path = osp.join(img_folder, frame_name)
    save_path = osp.join(save_vis_before, frame_name)
    img = load_jpg(frame_path)

    result, pred_boxes, proposal_visual_features, proposal_fused_visual_features = glip_demo.run_on_web_image(img, caption, threshold)
    assert pred_boxes.bbox.shape[0] == proposal_visual_features.shape[0]
    assert proposal_visual_features.shape == proposal_fused_visual_features.shape

    # before post-proc 
    cv2.imwrite(save_path, result)
    
    # after post-proc
    nms_output = nms(boxes=pred_boxes.bbox, scores=pred_boxes.get_field("scores"), iou_threshold=0.4)
    bboxes = pred_boxes.bbox[nms_output]
    scores = pred_boxes.get_field("scores")[nms_output]
    proposal_visual_features = proposal_visual_features[nms_output]
    proposal_fused_visual_features = proposal_fused_visual_features[nms_output]

    img = cv2.imread(frame_path)

    areas = (bboxes[:,2]-bboxes[:,0]+1)*(bboxes[:,3]-bboxes[:,1]+1)
    if areas.shape[0] != 0:
        median_area = statistics.median(areas)
    else:
        print('No box is found')
        median_area = 100000000
    keep = []
    for box_id in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[box_id] 
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if w * h > img.shape[0] * img.shape[1] / 6: 
            continue
        if condition and w * h > 5 * median_area:
            continue
        img = draw_box(img, bboxes[box_id])
        keep.append(box_id)
    
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

    for category in tqdm(os.listdir(DATA_DIR)):
        print(f'>>> Run {category}')
        img_folder = osp.join(DATA_DIR, category, 'img1')
        list_frame_names = os.listdir(img_folder)
        list_frame_names.sort()  
        choice_fanmes = list_frame_names #[list_frame_names[0]] #random.choices(list_frame_names, k=10)
        if CAPTION_MAP.get(category) is not None:
            caption, threshold, condition = CAPTION_MAP[category], THRES_MAP[category], COND_MAP[category]
        else:
            continue
        print(' caption, threshold, condition:', caption, threshold, condition)
        try:
            caption2, threshold2, condtition2 = CAPTION_MAP2[category], THRES_MAP2[category], COND_MAP2[category]
        except:
            caption2, threshold2, condtition2 = caption, threshold, condition

        save_vis_before = osp.join(SAVE_DIR, category + '-high')
        save_vis_after = osp.join(DROP_SAVE_DIR, category + '-high')
        save_vis_before2 = osp.join(SAVE_DIR, category + '-low')
        save_vis_after2 = osp.join(DROP_SAVE_DIR, category + '-low')
        save_vis_final = osp.join(FINAL_VIS_DIR, category)
        os.makedirs(save_vis_after, exist_ok=True)
        os.makedirs(save_vis_before, exist_ok=True)
        os.makedirs(save_vis_after2, exist_ok=True)
        os.makedirs(save_vis_before2, exist_ok=True)
        os.makedirs(save_vis_final, exist_ok=True)

        final_bboxes = []
        # topk = TOPK.get(category)
        for frame_id, frame_name in enumerate(choice_fanmes):
            # get features
            high_visual_bboxes, _, high_scores, high_visual_features = get_object_in_image(glip_demo, caption, threshold, condition, frame_name, img_folder, save_vis_before, save_vis_after)            
            if caption != caption2:
                low_visual_bboxes, _, low_scores, low_visual_features = get_object_in_image(glip_demo, caption2, threshold2, condtition2, frame_name, img_folder, save_vis_before2, save_vis_after2)  
            else:
                low_visual_bboxes, low_visual_features, low_scores = high_visual_bboxes, high_visual_features, high_scores
            
            if topk is not None:
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
            try:
                cosine_scores, _ = torch.max(cosine_scores, dim=0)
            except:
                continue
                # import pdb; pdb.set_trace()

            if THRES_MAP3.get(category) is not None:
                threshold3 = THRES_MAP3[category]
            else:
                threshold3 = 0.9
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
                score = str(round(float(selected_scores[id]), 2))
                cv2.putText(
                img, score, (int(x1), int(y1)-4), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                final_bboxes.append([frame_id, -1, x1, y1, w, h, 1, 1, 1])
                
            vis_final_path = osp.join(save_vis_final, frame_name)
            cv2.imwrite(vis_final_path, img)
            
            # import pdb; pdb.set_trace()

        np.savetxt(osp.join(BOX_SAVE_DIR, f'{category}.txt'), np.array(final_bboxes), fmt='%.6f', delimiter=',')

if __name__ == '__main__':
    main()
