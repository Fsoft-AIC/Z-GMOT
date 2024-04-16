from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from numpy import random
from pathlib import Path

import os
import cv2

from reid_model import Extractor

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory

import torch
from boxmot.tracker_zoo import create_tracker

model_path = '/cm/shared/kimth1/Tracking/DanceTrack/oracle_analysis/ckpt.t7'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3.5, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3.5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def tlwh_to_xyxy(bbox_tlwh, img_shape):
    """
    TODO:
        Convert bbox from xtl_ytl_w_h to xc_yc_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    x, y, w, h = bbox_tlwh
    x1 = max(int(x), 0)
    x2 = min(int(x+w), img_shape[1] - 1)
    y1 = max(int(y), 0)
    y2 = min(int(y+h), img_shape[0] - 1)
    return x1, y1, x2, y2 

def tlwh_to_xyxy_array(bbox_tlwh, img_shape):
    if isinstance(bbox_tlwh, np.ndarray):
        bbox_xyxy = bbox_tlwh.copy()
    elif isinstance(bbox_tlwh, torch.Tensor):
        bbox_xyxy = bbox_tlwh.clone()
    # import pdb;pdb.set_trace()
    bbox_xyxy[:, 0] = np.maximum(bbox_tlwh[:, 0], 0)
    bbox_xyxy[:, 2] = np.minimum(bbox_tlwh[:, 0] + bbox_tlwh[:, 2], img_shape[1] - 1)
    bbox_xyxy[:, 1] = np.maximum(bbox_tlwh[:, 1], 0)
    bbox_xyxy[:, 3] = np.minimum(bbox_tlwh[:, 1] + bbox_tlwh[:, 3], img_shape[0] - 1)
    return bbox_xyxy

class ReID(object):
    def __init__(self, model_path, use_cuda=True):
        self.extractor = Extractor(model_path, use_cuda=use_cuda)

    def get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = tlwh_to_xyxy(box, ori_img.shape[:2])
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

class Detections(object):
    def __init__(self):
        self.xyxy = None
        self.confidence = None

@torch.no_grad()
def run(args):
    start_frame = 1
    data_dir = '/cm/shared/kimth1/Tracking/G2MOT/frames/GMOT40/test'
    detection_dir = '/cm/shared/kimth1/Tracking/Z-GMOT[official]/iGLIP_openvoc/GMOT_40_boxes'
    save_dir = 'tracking_result'
    vid_list = os.listdir(data_dir)
    vid_list.sort()

    reid = ReID(model_path=model_path)
    for folder_name in vid_list:
        print(f'start tracking {folder_name}')
        
        src_folder = os.path.join(data_dir, folder_name, 'img1')
        dest_folder = os.path.join(save_dir, 'img', folder_name) # visualize
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # Create text path and clear content in text file
        parent_txt_path = os.path.join(save_dir, 'text', args.feature_mode) # write box
        if not os.path.exists(parent_txt_path):
            os.makedirs(parent_txt_path)

        txt_path = os.path.join(parent_txt_path, folder_name + '.txt')

        # Init Tracker
        tracking_config = \
            ROOT /\
            'boxmot' /\
            'deepocsort' /\
            'configs' /\
            ('deepocsort' + '.yaml')

        # import pdb;pdb.set_trace()
        tracker = create_tracker(
            args.tracking_method,
            tracking_config,
            device,
          )

        detections = None
        image = None
        frame_idx = start_frame
  
        det_path = os.path.join(detection_dir, folder_name + '.txt')
        detections_vid = np.loadtxt(det_path, dtype=np.float64, delimiter=',')
        for img in sorted(os.listdir(src_folder)):
        
            #Option for tracking on a snippet
            if frame_idx < args.start_frame:
                frame_idx += 1
                continue

            if (frame_idx > args.end_frame) and (args.end_frame != 0):
                break

            # load image
            img_dir = os.path.join(src_folder, img)
            image = cv2.imread(img_dir)

            # detect objects
            detections = detections_vid[detections_vid[:, 0] == frame_idx]
            bboxes_tlwh = detections[:, 2:6]

            if len(detections) == 0:
                frame_idx += 1
                continue
            embs = reid.get_features(bboxes_tlwh, image)

            detections = Detections()
            detections.xyxy = tlwh_to_xyxy_array(bboxes_tlwh, image.shape[:2])
            detections.confidence = np.ones((len(bboxes_tlwh),))

            mean_vector = torch.mean(embs, dim=0)
            measures = []
            for idx, emb in enumerate(embs):
                sim = torch.nn.functional.cosine_similarity(mean_vector, emb, dim=-1).cpu()
                measures.append(sim)
            outputs = tracker.update(detections, measures, embs.cpu(), image)

            if len(outputs) > 0:
                for j, output in enumerate(outputs):
                
                    bboxes = output[0:4]
                    id = output[4]
                    conf = output[5]
                    cls = output[6]
                    sim = output[7]

                    if args.save_txt:
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        with open(txt_path, 'a') as f:
                            f.write(('%g,' * 9 + '%g' + '\n') % (frame_idx, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, conf, -1, -1, -1))

                    # c = int(cls)  # integer class
                    # id = int(id)  # integer id
                    # label = (f'{id} {conf:.2f} {sim:.2f}')
                    # color = [204, 0, 102]
                    # plot_one_box(bboxes, image, label=label, color=color, line_thickness=2)

            # img_path = os.path.join(dest_folder, img)
            # cv2.imwrite(img_path, image)
            frame_idx += 1

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking-method', type=str, default='masort', help='masort, deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--save-txt', action='store_false', help='save tracking results in a txt file')
    parser.add_argument('--save-dir', type=str, default='/content/drive/MyDrive/FPT-AI/GDinoBase') # save vis and text
    parser.add_argument('--feature-mode', type=str, default='gdino')
    parser.add_argument('--short-mems', type=int, default=0, help='re-filter with best embedding')
    parser.add_argument('--long-mems', type=int, default=0, help='re-filter with best embedding')
    parser.add_argument('--cropped-mems', action='store_true', help='re-filter for occluded objects')
    parser.add_argument('--start-frame', type=int, default=0) # 
    parser.add_argument('--end-frame', type=int, default=0) # 
    opt = parser.parse_args()
    opt = parser.parse_args()
    # print_args(vars(opt))
    return opt

def main(opt):
    run(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)