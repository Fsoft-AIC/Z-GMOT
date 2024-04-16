from torchvision import transforms, ops
import torch.nn.functional as F
import torch

import numpy as np

class FilterTools():
    def __init__(self, short_mems, long_mems):
        self.short_mems = short_mems
        self.long_mems = long_mems

        self.best_conf = None
        self.best_emb = None
        self.target_mem = None

    def feature_sim_from_gdino(self, dets, feature, ref_pos, ref_conf, img_shape):
        rh,  rw = feature.tensors.shape[2] / img_shape[0] , feature.tensors.shape[3] / img_shape[1]
        det_feats = []
        cropped_feats = []
        for i, det in enumerate(dets):
            xc = int((det[0] + det[2])/2 * rw)
            yc = int((det[1] + det[3])/2 * rh)

            try:
                det_feats.append(feature.tensors[:, :, yc, xc])
            except:
                import pdb;pdb.set_trace()

            if i == ref_pos:
                x, y, xx, yy = det
                if xx - x == 0:
                    xx += 1
                if yy - y == 0:
                    yy += 1
    
                roi = feature.tensors[:, :, int(y * rh):int(yy * rh),
                                            int((x + (xx-x)/8) * rw):int(xx * rw)]
                roi = torch.nn.functional.interpolate(roi, size=(1, 1), mode='bilinear', align_corners=True)
                roi.squeeze()
                cropped_feats.append(roi)

                roi = feature.tensors[:, :, int(y * rh):int(yy * rh),
                                            int(x * rw):int((xx - (xx-x)/8) * rw)]
                roi = torch.nn.functional.interpolate(roi, size=(1, 1), mode='bilinear', align_corners=True)
                roi.squeeze()
                cropped_feats.append(roi)

                roi = feature.tensors[:, :, int((y + (yy-y)/8)* rh):int(yy * rh),
                                            int(x * rw):int(xx * rw)]
                roi = torch.nn.functional.interpolate(roi, size=(1, 1), mode='bilinear', align_corners=True)
                roi.squeeze()
                cropped_feats.append(roi)

                roi = feature.tensors[:, :, int(y * rh):int((yy - (yy-y)/8)* rh),
                                            int(x * rw):int(xx * rw)]
                roi = torch.nn.functional.interpolate(roi, size=(1, 1), mode='bilinear', align_corners=True)
                roi.squeeze()
                cropped_feats.append(roi)

            # # RoI Align
            # x, y, xx, yy = det
            # if xx - x == 0:
            #     xx += 1
            # if yy - y == 0:
            #     yy += 1

            # roi = feature.tensors[:, :, int(y * rh):int(yy * rh),
            #                             int(x * rw):int(xx * rw)]
            # roi = torch.nn.functional.interpolate(roi, size=(1, 1), mode='bicubic', align_corners=True)
            # roi.squeeze()
            # det_feats.append(roi)

        cropped_embs = torch.squeeze(torch.stack(cropped_feats, dim=0))
        embs = torch.squeeze(torch.stack(det_feats, dim=0), 1)
        ref_emb = embs[ref_pos].unsqueeze(0)

        if self.target_mem == None:
            self.target_mem = ref_emb

            if self.long_mems > 0:
                self.best_conf = np.array([ref_conf])
                self.best_emb = ref_emb
                self.cropped_embs = cropped_embs
        else:
            if self.long_mems > 0:
                if (self.best_emb.size()[0] < self.long_mems) and (self.best_conf.min() <= ref_conf):
                    np.append(self.best_conf , ref_conf)
                    self.best_emb = torch.cat((self.best_emb, ref_emb), dim=0)
                else:
                    if self.best_conf.min() <= ref_conf:
                        min_idx = self.best_conf.argmin()
                        self.best_conf[min_idx] = ref_conf
                        self.best_emb[min_idx] = ref_emb
                
                if self.best_conf.max() <= ref_conf:
                    self.cropped_embs = cropped_embs

            if self.target_mem.size()[0] < self.short_mems:
                self.target_mem = torch.cat((self.target_mem, ref_emb), dim=0)

            elif self.target_mem.size()[0] == self.short_mems:
                self.target_mem = torch.cat((self.target_mem[1:, :], ref_emb), dim=0)

        t1_norm = F.normalize(embs, dim=1)
        t2_norm = F.normalize(self.target_mem, dim=1)
        result = torch.mean(torch.mm(t1_norm, t2_norm.t()), dim=1)

        if self.long_mems:
            t3_norm = F.normalize(self.best_emb, dim=1)
            best_res = torch.mean(torch.mm(t1_norm, t3_norm.t()), dim=1)

            t4_norm = F.normalize(self.cropped_embs, dim=1)
            cropped_res = torch.mean(torch.mm(t1_norm, t4_norm.t()), dim=1)
        else:
            best_res = None
            cropped_res = None

        return result, best_res, cropped_res, embs

def nms(bounding_boxes, confidence_score, threshold=0.6):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = bounding_boxes

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return np.array(picked_boxes), np.array(picked_score)

def cal_iou(bbox1, bbox2, mode='iou'):
    # Tính toán tọa độ của vùng giao nhau (intersection)
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Tính toán diện tích của vùng giao nhau
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Tính toán diện tích tổng của hai bbox
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    # Tính toán IoU
    if mode == 'iou':
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    else:
        #bbox1 nên là box lớn hơn, nếu ko lớn hơn thì kqua sẽ không đủ cao
        iou = intersection_area / bbox2_area

    return iou

def contains_bbox(bbox1, bbox2):
  """
  Returns True if bbox2 is contained inside bbox1, False otherwise.

  Args:
    bbox1: The larger bounding box.
    bbox2: The smaller bounding box.

  Returns:
    True if bbox2 is contained inside bbox1, False otherwise.
  """

  top_left_1 = (bbox1[0], bbox1[1])
  bottom_right_1 = (bbox1[2], bbox1[3])
  top_left_2 = (bbox2[0], bbox2[1])
  bottom_right_2 = (bbox2[2], bbox2[3])

  return top_left_2[0] >= top_left_1[0] and top_left_2[1] >= top_left_1[1] \
    and bottom_right_2[0] <= bottom_right_1[0] and bottom_right_2[1] <= bottom_right_1[1]
