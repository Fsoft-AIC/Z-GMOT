import cv2
import numpy as np
import os
import os.path as osp

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_tracking(image, tlwhs, obj_ids, frame_id=0, ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = 2
    text_thickness = 2
    line_thickness = 6

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))

        obj_id = obj_ids[i]
        id_text = f'{int(obj_id)}'
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                    thickness=text_thickness)
    return im

split = '/cm/shared/kimth1/Tracking/G2MOT/frames/GMOT40/test'
detection_dir = '/cm/shared/kimth1/Tracking/Z-GMOT[official]/AC-Sort/tracking_result/text/gdino'

for video_name in os.listdir(split):
    if video_name != 'car-1':
        continue
    resutl_path = f'{detection_dir}/{video_name}.txt'
    original_folder = f'{split}/{video_name}/img1'

    vis_folder = f'/cm/shared/kimth1/Tracking/Z-GMOT[official]/AC-Sort/tracking_result/img/{video_name}'

    try:
        with open(resutl_path, 'r') as f:
            results = f.readlines()
            for i in range(len(results)):
                results[i] = results[i].split(',')
                for j in range(6):
                    results[i][j] = int(float(results[i][j])) 
    except:
        continue

    frames = {}
    for i in range(len(results)):
        frame_id = results[i][0]
        if frame_id in frames:
            continue
        frames[frame_id] = {}
        frames[frame_id]["obj_ids"] = []
        frames[frame_id]["tlwhs"] = []
        for j in range(len(results)):
            if results[j][0] == frame_id:
                frames[frame_id]["obj_ids"].append(round(float(results[j][1]), 2))
                frames[frame_id]["tlwhs"].append(results[j][2:6])
    print(video_name, len(frames.keys()))

    os.makedirs(vis_folder, exist_ok=True)
    # print(original_folder)
    for frame_file in os.listdir(original_folder):
        try:
            frame_id = int(frame_file[:-4])
        except:
            print(frame_file)
            continue
        if frame_id not in frames:
            continue
        img = cv2.imread(osp.join(original_folder, frame_file))
        tlwhs = frames[frame_id]["tlwhs"]
        obj_ids = frames[frame_id]["obj_ids"]
        img = plot_tracking(img, tlwhs, obj_ids, frame_id)
        cv2.imwrite(osp.join(vis_folder, frame_file), img)