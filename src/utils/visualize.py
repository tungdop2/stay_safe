#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np
import torch

from distance.distance import load_top_view_config

__all__ = ["vis"]

M = np.array(M)

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, heads, faces, frame_id=0, fps=0., limit=10):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    # top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1080.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 1080.))
    # text_scale = 2
    # text_thickness = 2
    # line_thickness = 2

    for i, face in enumerate(faces):
        x1, y1, w, h, prob = face
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        color = (255, 255, 0)
        if prob < 0.7:
            color = (0, 0, 255)
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
    M, w_scale, h_scale = load_top_view_config('distance/distance.txt')
    for i in range(len(heads) - 1):
        for j in range(i + 1, len(heads)):
            bc1 = [heads[i][0] + heads[i][2] / 2, heads[i][1] + heads[i][3]]
            bc2 = [heads[j][0] + heads[j][2] / 2, heads[j][1] + heads[j][3]]

            bc1_ = cv2.perspectiveTransform(np.array([[bc1]]), M)[0][0]
            bc2_ = cv2.perspectiveTransform(np.array([[bc2]]), M)[0][0]
            dw = np.abs(bc1_[0] - bc2_[0]) / w_scale
            dh = np.abs(bc1_[1] - bc2_[1]) / h_scale
            dist = np.sqrt(dw * dw + dh * dh)
            if dist < 1.5:
                heads[i][4] = 0
                heads[j][4] = 0
                cv2.line(im, tuple(map(int, bc1)), tuple(map(int, bc2)), (0, 0, 255), line_thickness)

    for i, person in enumerate(heads):
        x1, y1, w, h, tag = person
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        # id_text = '{} {:.2f}'.format(int(obj_id), scores[i]) if scores is not None else '{}'.format(int(obj_id))
        # if ids2 is not None:
        #     id_text = id_text + ', {}'.format(int(ids2[i]))
        color = (0, 255, 0)
        if tag == 0:
            color = (0, 0, 255)
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.circle(im, (int(x1 + w / 2), int(y1 + h)), 2, color=(255, 255, 255), thickness=-1)

    cv2.putText(im, 'frame: %d fps: %.2f' % (frame_id, fps),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=text_thickness)
    text_color = (0, 255, 0)
    if len(heads) > limit:
        text_color = (0, 0, 255)
    cv2.putText(im, 'People: {} / {}'.format(len(heads), limit), (0, int(30 * text_scale)),
                cv2.FONT_HERSHEY_PLAIN, 2, text_color, thickness=text_thickness)
    return im


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
