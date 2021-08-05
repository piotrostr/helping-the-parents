#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse 

from blazeface import BlazeFace


class Eyes:
    def __init__(
        self, 
        right_x: float, 
        right_y: float, 
        left_x: float, 
        left_y: float
    ):
        self.right_x = right_x
        self.right_y = right_y 
        self.left_x = left_x
        self.left_y = left_y


def crop_eyes(img_original: np.ndarray, eyes: Eyes):
    # get eyes cords in terms of px
    eyes.right_x *= 128
    eyes.right_y *= 128
    eyes.left_x *= 128
    eyes.left_y *= 128

    # multiply by ratio to get cords on original img
    h, w, _ = img_original.shape
    ratio_h = h / 128
    ratio_w = w / 128
    eyes.right_x = int(eyes.right_x * ratio_w)
    eyes.right_y = int(eyes.right_y * ratio_h)
    eyes.left_x = int(eyes.left_x * ratio_w)
    eyes.left_y = int(eyes.left_y * ratio_h)

    # crop out
    eye_right = img_original[eyes.right_y - 64: eyes.right_y + 64,
                             eyes.right_x - 64: eyes.right_x + 64]
    eye_left = img_original[eyes.left_y - 64: eyes.left_y + 64,
                             eyes.left_x - 64: eyes.left_x + 64]
    return np.concatenate((eye_right, eye_left), axis=1)
    

def crop_eyes_of_video(path, output_path):
    blaze = BlazeFace()
    blaze.load_weights('models/blazeface.pth')
    blaze.load_anchors('models/anchors.npy')
    blaze.min_score_thresh = 0.75
    blaze.min_supppression_threshold = 0.3

    # TODO load tinaface

    cap = cv2.VideoCapture(path)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(framerate), (256, 128))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _frame = cv2.resize(_frame, (128, 128))
        # TODO crop the face
        dets = blaze.predict_on_image(_frame)
        _, _, _, _, right_x, right_y, left_x, left_y, *_ = dets.flatten()
        eyes_bit = crop_eyes(frame, Eyes(right_x, right_y, left_x, left_y))
        out.write(eyes_bit)
    cap.release()
    out.release()
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='video path')
    parser.add_argument('output_path', type=str, help='output path')
    args = parser.parse_args()

    crop_eyes_of_video(args.path, args.output_path)

