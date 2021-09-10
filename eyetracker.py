import cv2
import numpy as np
import ffmpeg
import torch
import glob
import pickle
import argparse

from face_alignment import FaceAlignment, LandmarksType
from torch import Tensor
from scipy import io as sio
from eve import EVE
from utils import true_tensor
from preprocess import (
    calibrate, 
    undistort,
    preprocess_frames,
    get_frames_and_timestamps
)

from head_pose import HeadPoseEstimator
from normalize import normalize


class EyeTracker:

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.load_models()
        self.load_calibration_images()
        
    def load_models(self):
        self.fa = FaceAlignment(LandmarksType._3D,
                                  flip_input=False,
                                  device=str(self.device),
                                  face_detector='blazeface')
        self.eve = EVE().to(self.device)
        self.head_pose_estimator = HeadPoseEstimator()
        print('Models loaded.')

    def load_calibration_images(self):
        self.images = [cv2.imread(f) for f in glob.glob('./data/*.jpg')]
        self.images = [cv2.resize(img, (1920, 1080)) for img in self.images]
        assert len(self.images)
        print('Calibration images loaded.')

    def preprocess_inputs(self):
        for k, v in self.inputs.items():
            if 'validity' in k or 'timestamps' in k:
                self.inputs[k] = v.unsqueeze(0).to(self.device)
                continue
            if 'patch' in k or 'frame' in k:
                frames = np.stack(v)
                frames = preprocess_frames(frames)
                frames = Tensor(frames).to(self.device)
                self.inputs[k] = frames
                continue
            self.inputs[k] = torch.stack(v).unsqueeze(0).to(self.device)
        torch.save(self.inputs, 'inputs.pt')
        print('Inputs preprocessed and saved.')

    def initialize_inputs(self):
        self.inputs = {}
        self.inputs['millimeters_per_pixel'] = []
        self.inputs['pixels_per_millimeter'] = []
        self.inputs['camera_transformation'] = []
        self.inputs['inv_camera_transformation'] = []
        self.inputs['left_eye_patch'] = []
        self.inputs['left_h'] = []
        self.inputs['left_o'] = []
        self.inputs['left_R'] = []
        self.inputs['left_W'] = []
        self.inputs['right_eye_patch'] = []
        self.inputs['right_h'] = []
        self.inputs['right_o'] = []
        self.inputs['right_R'] = []
        self.inputs['right_W'] = []
        self.inputs['left_o_validity'] = []
        self.inputs['left_R_validity'] = []
        self.inputs['right_o_validity'] = []
        self.inputs['right_R_validity'] = []
        self.inputs['head_R'] = []
        self.inputs['timestamps'] = []
        self.inputs['screen_frame'] = []
        self.inputs['screen_timestamps'] = []
        print('Input dict initialized.')

    def fill_inputs(self, face: str, scene: str) -> dict:
        """
        :params face scene: 
            paths to captured videos of resp. webcam and screen
        :returns: 
            inputs dict ready to be fed straight into EVE
        """
        time_horizon = 1000  # ms
        decay_per_ms = 0.95

        # TODO get those params
        input_w = 1920
        input_h = 1080
        # those have to be deduced, 
        # based on that screen is not 2560x1600 227px/inch
        # but 1920x1080 ?px/inch
        screen_width_mm = 304.1  # 11.97 inches
        screen_height_mm = 212.3 # 8.36 inches

        ppm_w = input_w / screen_width_mm 
        ppm_h = input_h / screen_height_mm
        mpp_w = 1 / ppm_w
        mpp_h = 1 / ppm_h

        ret, mtx, dist, rvecs, tvecs = calibrate(self.images)

        face_frames, timestamps = get_frames_and_timestamps(face)
        screen_frames, screen_timestamps = get_frames_and_timestamps(scene)
        n, h, w, c = face_frames.shape

        self.inputs['timestamps'] = Tensor(timestamps)
        self.inputs['screen_timestamps'] = Tensor(screen_timestamps)
        self.inputs['millimeters_per_pixel'] = [Tensor([mpp_w, mpp_h])
                                                for i in range(n)]
        self.inputs['pixels_per_millimeter'] = [Tensor([ppm_w, ppm_h])
                                                for i in range(n)]
        self.inputs['left_o_validity'] = true_tensor(n) 
        self.inputs['right_o_validity'] = true_tensor(n)
        self.inputs['left_R_validity'] = true_tensor(n)
        self.inputs['right_R_validity'] = true_tensor(n)

        for face_frame, screen_frame in zip(face_frames, screen_frames):
            face_frame = cv2.resize(face_frame, (1920, 1080))
            frame = face_frame.copy()

            # Read intrinsics -> camera_matrix, distortion
            # TODO do this with ai
            camera_matrix, distortion = mtx, dist

            # undistort
            frame = undistort(frame, camera_matrix, distortion, input_w, input_h)

            # get transf and inverse transf from extrinsincs
            t_x = (screen_width_mm / 2) * ppm_w  # exactly in the center
            t_y = 5 * ppm_h  # half a centimeter
            camera_transformation = ext_mtx = [
                [-1, 0, 0, -t_x],
                [0, 1, 0, t_y],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
            inv_camera_transformation = inv = np.linalg.inv(ext_mtx)
            """
            Then, it should be possible to define the data field
            "camera_transformation" (from camera-perspective to screen-space),
            for example, as:
            where t_x is the horizontal offset (in mm) between camera-center and top-left screen-corner,
            and t_y is the vertical offset (in mm) between camera-center and top-left screen-corner,
            based on Fig. 2.1 (page 14) of my doctoral dissertation (see attached PDF),
            then apply a further scaling (multiply by pixels-per-millimeter) to arrive at the screen position in pixels.
            """

            # detect face and landmarks
            [landmarks_3d] = self.fa.get_landmarks(frame)
            landmarks_2d = landmarks_3d[:, :2]

            # smooth landmarks with ema
            # TODO

            # estimage head pose using head_pose_estimator
            out = self.head_pose_estimator(frame, landmarks_2d, camera_matrix)
            rvec, tvec, o_l, o_r, o_f = out
            head_pose = (rvec, tvec)
            head_R, jacobian = cv2.Rodrigues(rvec)
            gaze_origins_3D = [o_l, o_r, o_f]

            # smooth gaze origin with ema
            # TODO

            # Data Normalization procedure
            left = normalize(frame, camera_matrix, head_pose, o_l)
            left_eye_patch, left_head, left_R, left_W = left

            right = normalize(frame, camera_matrix, head_pose, o_r)
            right_eye_patch, right_head, right_R, right_W = right

            self.inputs['camera_transformation'].append(Tensor(ext_mtx))
            self.inputs['inv_camera_transformation'].append(Tensor(inv))
            self.inputs['left_eye_patch'].append(left_eye_patch)
            self.inputs['left_h'].append(Tensor(left_head))
            self.inputs['left_o'].append(Tensor(o_l))
            self.inputs['left_R'].append(Tensor(left_R))
            self.inputs['left_W'].append(Tensor(left_W))
            self.inputs['right_eye_patch'].append(right_eye_patch)
            self.inputs['right_h'].append(Tensor(right_head))
            self.inputs['right_o'].append(Tensor(o_r))
            self.inputs['right_R'].append(Tensor(right_R))
            self.inputs['right_W'].append(Tensor(right_W))
            self.inputs['head_R'].append(Tensor(head_R))
            self.inputs['screen_frame'].append(Tensor(screen_frame))
        print(f'Input dict filled with data from {n} frames.')

    def __call__(self, face_path: str, scene_path: str):
        self.initialize_inputs()
        self.fill_inputs(face_path, scene_path)
        self.preprocess_inputs()
        return self.eve(self.inputs)


def chunk(inputs: dict, size: int):
    _inputs = inputs.copy()
    for k, v in _inputs.items():
        _inputs[k] = v[:, :size]
        print(_inputs[k].shape)
    return _inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='input path')
    args = parser.parse_args()
    eyetracker = EyeTracker()
    if not args.input_path:
        out = eyetracker('./data/face.mp4', './data/scene.mp4')
        for k, v in out.items():
            out[k] = v.detach().cpu()
        with open('out.pkl', 'wb') as f:
            pickle.dump(out, f)
        print(out['PoG_px_final'])
    else:
        print('Using already create inputs.')
        inp = torch.load(args.input_path)
        inputs = chunk(inp, 100)
        out = eyetracker.eve(inp)
        print(out['PoG_px_final'])

