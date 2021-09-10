import cv2
import numpy as np
import ffmpeg
import torch
import glob
import pickle

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
        print('Inputs preprocessed.')

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
        screen_w = 1920
        screen_h = 1080
        ppm = 8.93  # 227 px per inch
        mpp = 1 / ppm  
            self.inputs['facial_landmarks'].append(Tensor(landmarks_2d))

        ret, mtx, dist, rvecs, tvecs = calibrate(self.images)

        face_frames, timestamps = get_frames_and_timestamps(face)
        screen_frames, screen_timestamps = get_frames_and_timestamps(scene)
        n, h, w, c = face_frames.shape

        self.inputs['timestamps'] = Tensor(timestamps)
        self.inputs['screen_timestamps'] = Tensor(screen_timestamps)
        self.inputs['millimeters_per_pixel'] = [Tensor([0.2880, 0.2880])
                                                for i in range(n)]
        self.inputs['pixels_per_millimeter'] = [Tensor([3.4720, 3.4727])
                                                for i in range(n)]
        self.inputs['left_o_validity'] = true_tensor(n) 
        self.inputs['right_o_validity'] = true_tensor(n)
        self.inputs['left_R_validity'] = true_tensor(n)
        self.inputs['right_R_validity'] = true_tensor(n)

        face = sio.loadmat('./faceModelGeneric.mat')['model']
        face_pts = face.T.reshape(face.shape[1], 1, 3)

        for face_frame, screen_frame in zip(face_frames, screen_frames):
            face_frame = cv2.resize(face_frame, (1920, 1080))
            frame = face_frame.copy()

            # Read intrinsics -> camera_matrix, distortion
            # TODO do this with ai
            camera_matrix, distortion = mtx, dist

            # undistort
            frame = undistort(frame, camera_matrix, distortion, input_w, input_h)

            # get transf and inverse transf from extrinsincs
            # TODO

            # detect face and landmarks
            [landmarks_3d] = fa.get_landmarks(frame)
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

            face = normalize(frame, camera_matrix, head_pose, o_f, is_face=True)
            face_patch, face_head, face_R, face_W  = face

            self.inputs['camera_transformation'].append(Tensor(ext_mtx))
            self.inputs['inv_camera_transformation'].append(Tensor(inv))
            self.inputs['left_o'].append(Tensor(left_o))
            self.inputs['right_o'].append(Tensor(right_o))
            self.inputs['left_R'].append(Tensor(left_R))
            self.inputs['right_R'].append(Tensor(right_R))
            self.inputs['head_R'].append(Tensor(head_R))
            self.inputs['left_eye_patch'].append(left_eye_patch)
            self.inputs['right_eye_patch'].append(right_eye_patch)
            self.inputs['screen_frame'].append(Tensor(screen_frame))
        print(f'Input dict filled with data from {n} frames.')

    def postprocess(self, x):
        return x
    
    def __call__(self, face_path: str, scene_path: str):
        self.initialize_inputs()
        self.fill_inputs(face_path, scene_path)
        self.preprocess_inputs()
        out = self.eve(self.inputs)
        return self.postprocess(out)


if __name__ == '__main__':
    eyetracker = EyeTracker()
    out = eyetracker('./data/face.mp4', './data/scene.mp4')
    for k, v in out.items():
        out[k] = v.detach().cpu()
    with open('out.pkl', 'wb') as f:
        pickle.dump(out, f)
    print(out['PoG_px_final'])

