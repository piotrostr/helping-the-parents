import cv2
import numpy as np
import ffmpeg
import torch
import glob
import pickle

from torch import Tensor
from scipy import io as sio
from face_alignment import FaceAlignment, LandmarksType
from eve import EVE
from utils import true_tensor
from preprocess import (
    calibrate, 
    normalize, 
    undistort_image, 
    preprocess_frames,
    get_origin_of_gaze,
    get_frames_and_timestamps
)


class EyeTracker:

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.load_models()
        self.load_calibration_images()
        
    def load_models(self):
        self.fa2d = FaceAlignment(LandmarksType._2D,
                                  flip_input=False,
                                  device=str(self.device),
                                  face_detector='blazeface')
        self.fa3d = FaceAlignment(LandmarksType._3D,
                                  flip_input=False,
                                  device=str(self.device),
                                  face_detector='blazeface')
        self.eve = EVE().to(self.device)
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
        self.inputs['left_o'] = []
        self.inputs['left_o_validity'] = []
        self.inputs['right_o'] = []
        self.inputs['right_o_validity'] = []
        self.inputs['left_R'] = []
        self.inputs['left_R_validity'] = []
        self.inputs['right_R'] = []
        self.inputs['right_R_validity'] = []
        self.inputs['head_R'] = []
        self.inputs['left_eye_patch'] = []
        self.inputs['right_eye_patch'] = []
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
            [preds] = self.fa2d.get_landmarks(face_frame)
            [preds_3d] = self.fa3d.get_landmarks(face_frame)
            landmarks = [
                preds[36], preds[39],  # left eye corners
                preds[42], preds[45],  # right eye corners
                preds[48], preds[54]   # mouth corners
            ]
            ret, rvec, tvec = cv2.solvePnP(face_pts, 
                                           np.array(landmarks),
                                           mtx, 
                                           dist,
                                           flags=cv2.SOLVEPNP_SQPNP)
            rotation_m, _ = cv2.Rodrigues(rvec)
            ext_mtx = np.hstack([rotation_m, tvec])
            ext_mtx = np.vstack([ext_mtx, [0, 0, 0, 1]])
            inv = np.linalg.inv(ext_mtx)   

            gc = np.array([-127.79, 4.62, -12.02])  # 3D gaze target position
            head_R, dat = normalize(face_frame, mtx, dist, landmarks, gc)
            left_eye, right_eye = dat
            left_eye_patch, _, _, left_R, left_W = left_eye
            right_eye_patch, _, _, right_R, right_W = right_eye
            left_o, right_o = get_origin_of_gaze(preds_3d)

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
    
    def infer(self, face_path: str, scene_path: str):
        self.initialize_inputs()
        self.fill_inputs(face_path, scene_path)
        self.preprocess_inputs()
        out = self.eve(self.inputs)
        return self.postprocess(out)


if __name__ == '__main__':
    eyetracker = EyeTracker()
    out = eyetracker.infer('./data/face.mp4', './data/scene.mp4')
    with open('out.pkl', 'wb') as f:
        pickle.dump(out, f)
    print(out['PoG_px_final'])

    """
    !!! preprocessing steps:

    1) intrinsic matrix calibration using opencv and ChArUco board 
    2) extrinsic camera calibration using mirrors [1]
    3) undistort the frames
    4) detect face
    5) detect face-region landmarks
    6) perform 3D morphable model (3DMM) to 3D landmarks [2]
    7) apply 'data normalization' for yielding eye patches [3, 4]
       under assumptions: 
           virtual camera is located 60cm away from the gaze origin
           focal length of 1800mm


    [1] https://www.jstage.jst.go.jp/article/ipsjtcva/8/0/8_11/_pdf/-char/en
    [2] https://openresearch.surrey.ac.uk/discovery/delivery/44SUR_INST:ResearchRepository/12139198320002346#13140605970002346
    [3] https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Sugano_Learning-by-Synthesis_for_Appearance-based_2014_CVPR_paper.pdf
    [4] https://www.perceptualui.org/publications/zhang18_etra.pdf 
    """

