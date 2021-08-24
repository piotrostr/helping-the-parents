import cv2
import numpy as np
import ffmpeg
import torch
import glob

from torch import Tensor
from face_alignment import FaceAlignment, LandmarksType
from eve import EVE
from EVE.src.datasources.eve_sequences import EVESequencesBase
from preprocess import (
    calibrate, 
    normalize, 
    undistort_image, 
    preprocess_frames,
    get_origin_of_gaze
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


    def load_calibration_images(self):
        self.images = [cv2.imread(f) for f in glob.glob('./data/*.jpg')]
        self.images = [cv2.resize(img, (1920, 1080)) for img in self.images]

    def move_to_right_device(self, inputs: dict):
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

    def create_inputs(self, face_path: str, scene_path: str) -> dict:
        cap_face = cv2.VideoCapture(face_path)
        ret, frame = cap.read()
        cap_screen = cv2.VideoCapture(scene_path)
        # todo go smart about stacking the below
        inputs = {}
        ret, mtx, dist, rvecs, tvecs = calibrate(self.images)
        
        # constant across frames
        mm_per_px = [0.2880, 0.2880]
        px_per_mm = [3.4720, 3.4727]
        inputs['millimeters_per_pixel'] = Tensor([mm_per_px]).unsqueeze(0)
        inputs['pixels_per_millimeter'] = Tensor([px_per_mm]).unsqueeze(0)

        # unique for each frame: 
        [preds] = self.fa2d.get_landmarks(img)
        [preds_3d] = self.fa3d.get_landmarks(img)
        landmarks = [
            preds[36], preds[39],  # left eye corners
            preds[42], preds[45],  # right eye corners
            preds[48], preds[54]   # mouth corners
        ]
        face = sio.loadmat('./faceModelGeneric.mat')['model']
        face_pts = face.T.reshape(face.shape[1], 1, 3)
        ret, rvec, tvec = cv2.solvePnP(face_pts, 
                                       np.array(landmarks),
                                       mtx, 
                                       dist,
                                       flags=cv2.SOLVEPNP_SQPNP)
        rotation_m, _ = cv2.Rodrigues(rvec)
        ext_mtx = np.hstack([rotation_m, tvec])
        ext_mtx = np.vstack([ext_mtx, [0, 0, 0, 1]])
        inputs['camera_transformation'] = Tensor([ext_mtx]).unsqueeze(0)
        inv = np.linalg.inv(ext_mtx)   
        inputs['inv_camera_transformation'] = Tensor([inv]).unsqueeze(0)

        gc = np.array([-127.79, 4.62, -12.02])  # 3D gaze taraget position
        head_R, [ 
            [left_eye_patch, _, _, left_R, left_W],
            [right_eye_patch, _, _, right_R, right_W]
        ] = head_R, data = normalize(img, mtx, dist, landmarks, gc)
        left_o, right_o = get_origin_of_gaze(preds_3d)
        inputs['left_o'] = Tensor([left_o]).unsqueeze(0)
        inputs['left_o_validity'] = Tensor([True]).unsqueeze(0)
        inputs['right_o'] = Tensor([right_o]).unsqueeze(0)
        inputs['right_o_validity'] = Tensor([True]).unsqueeze(0)

        inputs['left_R'] = Tensor([left_R]).unsqueeze(0)
        inputs['left_R_validity'] = Tensor([True]).unsqueeze(0)
        inputs['right_R'] = Tensor([right_R]).unsqueeze(0)
        inputs['right_R_validity'] = Tensor([True]).unsqueeze(0)
        inputs['head_R'] = Tensor([head_R]).unsqueeze(0)

        t0 = 925789010258708
        _left_eye_patch = np.expand_dims(left_eye_patch, axis=0)
        _left_eye_patch = preprocess_frames(_left_eye_patch)
        inputs['right_eye_patch'] = Tensor(_left_eye_patch)
        inputs['timestamps'] = Tensor([t0]).unsqueeze(0)

        _right_eye_patch = np.expand_dims(right_eye_patch, axis=0)
        _right_eye_patch = preprocess_frames(_right_eye_patch)
        inputs['right_eye_patch'] = Tensor(_right_eye_patch)
        inputs['timestamps'] = Tensor([t0]).unsqueeze(0)

        _screen_frame = cv2.resize(screen_frame, dsize=(128, 72))
        _screen_frame = np.expand_dims(screen_frame, axis=0)
        _screen_frame = preprocess_frames(_screen_frame)
        inputs['screen_frame'] = Tensor(_screen_frame)
        inputs['screen_timestamps'] = Tensor([t0]).unsqueeze(0)

        return self.move_to_right_device(inputs)

    def postprocess(self, x):
        stuff = None
        return stuff


if __name__ == '__main__':
    eyetracker = EyeTracker()


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

