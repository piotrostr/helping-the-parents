import cv2
import numpy as np
import ffmpeg
import torch
import glob

from eve import EVE
from EVE.src.datasources.eve_sequences import EVESequencesBase
from preprocess import (
    calibrate, 
    normalize, 
    undistort_image, 
    preprocess_frames
)


class EyeTracker:

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.eve = EVE().to(self.device)
        self.images = []
        fa2d = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                            flip_input=False,
                                            device=str(self.device),
                                            face_detector='blazeface')
        fa3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                            flip_input=False,
                                            device=str(self.device),
                                            face_detector='blazeface')

    def load_images(self):
        self.images = [cv2.imread(f) for f in glob.glob('./data/*.jpg')]
        self.images = [cv2.resize(img, (1920, 1080)) for img in self.images]

    def move_to_right_device(self, inp):
        for k, v in inp.items():
            inp[k] = v.to(self.device)

    def create_inputs(self) -> dict:
        i = {}
        self.load_images()

        ret, mtx, dist, rvecs, tvecs = calibrate(self.images)
        
        # todo automate the below for each frame

        left_o, right_o = get_origin_of_gaze(preds_3d)
        i['left_o'] = torch.Tensor([left_o]).unsqueeze(0)
        i['left_o_validity'] = torch.Tensor([True]).unsqueeze(0)
        i['right_o'] = torch.Tensor([right_o]).unsqueeze(0)
        i['right_o_validity'] = torch.Tensor([True]).unsqueeze(0)

        i['camera_transformation'] = torch.Tensor([ext_matrix]).unsqueeze(0)
        inv = np.linalg.inv(extrinsic_matrix)   
        i['inv_camera_transformation'] = torch.Tensor([inv]).unsqueeze(0)

        i['left_R'] = torch.Tensor([left_R]).unsqueeze(0)

        i['left_R_validity'] = torch.Tensor([True]).unsqueeze(0)

        i['right_R'] = torch.Tensor([right_R]).unsqueeze(0)

        i['right_R_validity'] = torch.Tensor([True]).unsqueeze(0)

        i['head_R'] = torch.Tensor([head_R]).unsqueeze(0)

        i['millimeters_per_pixel'] = inp['millimeters_per_pixel'][:, 0].unsqueeze(0)

        i['pixels_per_millimeter'] = inp['pixels_per_millimeter'][:, 0].unsqueeze(0)


        _left_eye_patch = preprocess_frames(np.expand_dims(left_eye_patch, axis=0))
        i['left_eye_patch'] = torch.from_numpy(_left_eye_patch)

        _right_eye_patch = preprocess_frames(np.expand_dims(right_eye_patch, axis=0))
        i['right_eye_patch'] = torch.from_numpy(_right_eye_patch)

        i['timestamps'] = torch.Tensor([925789010258708]).unsqueeze(0)

        cap = cv2.VideoCapture('./data/scene.mp4')
        ret, frame = cap.read()
        screen_frame = cv2.resize(frame, dsize=(128, 72))

        _screen_frame = preprocess_frames(np.expand_dims(screen_frame, axis=0))
        i['screen_frame'] = torch.from_numpy(_screen_frame)

        i['screen_timestamps'] = torch.Tensor([925789010258708]).unsqueeze(0)
        return inputs

    def postprocess(self, x):
        stuff = None
        return stuff
    
    @statimethod
    def preprocess_frames(frames):
        # Expected input:  N x H x W x C
        # Expected output: N x C x H x W
        frames = np.transpose(frames, [0, 3, 1, 2])
        frames = frames.astype(np.float32)
        frames *= 2.0 / 255.0
        frames -= 1.0
        frames = np.expand_dims(frames, axis=0)
        return frames

    @staticmethod
    def make_weird(x):
        try:
            return (float(x) / 1e-6) + 9.257890218838680000e+14
        except:
            return x

if __name__ == '__main__':
    dataset = EVESequencesBase(
        'sample/eve_dataset',
        participants_to_use=['train01']
    )
    dataloader = torch.utils.data.DataLoader(dataset)
    
    eyetracker = EyeTracker()
    
    inp = next(iter(dataloader))
    out = eve(inp)


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

