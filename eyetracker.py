import cv2
import numpy as np
import ffmpeg
import torch

from eve import EVE
from EVE.src.datasources.eve_sequences import EVESequencesBase


class EyeTracker:

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.eve = EVE().to(self.device)
        self.extractor = None

    def create_inputs(self, face_vid, scene_vid, mouse_position) -> dict:
        inputs = {}
        cap = cv2.VideoCapture(path)
        frames = np.array([])
        timestamps = []
        frame_count = 0
        if cap.isOpened():
            print('Parsing frames...')
            while True:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                ret, frame = cap.read()
                frame_count += 1
                try:
                    frame.shape
                except:
                    break
                if not len(frames):
                    frames = np.expand_dims(frame, axis=0)
                else:
                    if timestamp == 0.0:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        timestamp = (float(frame_count) / fps) * 1000.
                    timestamps.append(timestamp)
                    frame = np.expand_dims(frame, axis=0)
                    frames = np.concatenate((frames, frame), axis=0)
                print('.', end='')
        else:
            raise Exception('Capture could not be opened.')
        print('\nDone!')
        frames = self.preprocess_frames(frames)
        alpha = 1780
        _, n, c, h, w = frames.shape
        single_camera_matrix = np.array([
            [alpha, 0, w / 2], 
            [0, alpha, h / 2], 
            [0, 0, 1]
        ])
        camera_matrix = torch.Tensor([camera_matrix for i in range(n)])
        camera_matrix = camera_matrix.type(torch.float32).unsqueeze(0)
        timestamps = list(map(self.make_weird, timestamps))
        return frames, timestamps

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

