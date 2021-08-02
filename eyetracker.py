import cv2
import numpy as np
import ffmpeg
import torch

from eve import EVE
from config import Config as cfg


class EyeTracker:

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.eve = EVE().to(self.device)
        self.extractor = None

    def preprocess(self, face, scene):
        x = None
        return x

    def __call__(self, x):
        pass

    def postprocess(self, x):
        stuff = None
        return stuff

