import cv2
import torch
import PIL.Image as pil
import numpy as np

from rdn import RDN

rdn = RDN()
rdn.load_state_dict(torch.load('./rdn.pth', map_location='cpu'))

img = pil.open('./data/1.jpg').convert('RGB')

lr = np.expand_dims(np.array(img).astype(np.float32).transpose([2, 0, 1]), 0) 
lr /= 255.0
hr = rdn(torch.from_numpy(lr))

