import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio

from utils import make_weird
from normalize_data import estimateHeadPose, normalizeData


def preprocess_frames(frames):
    """
    :param frames: shape N x H x W x C
    :returns frames: shape: N x C x H x W
    """
    frames = np.transpose(frames, [0, 3, 1, 2])
    frames = frames.astype(np.float32)
    frames *= 2.0 / 255.0
    frames -= 1.0
    frames = np.expand_dims(frames, axis=0)
    return frames


def get_frames_and_timestamps(path):
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
                frames = np.concatenate((frames, np.expand_dims(frame, axis=0)), axis=0)
            print('.', end='')
    print('\nDone!')
    return frames, list(map(make_weird, timestamps))


def normalize(img, mtx, dist, landmarks, gc):
    landmarks = np.array(landmarks)
    _img = img.copy()
    _img = cv2.undistort(_img, mtx, dist)
    face = sio.loadmat('./faceModelGeneric.mat')['model']
    face_pts = face.T.reshape(face.shape[1], 1, 3)
    hr, ht = estimateHeadPose(landmarks, face_pts, mtx, dist)
    data = normalizeData(img, face, hr, ht, gc, mtx)
    return data


def calibrate(images):
    """
    :returns: (ret, mtx, dist, rvecs, tvecs)
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    w, h = gray.shape[::-1]
    return cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)


def undistort_image(img, mtx, dist):
    plt.imshow(img)
    plt.show()
    h, w = img.shape[:2]
    params = [mtx, dist, (w, h), 1, (w, h)]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(*params)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    plt.imshow(dst)
    plt.show()
    undistorted = dst[y:y+h, x:x+w]
    plt.imshow(undistorted)
    plt.show()
    return undistorted

