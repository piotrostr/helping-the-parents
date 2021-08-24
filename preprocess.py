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


def get_origin_of_gaze(preds_3d):
    """
    :param preds: 3d landmarks predictions
    """
    preds = np.array(preds_3d)
    left_coords = preds[slice(36, 42)]
    right_coords = preds[slice(42, 48)]
    left_o = [
        left_coords[:, 0].mean(), 
        left_coords[:, 1].mean(),
        left_coords[:, 2].mean()
    ]
    right_o = [
        right_coords[:, 0].mean(), 
        right_coords[:, 1].mean(),
        right_coords[:, 2].mean()
    ]
    return left_o, right_o


def get_frames_and_timestamps(path):
    cap = cv2.VideoCapture(path)
    frames = []
    timestamps = []
    frame_count = 0
    if cap.isOpened():
        print(f'Parsing frames from {path}...')
        for i in range(1000):
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = cap.read()
            frame_count += 1
            try:
                frame.shape
            except:
                break
            if timestamp == 0.0:
                fps = cap.get(cv2.CAP_PROP_FPS)
                timestamp = (float(frame_count) / fps) * 1000.
            timestamps.append(timestamp)
            if 'scene' in path:
                frame = cv2.resize(frame, dsize=(128, 72))
            frames.append(frame)
            print('.', end='')
        print('\nDone!')
    return np.stack(frames), [make_weird(t) for t in timestamps]


def normalize(img, mtx, dist, landmarks, gc):
    landmarks = np.array(landmarks)
    _img = img.copy()
    _img = cv2.undistort(_img, mtx, dist)
    face = sio.loadmat('./faceModelGeneric.mat')['model']
    face_pts = face.T.reshape(face.shape[1], 1, 3)
    hr, ht = estimateHeadPose(landmarks, face_pts, mtx, dist)
    hR, data = normalizeData(img, face, hr, ht, gc, mtx)
    return hR, data


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


def solve_extrinsic_matrix(corners_world, corners_image, intrinsic_matrix, distortion):
    """
    https://github.com/HViktorTsoi/ACSC/
    """
    assert len(corners_world) == len(corners_image)
    corners_world = np.array(corners_world)
    corners_image = np.array(corners_image)
    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        corners_world, corners_image, intrinsic_matrix, distortion
    )
    rotation_m, _ = cv2.Rodrigues(rvec)
    extrinsic_matrix = np.hstack([rotation_m, tvec])
    extrinsic_matrix = np.vstack([extrinsic_matrix, [0, 0, 0, 1]])
    return extrinsic_matrix, rvec, tvec

