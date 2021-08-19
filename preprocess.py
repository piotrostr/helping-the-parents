import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio

from normalize_data import estimateHeadPose, normalizeData


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

