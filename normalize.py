import cv2 as cv
import numpy as np


def normalize(frame, camera_matrix, head_pose, gaze_origin, is_face=False):
    # Output parameters (for the eyes)
    n_focal_length = 1800
    n_distance = 600
    n_w, n_h = 128, 128
    if is_face:
        n_focal_length = 1200
        n_distance = 600
        n_w, n_h = 256, 256
    n_camera_matrix = np.array([
        [n_focal_length, 0.0, 0.5*n_w],
        [0.0, n_focal_length, 0.5*n_h],
        [0, 0, 1],
    ], dtype=np.float64)

    # Calculate rotation matrix and euler angles
    rvec, tvec = head_pose
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)
    rotate_mat, _ = cv.Rodrigues(rvec)

    # Form initial gaze in camera coordinates system
    g_o = gaze_origin.reshape(3, 1)

    # Code below is an adaptation of code by Xucong Zhang
    # https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation/

    distance = np.linalg.norm(g_o)  # actual distance between eye and original camera
    z_scale = n_distance / distance
    S = np.eye(3, dtype=np.float64)
    S[2, 2] = z_scale

    hRx = rotate_mat[:, 0]
    forward = (g_o / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(n_camera_matrix, S),
               np.dot(R, np.linalg.inv(camera_matrix)))  # transformation matrix

    patch = cv.warpPerspective(frame, W, (n_w, n_h))  # image normalization

    R = np.asmatrix(R)

    # Correct head pose
    head_mat = R * rotate_mat
    n_h = np.array([np.arcsin(head_mat[1, 2]), np.arctan2(head_mat[0, 2], head_mat[2, 2])])

    # Correct gaze
    return patch, n_h, R, W


def vector_to_pitchyaw(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def pitchyaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def apply_transformation(T, vec):
    h_vec = np.asmatrix(np.ones((4, 1)))
    h_vec[:3, 0] = vec.reshape(3, 1)
    return np.matmul(T, h_vec)[:3, 0]


def apply_rotation(T, vec):
    new_T = np.asmatrix(np.eye(4))
    new_T[:3, :3] = T[:3, :3]
    return apply_transformation(new_T, vec)


def get_intersect_with_zero(o, g):
    """Intersects a given gaze ray (origin o and direction g) with z = 0."""
    o = np.asarray(o).reshape(3, 1)
    g = np.asarray(g).reshape(3, 1)
    n = np.array([0.0, 0.0, 1.0]).reshape(3, 1)  # plane normal
    a = np.array([1.0, 0.0, 0.0]).reshape(3, 1)  # another point on plane
    denom = np.sum(np.multiply(g, n), axis=0)
    numer = np.sum(np.multiply(a - o, n), axis=0)
    t = np.divide(numer, denom)
    por = o + t.reshape(-1, 1) * g
    return por[:2, 0]  # Return just x, y (or u, v depending on notation)


def validate_gaze(origin=None, direction=None, rotation=None, target_2D=None,
                  camera_transformation=None, inv_camera_transformation=None,
                  pixels_per_millimeter=None):
    origin = np.asmatrix(origin).reshape(3, 1)
    direction = direction.reshape(1, 2)
    direction = np.asmatrix(pitchyaw_to_vector(direction)).reshape(3, 1)
    rotation = np.asmatrix(rotation).reshape(3, 3)
    camera_transformation = np.asmatrix(camera_transformation)
    inv_camera_transformation = np.asmatrix(inv_camera_transformation)

    # Negate gaze vector back (to camera perspective)
    direction = -direction

    # De-rotate gaze vector
    direction = np.matmul(rotation.T, direction)

    # Transform values
    direction = apply_rotation(inv_camera_transformation, direction)
    origin = apply_transformation(inv_camera_transformation, origin)

    # Intersect with z = 0
    recovered_target_2D = get_intersect_with_zero(origin, direction)

    # Convert back from mm to pixels
    ppm_w, ppm_h = pixels_per_millimeter
    recovered_target_2D[0] *= ppm_w
    recovered_target_2D[1] *= ppm_h

    return np.all(np.isclose(recovered_target_2D, target_2D))
