"""
Copyright 2019 - 2020 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os

import cv2 as cv
import eos
import numpy as np


def landmarks_to_eos(landmarks):
    out = []
    for i, (x, y) in enumerate(landmarks[:68, :]):
        out.append(eos.core.Landmark(str(i + 1), [x, y]))
    return out


class HeadPoseEstimator(object):

    def __init__(self):
        cwd = os.path.dirname(__file__)
        base_dir = cwd + '/eos'

        # Morphable model definition
        model = eos.morphablemodel.load_model(base_dir + '/share/sfm_shape_3448.bin')
        self.blendshapes = eos.morphablemodel.load_blendshapes(
            base_dir + '/share/expression_blendshapes_3448.bin')
        self.morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(
            model.get_shape_model(),
            self.blendshapes,
            model.get_color_model(),
            None,
            model.get_texture_coordinates(),
        )
        self.landmark_mapper = eos.core.LandmarkMapper(
            base_dir + '/share/ibug_to_sfm.txt')
        self.edge_topology = eos.morphablemodel.load_edge_topology(
            base_dir + '/share/sfm_3448_edge_topology.json')
        self.contour_landmarks = eos.fitting.ContourLandmarks.load(
            base_dir + '/share/ibug_to_sfm.txt')
        self.model_contour = eos.fitting.ModelContour.load(
            base_dir + '/share/sfm_model_contours.json')

    def mesh_fit(self, frame, landmarks, num_iterations=5):
        # Fit morphable SfM model
        h, w, _ = frame.shape
        [
            eos_mesh, eos_pose, eos_shape_coeffs, eos_blendshape_coeffs
        ] = eos.fitting.fit_shape_and_pose(
            self.morphablemodel_with_expressions, landmarks_to_eos(landmarks),
            self.landmark_mapper, w, h, self.edge_topology,
            self.contour_landmarks, self.model_contour,
            num_iterations=num_iterations,
        )
        return eos_mesh, eos_pose, eos_shape_coeffs, eos_blendshape_coeffs

    def head_pose_fit(self, landmarks_2D, deformed_mesh, camera_matrix, scaling_factor=1.0):
        # Get all deformed mesh vertices
        mesh_vertices = np.asarray(deformed_mesh.vertices)
        mesh_vertices *= scaling_factor

        # Rotate face around
        rotate_mat = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        mesh_vertices = np.matmul(mesh_vertices.reshape(-1, 3), rotate_mat)

        # Subset mesh coordinates for PnP steps
        sfm_points_ibug_subset = np.asarray([
            mesh_vertices[int(self.landmark_mapper.convert(str(d)))]
            for d in range(1, 69)
            if self.landmark_mapper.convert(str(d)) is not None
        ])
        landmarks_2D = np.asarray([
            landmarks_2D[d - 1, :] for d in range(1, 69)
            if self.landmark_mapper.convert(str(d)) is not None
        ])

        # Initial fit
        success, rvec, tvec, inliers = cv.solvePnPRansac(
            sfm_points_ibug_subset, landmarks_2D, camera_matrix, None,
            flags=cv.SOLVEPNP_EPNP)

        # Second fit for higher accuracy
        success, rvec, tvec = cv.solvePnP(
            sfm_points_ibug_subset, landmarks_2D, camera_matrix, None,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            flags=cv.SOLVEPNP_ITERATIVE)

        # Reproject deformed mesh keypoints for later confirmation
        reprojected_points, _ = \
            cv.projectPoints(sfm_points_ibug_subset, rvec, tvec, camera_matrix, None)
        reprojected_points = reprojected_points.reshape(-1, 2)

        # Define 3D gaze origin coordinates
        # Ref. iBUG -to- SfM indices
        #       37   =>  177  # right eye outer-corner (1)
        #       40   =>  181  # right eye inner-corner (5)
        #       43   =>  614  # left eye inner-corner (8)
        #       46   =>  610  # left eye outer-corner (2)
        #       31   =>  114  # nose tip (3)
        #       49   =>  398  # right mouth corner (12)
        #       55   =>  812  # left mouth corner (13)
        o_r = np.mean([mesh_vertices[177], mesh_vertices[181]], axis=0)
        o_l = np.mean([mesh_vertices[614], mesh_vertices[610]], axis=0)
        o_f = np.mean([mesh_vertices[177], mesh_vertices[181],
                       mesh_vertices[614], mesh_vertices[610],
                       mesh_vertices[398], mesh_vertices[812]], axis=0)

        return rvec, tvec, reprojected_points, o_l, o_r, o_f

    def __call__(self, frame, landmarks, camera_matrix, target_io_dist=None, visualize=False):
        # Fit morphable SfM model
        h, w, _ = frame.shape
        [
            eos_mesh, eos_pose, eos_shape_coeffs, eos_blendshape_coeffs
        ] = self.mesh_fit(frame, landmarks)

        # Set scaling factor as necessary
        scaling_factor = 1.0
        if target_io_dist is not None:
            current_io_dist = np.linalg.norm(eos_mesh.vertices[177] - eos_mesh.vertices[610])
            scaling_factor = target_io_dist / current_io_dist

        # Do PnP-based head pose fitting
        rvec, tvec, reprojected_points, o_l, o_r, o_f = \
            self.head_pose_fit(landmarks, eos_mesh, camera_matrix, scaling_factor)
        if visualize:
            o_r_2D = cv.projectPoints(o_r, rvec, tvec, camera_matrix, None)[0].reshape(2)  # noqa
            o_l_2D = cv.projectPoints(o_l, rvec, tvec, camera_matrix, None)[0].reshape(2)  # noqa
            o_f_2D = cv.projectPoints(o_f, rvec, tvec, camera_matrix, None)[0].reshape(2)  # noqa

        # Transform gaze origins into the camera coordinate system
        transform = np.asmatrix(np.eye(4))
        transform[:3, :3] = cv.Rodrigues(rvec)[0]
        transform[:3, 3] = tvec
        o_r = np.asarray(np.matmul(transform, np.asmatrix([*o_r, 1.0]).reshape(-1, 1)))[:3, 0]
        o_l = np.asarray(np.matmul(transform, np.asmatrix([*o_l, 1.0]).reshape(-1, 1)))[:3, 0]
        o_f = np.asarray(np.matmul(transform, np.asmatrix([*o_f, 1.0]).reshape(-1, 1)))[:3, 0]

        if visualize:
            # # Visualize textured face mesh
            # isomap = eos.render.extract_texture(eos_mesh, eos_pose, frame)
            # isomap = cv.cvtColor(isomap, cv.COLOR_BGRA2BGR)
            # isomap = np.transpose(isomap, [1, 0, 2])
            # cv.imshow('isomap', isomap)

            # Label landmarks in frame
            frame = np.copy(frame)
            for landmark in reprojected_points:
                cv.drawMarker(frame, tuple([int(l) for l in landmark]), color=(0, 0, 255),
                              markerType=cv.MARKER_STAR, markerSize=7, thickness=1,
                              line_type=cv.LINE_AA)

            # Project determined 3D gaze origins and draw markers
            cv.drawMarker(frame, (int(o_l_2D[0]), int(o_l_2D[1])), color=(255, 0, 0),
                          markerType=cv.MARKER_CROSS, markerSize=7, thickness=1,
                          line_type=cv.LINE_AA)
            cv.drawMarker(frame, (int(o_r_2D[0]), int(o_r_2D[1])), color=(255, 0, 0),
                          markerType=cv.MARKER_CROSS, markerSize=7, thickness=1,
                          line_type=cv.LINE_AA)
            cv.drawMarker(frame, (int(o_f_2D[0]), int(o_f_2D[1])), color=(255, 0, 0),
                          markerType=cv.MARKER_CROSS, markerSize=7, thickness=1,
                          line_type=cv.LINE_AA)

            # Rescale to display
            cv.imshow('tmp', cv.resize(frame, (960, 540)))
            cv.waitKey(1)

        return rvec, tvec, o_l, o_r, o_f
