
import os
import cv2
import json
import numpy as np


def read_cam_param(cam_param):
    K = np.array(cam_param["camera_matrix"]).reshape((3, 3))
    dist_coeff = np.array(cam_param["distortion_coefficients"])

    R_wtc = np.array(cam_param["rotation_matrix"]).reshape((3, 3))
    R_wtv = np.array([0, -1, 0, 0, 0, -1, 1, 0, 0]).reshape((3, 3))
    R = R_wtv.dot(R_wtc.T)

    Tx = cam_param["cam_to_front"]
    Ty = (cam_param["cam_to_left"] - cam_param["cam_to_right"]) / 2
    T = np.array([Tx, Ty, cam_param["camera_height"]])

    return K, dist_coeff, R, T


def bird_view_proj(u, v, K, dist_coeff, R, T):
    """
    :param u: horizontal pixel coordinate of ground point
    :param v: vertical pixel coordinate of ground point
    :param K: intrinsic parameters
    :param dist_coeff: distortion coefficients
    :param R: rotation matrix from real camera to virtual camera
    :param T: translation
    :return: bird-view coordinate of ground point
    """

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # homogeneous coordinate
    pt = np.array([u, v]).reshape((1, 1, 2))
    undist_pt = cv2.undistortPoints(pt, K, dist_coeff, R=R, P=K)
    undist_pt = undist_pt[0][0]

    # point on bird-view
    bv_x = fy / (undist_pt[1] - cy) * T[2] if v > cy + 1 else 200
    bv_y = - (undist_pt[0] - cx) / fx * bv_x

    return bv_x, bv_y


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Bird-view projection.')
    parser.add_argument('-c', '--cam_calib', required=True,
                        help='calibrated camera parameter file path')
    args = parser.parse_args()

    # read camera parameters
    with open(args.cam_calib) as f:
        cam_param = json.load(f)

    K, dist_coeff, R, T = read_cam_param(cam_param)

    # pixel coordinate of ground point on image
    u = 980.0
    v = 900.0

    # project to bird-view
    x, y = bird_view_proj(u, v, K, dist_coeff, R, T)

    print x, y
