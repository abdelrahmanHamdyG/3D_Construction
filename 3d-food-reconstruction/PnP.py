#zeyad 11/12/2024

import cv2
import numpy as np
import Relative_Pose


def solve_PnP(_3d_points, _2d_points, K, distortion_params):

    retVal, rvec, tvec, inliers = cv2.solvePnPRansac(
    objectPoints=_3d_points, 
    imagePoints=_2d_points, 
    cameraMatrix=K, 
    distCoeffs=distortion_params, 
    reprojectionError=8.0,  #threshold
    confidence=0.99,
    flags=cv2.SOLVEPNP_ITERATIVE
    )
    R, _ = cv2.Rodrigues(rvec) #convertin rotation vector into rotation matrix
    
    return R, tvec
    



def main():

    #getting the triangulated 3d points with the computed reltaive pose 
    _, __, _3d_points = Relative_Pose.getRelativePose()

    # _2d_points = to be passed

    K = np.array([[1972.47338e+00, 0.00000000e+00, 722.00547e+00],

              [0.00000000e+00, 1978.22943e+00, 955.12887e+00],

              [0.00000000e+00, 0.00000000e+00, 1.0000000e+00]])
    distortion1 = np.array([2.443010e-01, -3.067277e+00, -8.641130e-04, 3.544338e-03, 7.548995e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00 ])

