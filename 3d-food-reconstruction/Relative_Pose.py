#zeyad 11/5/2024


import numpy as np
import geometric_verification_RANSAC
import cv2 



# Function to generate 3D points for the checkerboard
def getRelativePose(E,  MatchedPoints1, MatchedPoints2):
    K=np.array([[1972.47338e+00, 0.00000000e+00, 722.00547e+00],

              [0.00000000e+00, 1978.22943e+00, 955.12887e+00],

              [0.00000000e+00, 0.00000000e+00, 1.0000000e+00]])
    
    # applying SVD on E to compute R and t
    U, D, VT = np.linalg.svd(E)
    W = np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]])
        
    R1_i = U @ W @ VT
    R2_i = U @ W.T @ VT

    
    # we get two potential values for R 
    R1 = R1_i * sign(np.linalg.det(R1_i)) * sign(np.linalg.det(K))
    R2 = R2_i * sign(np.linalg.det(R2_i)) * sign(np.linalg.det(K))

    # we get two potential values for t which are the positive and negative values of the last column of U, corresponding to the smallest singular value.
    v = np.array([
                        [0],
                        [0],
                        [1]])
    t1 = U @ v
    t2 = -U @ v    

    # Constructing projection matrices with the potential R and t pairs
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R1, t1))
    P3 = K @ np.hstack((R1, t2))
    P4 = K @ np.hstack((R2, t1))
    P5 = K @ np.hstack((R2, t2))



    MatchedPoints1_non_homogeneous = MatchedPoints1[:2, :] / MatchedPoints1[2, :]
    MatchedPoints2_non_homogeneous = MatchedPoints2[:2, :] / MatchedPoints2[2, :]

    # Triangulating points
    TriPoints1 = cv2.triangulatePoints(P1, P2, MatchedPoints1_non_homogeneous, MatchedPoints2_non_homogeneous)
    TriPoints2 = cv2.triangulatePoints(P1, P3, MatchedPoints1_non_homogeneous, MatchedPoints2_non_homogeneous)
    TriPoints3 = cv2.triangulatePoints(P1, P4, MatchedPoints1_non_homogeneous, MatchedPoints2_non_homogeneous)
    TriPoints4 = cv2.triangulatePoints(P1, P5, MatchedPoints1_non_homogeneous, MatchedPoints2_non_homogeneous)
    #[,3,6,8,7,]
    #[3/7]
    # Homogenizing 3D points 
    TriPoints1_homo = TriPoints1 / TriPoints1[3, :]
    TriPoints2_homo = TriPoints2 / TriPoints2[3, :]
    TriPoints3_homo = TriPoints3 / TriPoints3[3, :]
    TriPoints4_homo = TriPoints4 / TriPoints4[3, :]
    
    dep11 = P1[2, :] @ TriPoints1_homo
    dep12 = P2[2, :] @ TriPoints1_homo

    dep21 = P1[2, :] @ TriPoints2_homo
    dep23 = P3[2, :] @ TriPoints2_homo

    dep31 = P1[2, :] @ TriPoints3_homo
    dep34 = P4[2, :] @ TriPoints3_homo

    dep41 = P1[2, :] @ TriPoints4_homo
    dep45 = P5[2, :] @ TriPoints4_homo

    eligibles1 = (dep11 > 0).astype(int) + (dep12 > 0).astype(int)
    eligibles1 = np.sum(eligibles1 == 2)

    eligibles2 = (dep21 > 0).astype(int) + (dep23 > 0).astype(int)
    eligibles2 = np.sum(eligibles2 == 2)
    
    eligibles3 = (dep31 > 0).astype(int) + (dep34 > 0).astype(int)
    eligibles3 = np.sum(eligibles3 == 2)
    
    eligibles4 = (dep41 > 0).astype(int) + (dep45 > 0).astype(int)
    eligibles4 = np.sum(eligibles4 == 2)

    eligibles = [eligibles1, eligibles2, eligibles3, eligibles4]
    bestPose = np.argmax(eligibles)

    solutions = {
    0: (R1, t1, TriPoints1),
    1: (R1, t2, TriPoints2),
    2: (R2, t1, TriPoints3),
    3: (R2, t2, TriPoints4)
    }

    R, t, TriPoints  = solutions[bestPose]

    print("Rotation: ", R)
    print("Translation", t)

    return R, t, TriPoints



def sign(determinant):
    return 1 if determinant >= 0 else -1

def main():
    img1_path = "epipolar_images/21.jpg"
    img2_path = "epipolar_images/28.jpg"
    
    # Get the Essential matrix, camera intrinsics, and matched points
    E, K, MatchedPoints1, MatchedPoints2 = geometric_verification_RANSAC.getBestEssentialMatrixBetween2Images(img1_path, img2_path)
    
    # Compute relative pose
    R, t, TriPoints = getRelativePose(E, K, MatchedPoints1, MatchedPoints2)
    
   

if __name__== "__main__":
    main()