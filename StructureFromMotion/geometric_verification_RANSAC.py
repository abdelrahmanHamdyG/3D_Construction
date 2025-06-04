import numpy as np
import cv2
import matplotlib.pyplot as plt
import zhang_algorithm as zo  
import sift as sf             


# Ghania 11/4/2024
def draw_inlier_matches(img1, img2, kp1, kp2):
    
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # print(kp1.shape)
    

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    
    new_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    new_img[:h1, :w1] = img1_color
    new_img[:h2, w1:w1 + w2] = img2_color

    
    for pt1,pt2 in zip(kp1,kp2):
        
        pt1_s = (int(pt1[0]), int(pt1[1]))
        pt2_s = (int(pt2[0] + w1), int(pt2[1]))

        cv2.circle(new_img, pt1_s, 2, (0, 255, 0), 1)
        cv2.circle(new_img, pt2_s, 2, (0, 0, 255), 1)
        cv2.line(new_img, pt1_s, pt2_s, (255, 0, 0), 1)

    plt.figure(figsize=(15, 10))
    plt.imshow(new_img)
    plt.axis('off')
    plt.title(f"Inlier matches after RANSAC")
    plt.show()
    

# we used chatgpt in some parts of the code specially in the plotting parts 
def draw_epipolar_lines(img, lines, pts):

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 255),
        (0, 0, 0), (255, 255, 0), (0, 255, 255), (173, 216, 230),
        (255, 192, 203), (165, 42, 42)
    ]

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width = img.shape[:2]
    ccc=0
    for r_line, pt in zip(lines, pts):
        ccc+=1
        a, b, c = r_line
        if b != 0:
            x0 = 0
            y0 = int(-c / b)
            x1 = width
            y1 = int(-(a * width + c) / b)
        else:
            x0 = int(-c / a)
            x1 = x0
            y0 = 0
            y1 = height

        img = cv2.line(img, (x0, y0), (x1, y1), colors[ccc%10], 1)
        img = cv2.circle(img, (int(pt[0]), int(pt[1])), 5, colors[ccc%10], -1)
    return img



# according to this https://faculty.cc.gatech.edu/~hays/compvision2021/proj3/proj3.pdf
#  it tells that normalizing the points and the matrix  makes the accuracy better
def compute_E(pts1, pts2):

    pts1_normalized, T1 = normalize_points(pts1)
    pts2_normalized, T2 = normalize_points(pts2)

    
    A = np.zeros((len(pts1_normalized), 9))
    
    for i in range(len(pts1_normalized)):
        x1, y1 = pts1_normalized[i][0], pts1_normalized[i][1]
        x2, y2 = pts2_normalized[i][0], pts2_normalized[i][1]
        A[i] = [x2 *x1, x2*y1, x2,
                y2 *x1, y2 * y1,y2,
                x1,  y1,1]

    # Computing E using SVD Same idea like in zhang (DLT)
    U, S, VT = np.linalg.svd(A)
    E = VT[-1].reshape(3, 3)

    U, S, VT = np.linalg.svd(E)
    # enforcing rank 2 
    S = [1, 1, 0]
    E_rank2 = U @ np.diag(S) @ VT

    E = T2.T @ E_rank2 @ T1

    return E


def get_lines_equations(pts, F):
    
    epip_lines = np.dot(F ,   pts.T).T
    
    # normalizing the lines 
    # we found it doesn't matter when we remove it but is present in some online implementations 
    mags = np.sqrt(epip_lines[:, 0]**2 + epip_lines[:, 1]**2).reshape(-1,1)
    epip_lines /= mags
    return epip_lines

# from this tutorial https://cs.brown.edu/courses/csci1430/2021_Spring/proj5_cameras/
def normalize_points(pts):
    
    means = np.mean(pts[:, :2], axis=0)
    std_dev = np.std(pts[:, :2], axis=0)
    std_dev[std_dev == 0] = 0.0000001

    scale = np.sqrt(2) / std_dev
    T = np.array([
        [scale[0], 0,-scale[0] * means[0]],
        [0, scale[1], -scale[1] * means[1]],
        [0, 0,1]
    ])

    pts_normalized = (T @ pts.T).T
    return pts_normalized, T



def filter_outliers():
    pass








def Ransac_E(pts1, pts2, m1,m2,verified_1,verified_2,thresold=0.001):
    
    n = len(pts1)
    verified_1=np.array(verified_1)
    verified_2=np.array(verified_2)
    best_inliers_idx = None
    best_count = 0
    for attempt in range(2000):
        if n<9:
            return None,None,None,None,None,None
        indices = np.random.choice(n, 8, replace=False)
        chosen_1 = pts1[indices]
        chosen_2 = pts2[indices]

        
        E = compute_E(chosen_1, chosen_2)

        #  lines in image 2 for pts1
        l2 = (E @ pts1.T).T 
        # lines in image 1 for pts2
        l1 = (E.T @ pts2.T).T 

        l1_norm = l1 / np.linalg.norm(l1[:, :2], axis=1).reshape(-1,1)
        l2_norm = l2 / np.linalg.norm(l2[:, :2], axis=1).reshape(-1,1)

        errors_arr = np.abs(np.sum(pts1 * l1_norm, axis=1))  + np.abs(np.sum(pts2 * l2_norm, axis=1))  

        # Determine inliers based on the threshold
        inliers = errors_arr < thresold
        current_count = np.sum(inliers)

        # see if it is better model 
        if current_count > best_count:
            best_count = current_count
            best_inliers_idx = inliers

    # Recompute E with all inliers
    
    if best_count<=8:
        print("yes in ransac they are less than 3 ")
        return None,None,None,None,None,None
    # print("I am here")
    inlier_pts1 = pts1[best_inliers_idx]
    inlier_pts2 = pts2[best_inliers_idx]
    
    

    final_E = compute_E(inlier_pts1, inlier_pts2)
    return final_E, best_inliers_idx,m1[best_inliers_idx],m2[best_inliers_idx],verified_1[best_inliers_idx],verified_2[best_inliers_idx]
    



def getF(K,E):
    K_inv=np.linalg.inv(K)
    return K_inv.T @ E @ K_inv
import os
def getBestEssentialMatrixBetween2Images(img1_path,folder1,img2_path,folder2,K):
    
   

    if os.path.exists(f"matches_saved/{folder1}/{img1_path}_{img2_path}.pkl"):
        print("it exists")
        return sf.load_keypoints(f"matches_saved/{folder1}/{img1_path}_{img2_path}.pkl")



    
    matches, kp1, kp2 = sf.get_matches(img1_path,folder1, img2_path,folder2)
    print(f"Number of matches obtained from SIFT: {len(matches)}")
    if len(matches)<8:
        return (None,None,None,None,None,None,None,None)

    # Extract matched keypoints
    matched_pts1 = []
    matched_pts2 = []
    matched_pts1_homo=[]
    matched_pts2_homo=[]
    indices_of_verified_points_1=[]
    indices_of_verified_points_2=[]
    for m in matches:
        
        x1_sift,y1_sift = kp1[m.queryIdx][1], kp1[m.queryIdx][0]
        x2_sift,y2_sift = kp2[m.trainIdx][1], kp2[m.trainIdx][0]
        indices_of_verified_points_1.append(m.queryIdx)
        indices_of_verified_points_2.append(m.trainIdx)

        matched_pts1.append([x1_sift, y1_sift])
        matched_pts2.append([x2_sift, y2_sift])
        matched_pts1_homo.append([x1_sift, y1_sift,1])
        matched_pts2_homo.append([x2_sift, y2_sift,1])

    matched_pts1 = np.array(matched_pts1)
    matched_pts2 = np.array(matched_pts2)

    
    matched_pts1_homo=np.array(matched_pts1_homo)
    matched_pts2_homo=np.array(matched_pts2_homo)
    
    # Compute projected  coordinates so we don't care about the intrinsics now
    K_inv = np.linalg.inv(K)
    matched_pts1_projected = (K_inv @ (matched_pts1_homo.T)).T
    matched_pts2_projected = (K_inv @ (matched_pts2_homo.T)).T

    # Ransac for computing E 
    E, best_inliers,inlier_pts1_homo,inlier_pts2_homo,verified_1,verified_2 = Ransac_E(matched_pts1_projected, matched_pts2_projected,matched_pts1_homo,matched_pts2_homo,indices_of_verified_points_1,indices_of_verified_points_2)
    if E is None:
        return (None, None, None, None,None,None,None,None)
    sf.save_keypoints((E, K, inlier_pts1_homo.T, inlier_pts2_homo.T,kp1,kp2,verified_1,verified_2),f"matches_saved/{folder1}/{img1_path}_{img2_path}.pkl")
    # return E 3*N 3*N


    

    # this should be undistorted but doesn't matter a lot 
    img1 = cv2.imread(f"{folder1}/{img1_path}", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f"{folder2}/{img2_path}", cv2.IMREAD_GRAYSCALE)

    inlier_pts1 = matched_pts1[best_inliers]
    inlier_pts2 = matched_pts2[best_inliers]
    draw_inlier_matches(img1,img2,inlier_pts1[:150],inlier_pts2[:150])
    return (E, K, inlier_pts1_homo.T, inlier_pts2_homo.T,kp1,kp2,verified_1,verified_2)



if __name__ == "__main__":
    exit()
    # K, distortion = zo.get_intrinsic_params("epipolar_images/*.jpg")
    K = np.array([[1972.47338e+00, 0.00000000e+00, 722.00547e+00],

              [0.00000000e+00, 1978.22943e+00, 955.12887e+00],

              [0.00000000e+00, 0.00000000e+00, 1.0000000e+00]])
    distortion1 = np.array([2.443010e-01, -3.067277e+00, -8.641130e-04, 3.544338e-03, 7.548995e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00 ])

    


    # print(f"K:\n{K}")
    

    
    img1 = cv2.imread("nachos_resized/177.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("nachos_resized/184.jpg", cv2.IMREAD_GRAYSCALE)

    
    matches, kp1, kp2 = sf.get_matches("177.jpg","nachos_resized", "184.jpg","nachos_resized")
    # print(f"Number of matches obtained from SIFT: {len(matches)}")

    # Extract matched keypoints
    matched_pts1 = []
    matched_pts2 = []
    matched_pts1_homo=[]
    matched_pts2_homo=[]
    indices_of_verified_points_1=[]
    indices_of_verified_points_2=[]
    
    for m in matches:
        indices_of_verified_points_1.append(m.queryIdx)
        indices_of_verified_points_2.append(m.trainIdx)

        x1_sift,y1_sift = kp1[m.queryIdx][1], kp1[m.queryIdx][0]
        x2_sift,y2_sift = kp2[m.trainIdx][1], kp2[m.trainIdx][0]
        matched_pts1.append([x1_sift, y1_sift])
        matched_pts2.append([x2_sift, y2_sift])
        matched_pts1_homo.append([x1_sift, y1_sift,1])
        matched_pts2_homo.append([x2_sift, y2_sift,1])

    matched_pts1 = np.array(matched_pts1)
    matched_pts2 = np.array(matched_pts2)

    
    matched_pts1_homo=np.array(matched_pts1_homo)
    matched_pts2_homo=np.array(matched_pts2_homo)
    
    # Compute projected  coordinates so we don't care about the intrinsics now
    K_inv = np.linalg.inv(K)
    matched_pts1_projected = (K_inv @ (matched_pts1_homo.T)).T
    matched_pts2_projected = (K_inv @ (matched_pts2_homo.T)).T

    # Ransac for computing E 
    E, best_inliers,inlier_pts1_homo,inlier_pts2_homo,verified_1,verified_2 = Ransac_E(matched_pts1_projected, matched_pts2_projected,matched_pts1_homo,matched_pts2_homo,indices_of_verified_points_1,indices_of_verified_points_2)
    # print(f"E :{E}")
    # exit()

    # print(kp1[verified_1[0]][1],kp1[verified_1[0]][0])
    # print(inlier_pts1_homo)
    # # Extract inlier points
    
    
    F =getF(K,E) 
    print(f"F is {F}")

    
    lines1 = get_lines_equations(inlier_pts2_homo, F.T)  
    lines2 = get_lines_equations(inlier_pts1_homo, F)    
    # print(lines1.shape)
    #exit()
    print(lines1.shape)
    inlier_pts1 = matched_pts1[best_inliers]
    inlier_pts2 = matched_pts2[best_inliers]
    img1_with_lines = draw_epipolar_lines(img1, lines1, inlier_pts1)
    img2_with_lines = draw_epipolar_lines(img2, lines2, inlier_pts2)
    # print("inlier_pts1_homo.shape: ")
    # print(inlier_pts1_homo.shape)
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img1_with_lines, cv2.COLOR_BGR2RGB))
    plt.title(' Image 1')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img2_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')
    plt.axis('off')
    plt.show()
    
    draw_inlier_matches(img1,img2,inlier_pts1[:150],inlier_pts2[:150])

def match_2_images(img1_path,img2_path):
        
    # K, distortion = zo.get_intrinsic_params("epipolar_images/*.jpg")
    K = np.array([[1972.47338e+00, 0.00000000e+00, 722.00547e+00],

              [0.00000000e+00, 1978.22943e+00, 955.12887e+00],

              [0.00000000e+00, 0.00000000e+00, 1.0000000e+00]])
    distortion1 = np.array([2.443010e-01, -3.067277e+00, -8.641130e-04, 3.544338e-03, 7.548995e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00 ])

    


    print(f"K:\n{K}")
    

    
    img1 = cv2.imread(f"apples_segmented/f{img1_path}", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f"apples_segmented/f{img2_path}", cv2.IMREAD_GRAYSCALE)

    
    matches, kp1, kp2 = sf.get_matches(img1_path,"apples_segmented", img2_path,"apples_segmented")
    print(f"Number of matches obtained from SIFT: {len(matches)}")

    # Extract matched keypoints
    matched_pts1 = []
    matched_pts2 = []
    matched_pts1_homo=[]
    matched_pts2_homo=[]
    
    for m in matches:
        
        x1_sift,y1_sift = kp1[m.queryIdx][1], kp1[m.queryIdx][0]
        x2_sift,y2_sift = kp2[m.trainIdx][1], kp2[m.trainIdx][0]
        matched_pts1.append([x1_sift, y1_sift])
        matched_pts2.append([x2_sift, y2_sift])
        matched_pts1_homo.append([x1_sift, y1_sift,1])
        matched_pts2_homo.append([x2_sift, y2_sift,1])

    matched_pts1 = np.array(matched_pts1)
    matched_pts2 = np.array(matched_pts2)

    
    matched_pts1_homo=np.array(matched_pts1_homo)
    matched_pts2_homo=np.array(matched_pts2_homo)
    
    # Compute projected  coordinates so we don't care about the intrinsics now
    K_inv = np.linalg.inv(K)
    matched_pts1_projected = (K_inv @ (matched_pts1_homo.T)).T
    matched_pts2_projected = (K_inv @ (matched_pts2_homo.T)).T

    # Ransac for computing E 
    E, best_inliers,inlier_pts1_homo,inlier_pts2_homo = Ransac_E(matched_pts1_projected, matched_pts2_projected,matched_pts1_homo,matched_pts2_homo)
    print(f"E :{E}")
    # exit()

    # Extract inlier points
    
    
    F =getF(K,E) 
    print(f"F is {F}")

    
    lines1 = get_lines_equations(inlier_pts2_homo, F.T)  
    lines2 = get_lines_equations(inlier_pts1_homo, F)    
    # print(lines1.shape)
    #exit()
    print(lines1.shape)
    inlier_pts1 = matched_pts1[best_inliers]
    inlier_pts2 = matched_pts2[best_inliers]
    
    
    return inlier_pts1,inlier_pts2


