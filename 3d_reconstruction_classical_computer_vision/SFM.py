import numpy as np
import cv2
import os
import open3d as o3d
import pickle
from datetime import datetime

import sift as sf 
import geometric_verification_RANSAC as gm  
import Relative_Pose as rp  
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from logger import Logger


logger=Logger().get_logger()

# intrinsic parameters

distortion1 = np.array([2.443010e-01, -3.067277e+00, -8.641130e-04, 3.544338e-03, 7.548995e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00 ])





# we got some help from chatgpt specially in plotting, getting the color of the constructed 3d pixels, sparse matrix for optimization and debugging errors




def generate_tracks(matches, keypoint_id_map):
    
    # disjoint set data structure 
    parent = {}
    rank = {}

    # like serial number for 3d point
    unique_id = 0

    
    def find_parent(idx):
        if parent[idx] != idx:
            parent[idx] = find_parent(parent[idx])
        return parent[idx]

    def union(idx1, idx2):
        root1 = find_parent(idx1)
        root2 = find_parent(idx2)
        if root1 == root2:
            return
        if rank[root1] < rank[root2]:
            parent[root1] = root2
        else:
            parent[root2] = root1
            if rank[root1] == rank[root2]:
                rank[root1] += 1

    for (i, j), matches_indices in matches.items():
        verified_indices_1 = matches_indices[0]
        verified_indices_2 = matches_indices[1]

        for first_index, second_index in zip(verified_indices_1, verified_indices_2):
            first_feature = (i, first_index)
            second_feature = (j, second_index)

            if first_feature not in keypoint_id_map:
                keypoint_id_map[first_feature] = unique_id
                parent[unique_id] = unique_id
                rank[unique_id] = 0
                unique_id += 1

            if second_feature not in keypoint_id_map:
                keypoint_id_map[second_feature] = unique_id
                parent[unique_id] = unique_id
                rank[unique_id] = 0
                unique_id += 1

            idx1 = keypoint_id_map[first_feature]
            idx2 = keypoint_id_map[second_feature]

            union(idx1, idx2)

    for feature in keypoint_id_map:
        keypoint_id_map[feature] = find_parent(keypoint_id_map[feature])

    tracks = {}
    for feature, track_id in keypoint_id_map.items():
        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append(feature)

    return tracks, keypoint_id_map

def candidate_score(kps, image_width=340, image_height=256):
    total_score = 0

       
    keypoints_normalized = [(x / image_width, y / image_height) for x, y in kps]

    for l in range(1, 4):
        K_l = 2 ** l
        grid = np.zeros((K_l, K_l), dtype=int)

        for x_norm, y_norm in keypoints_normalized:
            bin_x = int(np.floor(x_norm * K_l))
            bin_y = int(np.floor(y_norm * K_l))
            bin_x = min(bin_x, K_l - 1)
            bin_y = min(bin_y, K_l - 1)

            grid[bin_x, bin_y] = 1
        C_l = np.sum(grid)
        w_l = K_l**2
        S_l = w_l * C_l

        total_score += S_l

    return total_score


# def calculte_triangulation_angle(_3d_point, RT1, RT2):
#     # Calculate the angle between the two rays
#     # from the camera centers to the 3D point
#     C1 = -RT1[:,:3].T @ RT1[:,3]
#     C2 = -RT2[:,:3].T @ RT2[:,3]

#     ray1 = _3d_point - C1
#     ray2 = _3d_point - C2

#     cos_angle = np.dot(ray1, ray2)/(np.linalg.norm(ray1)* np.linalg.norm(ray2))
#     angle = np.arccos(cos_angle)

#     return angle

# def cheirality_check(RT1, RT2, point_3d):
#     # Check if the 3D point is in front of both cameras
#     C1 = -RT1[:, :3].T @ RT1[:, 3]
#     C2 = -RT2[:, :3].T @ RT2[:, 3]
#     ray1 = point_3d - C1
#     ray2 = point_3d - C2
#     if ray1[2] < 0 or ray2[2] < 0:
#         return False
#     return True


def bundle_adjustment(camera_poses, unique_id2reconstructed, tracks, keypoint_id_map, all_kps, K,visited_tracks):
    
    image_indices = list(camera_poses.keys())

    
    # if image_indices = [0, 2, 5]-> image_idx_to_cam_idx = {0:0, 2:1, 5:2}
    image_idx_to_cam_idx = {img_idx: cam_idx for cam_idx, img_idx in enumerate(image_indices)}

    
    numberOfCameras = len(image_indices)

    
    # unique_ids = list(unique_id2reconstructed.keys())
    unique_ids = [unique_id for unique_id in unique_id2reconstructed.keys() if visited_tracks.get(unique_id, False)]


    unique_id_to_point_idx = {unique_id: idx for idx, unique_id in enumerate(unique_ids)}

    # Total number of 3D points
    num_points = len(unique_ids)
    print(f"total number of points inside bundle_adjustment {num_points}")

    camera_params = np.zeros((numberOfCameras, 6))

    
    for cam_idx, img_idx in enumerate(image_indices):
        
        RT = camera_poses[img_idx]
        R = RT[:, :3]  
        t = RT[:, 3]   

        
        rotationAsVec = cv2.Rodrigues(R)[0]  

        
        camera_params[cam_idx, :3] = rotationAsVec.flatten() 
        camera_params[cam_idx, 3:] = t.flatten()             

    
    points_3d = np.zeros((num_points, 3))

    
    for idx, unique_id in enumerate(unique_ids):
        point = unique_id2reconstructed[unique_id]  
        points_3d[idx, :] = point

    
    camera_indices = []  
    point_indices = []   
    points_2d = []       

    
    for unique_id in unique_ids:
        point_idx = unique_id_to_point_idx[unique_id]       
        observations_for_point = tracks[unique_id]          

        for obs in observations_for_point:
            img_idx, kp_idx = obs                           

            
            if img_idx in image_idx_to_cam_idx:
                cam_idx = image_idx_to_cam_idx[img_idx] 

                
                kp = all_kps[img_idx][kp_idx]               
                observed_2d = np.array([kp[1], kp[0]])     

                
                camera_indices.append(cam_idx)
                point_indices.append(point_idx)
                points_2d.append(observed_2d)  

    
    camera_indices = np.array(camera_indices)  
    point_indices = np.array(point_indices)    
    points_2d = np.array(points_2d)           

    
    n_observations = points_2d.shape[0]
    

    n_params = numberOfCameras * 6 + num_points * 3

    
    A = lil_matrix((n_observations * 2, n_params), dtype=int) 

    
    i = np.arange(n_observations)

    
    for s in range(6):  
        
        A[2 * i, camera_indices * 6 + s] = 1      
        A[2 * i + 1, camera_indices * 6 +s] = 1  

    
    for s in range(3):  
        
        A[2 * i, numberOfCameras * 6 + point_indices * 3 + s] = 1      
        A[2 * i + 1, numberOfCameras * 6 + point_indices * 3 + s] = 1  

    
    def residuals(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        

        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))  
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))       

        
        residuals = np.empty(points_2d.shape[0] * 2)  

        
        for i in range(points_2d.shape[0]):
            cam_idx = camera_indices[i]      
            point_idx = point_indices[i]     
            observed_2d = points_2d[i]       

            
            rotation_vector = camera_params[cam_idx, :3]  
            translation_vector = camera_params[cam_idx, 3:]

            
            point_3d = points_3d[point_idx]  

            projected_2d, _ = cv2.projectPoints(
                point_3d.reshape(1, 3),        
                rotation_vector,               
                translation_vector,            
                K,                             
                distCoeffs=None
            )

            projected_2d = projected_2d.reshape(2)  

            residuals[2 * i] = observed_2d[0] - projected_2d[0]  #
            residuals[2 * i + 1] = observed_2d[1] - projected_2d[1]  

        return residuals 

    params = np.hstack((camera_params.ravel(), points_3d.ravel()))  # Shape: (n_params,)

    results = least_squares(
        residuals,               
        params,                  
        jac_sparsity=A,          
        verbose=2,               
        x_scale='jac',           
        ftol=1e-4,               
        method='trf',            
        max_nfev=15,             
        args=(                   
            numberOfCameras,
            num_points,
            camera_indices,
            point_indices,
            points_2d
        )
    )

    new_params = results.x  

    optimized_camera_params = new_params[:numberOfCameras * 6].reshape((numberOfCameras, 6)) 
    optimized_points_3d = new_params[numberOfCameras * 6:].reshape((num_points, 3))          

    for cam_idx, img_idx in enumerate(image_indices):
        R = cv2.Rodrigues(optimized_camera_params[cam_idx, :3])[0]  
        tr = optimized_camera_params[cam_idx, 3:].reshape(3, 1)     

        RT = np.hstack((R, tr))  

        camera_poses[img_idx] = RT

    for idx, unique_id in enumerate(unique_ids):
        point = optimized_points_3d[idx]  
        unique_id2reconstructed[unique_id] = point

    return camera_poses, unique_id2reconstructed

def write_ply(filename, points, colors=None):
    with open(filename, 'w') as f:
        num_points = points.shape[0]
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {num_points}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if colors is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(num_points):
            line = f'{points[i, 0]} {points[i, 1]} {points[i, 2]}'
            if colors is not None:
                # RGB
                line += f' {int(colors[i, 2])} {int(colors[i, 1])} {int(colors[i, 0])}'  # Convert BGR to RGB for PLY
            f.write(line + '\n')

def visualize_with_open3d(points3D, colors_bgr):
    # if points3D.shape[0] == 0:
    #     print("No points to visualize.")
    #     return

    # Convert BGR to RGB and normalize
    colors_rgb = colors_bgr[..., ::-1] / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
    o3d.visualization.draw_geometries([pcd])

def load_reconstructed_data(points_file='reconstructed_points.npy', colors_file='colors.npy'):
    points3D = np.load(points_file)
    if os.path.exists(colors_file):
        print("here")
        colors = np.load(colors_file)
        print(f"Loaded colors for {colors.shape[0]} points.")
        # return
    return points3D, colors

def calculate_reprojection_error(tracks, camera_poses, keypoint_id_map, all_kps, K, unique_id2reconstructed,visited_tracks):
    total_error = 0
    total_points = 0

    for track_id, observations in tracks.items():
        if track_id not in unique_id2reconstructed:
            continue
        if track_id not in visited_tracks:
            continue
        if visited_tracks[track_id]==False:
            continue
        point_3d = unique_id2reconstructed[track_id]

        for obs in observations:
            img_idx, kp_idx = obs
            if img_idx not in camera_poses:
                continue

            RT = camera_poses[img_idx]
            R, t = RT[:, :3], RT[:, 3]

            rotationAsVec = cv2.Rodrigues(R)[0]
            tr = t.reshape(3, 1)
            projected_2d, _ = cv2.projectPoints(
                point_3d.reshape(1, 3), rotationAsVec, tr, K, distCoeffs=None
            )
            projected_2d = projected_2d.reshape(2)

            kp = all_kps[img_idx][kp_idx]
            observed_kp = np.array([kp[1], kp[0]])  # (x, y)

            error = ((observed_kp[0] - projected_2d[0]) ** 2) + ((observed_kp[1] - projected_2d[1]) ** 2)

            total_error += error
            total_points += 1

    if total_points==0:
        print("number of points = 0")
    mean_reprojection_error = total_error / total_points 

    print(f"Total error: {total_error}")
    print(f"number of points: {total_points}")
    print(f"Mean reprojeciton Error: {mean_reprojection_error}")
    return mean_reprojection_error



# 2  for min triangulation angle and 8 for reproj threshold is from the paper
def multiViewTriangulation(observations, camera_poses, all_kps, K, min_triangulation_angle=2, reproj_threshold=8):
    
    
    registered_images_in_track = []
    for img_idx, kp_idx in observations:
        if img_idx in camera_poses:
            kp = all_kps[img_idx][kp_idx]
            x_hom = np.array([kp[1], kp[0], 1.0]) 
            RT = camera_poses[img_idx]
            registered_images_in_track.append((img_idx, kp_idx, x_hom, RT))


    #logger.info(f"registered_images_in_track size is {len(registered_images_in_track)}")
    

    projection_matrices = {}
    for img_idx, _, p, RT in registered_images_in_track:
        P = K @RT
        projection_matrices[img_idx] = P

    
    
    best_inliers = []
    best_point = None
    iter = 0
    sampled_pairs = set()
    indices = list(range(len(registered_images_in_track)))

    # RANSAC
    max_iterations = 1000

    while iter<max_iterations:
        idx_a, idx_b = np.random.choice(indices, 2, replace=False)
        key = tuple(sorted([idx_a,idx_b]))
        if key in sampled_pairs:
            iter += 1
            continue
        sampled_pairs.add(key)
        # print(iter)
        # exit()
        obs_a = registered_images_in_track[idx_a]
        obs_b = registered_images_in_track[idx_b]
        img_idx_a, kp_idx_a, x_hom_a, RT_a = obs_a
        img_idx_b, kp_idx_b, x_hom_b, RT_b = obs_b

        # Projection matrices
        P_a = projection_matrices[img_idx_a]
        P_b = projection_matrices[img_idx_b]

        # Triangulate point
        x1 = x_hom_a[:2].reshape(2, 1)
        x2 = x_hom_b[:2].reshape(2, 1)
        X_hom = cv2.triangulatePoints(P_a, P_b, x1, x2)
        if X_hom is None or X_hom.shape[1] == 0:
            iter += 1
            continue
        X = X_hom[:3] / X_hom[3]
        X = X.flatten()

        # Check triangulation angle
        C_a = -RT_a[:,:3].T@ RT_a[:, 3]
        C_b = -RT_b[:, :3].T @ RT_b[:, 3]
        ray_a = X - C_a
        ray_b = X - C_b
        norm_ray_a = np.linalg.norm(ray_a)
        norm_ray_b = np.linalg.norm(ray_b)
        if norm_ray_a == 0 or norm_ray_b == 0:
            iter += 1
            continue
        cos_angle =np.dot(ray_a,ray_b) / (norm_ray_a *norm_ray_b)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        # #logger.info(f"angle between {img_idx_a} and {img_idx_b}: {angle} degree")

        if angle<min_triangulation_angle:
            iter += 1
            continue

        #  cheirality constraint
        depths = []
        for P in [P_a, P_b]:
            depth = P[2,:3] @ X + P[2,3]
            depths.append(depth)
        if any(d <= 0 or not np.isfinite(d) for d in depths):
            iter += 1
            continue

        # counting the number of inliers inliers
        inliers = []
        for idx, (img_idx, kp_idx, x_hom, RT) in enumerate(registered_images_in_track):
            P = projection_matrices[img_idx]
            x_proj_hom = P @ np.append(X, 1)
            x_proj= x_proj_hom[:2] / x_proj_hom[2]
            x_obs = x_hom[:2]
            error = np.linalg.norm(x_obs - x_proj)
            # we only need the z-axis
            depth = P[2,:3] @ X + P[2, 3]
            if error<reproj_threshold and depth > 0:
                inliers.append(idx)

        # Update best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_point = X.copy()
            inlier_ratio = len(best_inliers) / len(registered_images_in_track)
            max_iterations = min(
                max_iterations,
                int(np.log(1 - 0.99) / np.log(1 - inlier_ratio ** 2) + 1)
            )

        iter+= 1
        if len(best_inliers) == len(registered_images_in_track):
            break

    if len(best_inliers)<3:
        return None,[]

    # refine point using all inliers using similar way to DLT
    A = []
    for idx in best_inliers:
        img_idx, kp_idx,x_hom, RT = registered_images_in_track[idx]
        P = projection_matrices[img_idx]
        x = x_hom[0]
        y = x_hom[1]
        A.append(x*P[2,:]-P[0,:])
        A.append(y*P[2,:]-P[1,:])
    A = np.vstack(A)
    Vt = np.linalg.svd(A)[2]
    X_refined = Vt[-1]
    X_refined = X_refined[:3]/X_refined[3] # non homgenous 

    
    #logger.info(f"difference between X and X_refined{X-X_refined}")
   
    inlier_observations = [(registered_images_in_track[idx][0] ,registered_images_in_track[idx][1]) for idx in best_inliers]
    return X_refined, len(inlier_observations)



def SFM():
    # K =np.array ([[1484.00168,    0.      , 929.48773],
    # [   0.      ,1629.13411 ,1365.97989],
    # [   0.      ,   0.         ,1.     ]])
    # K = np.array([[1972.47338e+00, 0.00000000e+00, 722.00547e+00],

    #           [0.00000000e+00, 1978.22943e+00, 955.12887e+00],

    #           [0.00000000e+00, 0.00000000e+00, 1.0000000e+00]])

    K = np.array([
    [477.60000000, 0.00000000e+00, 199.000000000],
    [0.00000000e+00, 477.60000000, 169.000000000],
    [0.00000000e+00, 0.00000000e+00, 1.0000000e+00]
    ])



    # distortion1=np.array([  0.55793, -3.13301,  0.00011, -0.0283,   3.69096])

    folder_path = "cupcake_final_last_one"
    print("sds")
    img_mask=None
    
    # dictionary(image idx)-> image keypoints             1->[(3,4),(5,6),(1,2)]
    all_kps = {}

    flag=0
    unique_id2_set_size={}

    # dictionary(keypoint)-> unique_id
    keypoint_id_map = {}


    # registered images till now initally empty
    registered_images = []


    # unregistered images till now initially all the images 
    unregistered_images = list(range(0,200))  

    # folder that I will read the images from

    # dictionary (image_i,image_j)->
    matches = {}

    # variable that hold the maximum number of matches between any pair to choose the initial image
    maximum_number_of_matches = 0


    # reconstructed 3d points till now intially empty
    reconstructed_points = []

    # 
    colors = []

    # unique_id-> 3d point             3->[10,3,4]
    unique_id2reconstructed = {}

    # initial pair choosen
    initial_pair = (0,2)


    # dictionary (image_index)->(the image itself)       1-> [[[2,3,4],[2,3,3].. ] ... ] (400,300,3) hight width RGB
    images = {}
    


    # new code
    for img_idx in range(0,200):
        # print("we are here ")
        img_path = os.path.join(folder_path, f"{img_idx}.jpg")
        img = cv2.imread(img_path)
        # if img is None:
        #     print(f"Image {img_path} not found.")
        #     continue
        # h, w = img.shape[:2]
        # # Compute the optimal new camera matrix
        # new_K, roi = cv2.getOptimalNewCameraMatrix(K, distortion1, (w, h), 1, (w, h))
        # # Undistort the image
        # img_undistorted = cv2.undistort(img, K, distortion1, None, new_K)
        images[img_idx] = img

    

    # reading all images and save them in images
    # for img_idx in range(1,67):  
    #     img_path = os.path.join(folder_path, f"{img_idx}.jpg")
    #     img = cv2.imread(img_path)
    #     img_undistorted = cv2.undistort(img, K, distortion1)
    #     images[img_idx] = img


    print("we are here")
    

    # iterating over all the pairs
    for i in range(0,200):

        # reading the keypoints of each image
        keypoints_path = f"key_points_saved/{folder_path}/{i}.jpg.pkl"
        if not os.path.exists(keypoints_path):
            continue
        # assign the keypoint to the dict all_kps
        all_kps[i] = sf.load_keypoints(keypoints_path)


        for j in range(i + 1, min(i+25,200)):
            
            print(f"matches ({i},{j})")

            print("")
            E, _, inliers1, inliers2, kp1, kp2, verified_indices_1, verified_indices_2 = gm.getBestEssentialMatrixBetween2Images(
                f"{i}.jpg", folder_path, f"{j}.jpg", folder_path,K)
            
            # E: Essential matrix 
            # K: Intrinsics useless here 
            # inliers1 3*N:  homogenous keypoints from the first point that match to the second point
            # inliers2 3*N: homogenous keypoints from the second point that match to the first point
            # kp1 X*2: all key points of the first image 
            # kp2 Y*2: all keypoints of the second image
            # verified_indices_1: indices of keypoint in the first image  that matches in the second image kp1[verified_indices_1[0]]=inliers1[0]
            # verified_indices_2:indices of keypoint in the second image  that matches in the first image kp2[verified_indices_2[0]]=inliers2[0]


            if inliers1 is not None:

                # this is what the dictionary matches save
                matches[(i, j)] = (verified_indices_1, verified_indices_2, len(verified_indices_1), E, inliers1, inliers2)

                # maximizing on the number of matches to extract the best initial pair 
                if len(verified_indices_1) > maximum_number_of_matches and j-i>1:
                    maximum_number_of_matches = len(verified_indices_1)
                    initial_pair = (i, j)
                    # angle 
            else:
                pass


    print(f"Initial pair ({initial_pair[0]}, {initial_pair[1]})")


    # tracks: dictionary(j)->[(img_idx1,kp1), (img_idx2,kp2), ... ]          (3)->[(1,5),(9,300),(10,67)] 
    # the corrosponding 2d points of the 3d point i that has unique_id = j are the one that has index kp1 in image imag_idx1 and the one that has index kp2 in img_idx2

    # keypoint_id_map(img_idx1,kp1)->(j)     the keypoint with index kp1 in img_idx1 corrospond to the 3d point with unique id j
    tracks, keypoint_id_map = generate_tracks(matches, keypoint_id_map)


    # visited_tracks (j)->(true or false),  the 3d point with unique id (j)  has been reconstructed before or not 
    visited_tracks = {} 

    # registering the intial pair 
    registered_images = [initial_pair[0], initial_pair[1]]
    unregistered_images.remove(initial_pair[0])
    unregistered_images.remove(initial_pair[1])



    # Get the initial relative pose between the initial pair
    # inliers1 and inliers is 3*N   verified_indices_1 are the indices of the matched keypoints as told before

    (verified_indices_1, verified_indices_2, numOfMatches, E, inliers1, inliers2) = matches[(initial_pair[0], initial_pair[1])]


    # we are getting the relative pose between the initial pair 
    R, tr, _ = rp.getRelativePose(E, inliers1, inliers2)


    
    
    
    
    # the extrinsic parameter of the intial pair
    # we assume always that the first image is like the origin with translation and rotation
    # RT is 3*4

    

    
    RT0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    RT1 = np.hstack((R,tr.reshape(3, 1)))



    # dictionary(img_idx)->RT matrix             
    camera_poses = {}
     

    

    # assigning the values of RT for the initial pair 
    camera_poses[initial_pair[0]] = RT0
    camera_poses[initial_pair[1]] = RT1

    # calculating the projection matrix of the initial pair P0(3*4) = K(3*3) @ RT0 (3*4)  
    P0 = K @ RT0
    P1 = K @ RT1


    # converting the matched keypoints from 3*N to 2*N  (not homogenous now )
    inliers1_2d = inliers1[:2, :] / inliers1[2, :]
    inliers2_2d = inliers2[:2, :] / inliers2[2, :]


    # triangulating the 3d points of the intial pair the output is homo in shape 4*N
    points4D_hom = cv2.triangulatePoints(P0, P1, inliers1_2d, inliers2_2d)
    
    # points 3D (3*N) all reconstructed points from the initial pair 
    points3D = points4D_hom[:3, :] / points4D_hom[3, :]

    

    # we are enumerating over the already reconstruected points from the initial pair N*3 
    for idx, point in enumerate(points3D.T):
        #idx: counter from 0 to N-1
        #point: one 3d point reconstructed from initial pair with shape (3,) 
        
        # reconstructed_points.append(point)
        # getting the unique id of the 3d point to assign it as visited 
        unique_id = keypoint_id_map[(initial_pair[0], verified_indices_1[idx])]
        visited_tracks[unique_id] = True
        unique_id2reconstructed[unique_id] = point
        
        
    
    # while there are some image that are not registered yet 
    while unregistered_images:
        flag+=1

        #dictionary  next_view_candidates[img_idx] = (points_2d, points_3d)  best explained below
        next_view_candidates = {}
    


        # iterating over all unregistered images 
        for img_idx in unregistered_images:
            # 
            
            img_kp=all_kps[img_idx]
            # points_3d will save the  3d points that already reconstructed and have corrosponding keypoints for the current image (img_idx)
            points_3d = []

            # points_32 will save the  2d points keypoints  that corrosponds to the  already reconstructed 3d points
            points_2d = []

            # iterating over all the 3d points and their observation 
            for unique_id, observations in tracks.items():
                
                # if the 3d point with id (unique_id) has been visited before 
                if unique_id in visited_tracks:
                    for obs in observations:
                        # if the current image has a keypoints that match the current 3d point 
                        if obs[0] == img_idx:
                            kp = img_kp[obs[1]]
                            points_2d.append((kp[1], kp[0])) 
                            points_3d.append(unique_id2reconstructed[unique_id])
                            break
            # if the number of corrospondence is greater than 10 it is a candidate
            if len(points_2d) >= 10:
                next_view_candidates[img_idx] = (points_2d, points_3d)
        if not next_view_candidates:
            break
        


        print(next_view_candidates)
        # scores(img_idx)-> score (score is based on the method of grid SFM revisited paper)
        scores = {}
        

        # we calculate the score of each candidate 
        for img_idx, (kps_2d, kps_3d) in next_view_candidates.items():
            scores[img_idx] = candidate_score(kps_2d)


        
        
        

        # we get the best candindate next_view
        next_view = max(scores, key=scores.get)
        # print(f"Registering image {next_view}")
        


        # registering the next_view 
        registered_images.append(next_view)
        unregistered_images.remove(next_view)


        
        points_2d, points_3d = next_view_candidates[next_view]
        points_2d = np.array(points_2d, dtype=np.float32)
        points_3d = np.array(points_3d, dtype=np.float32)

        # Getting The Absolute Pose of the new image # may be problem here 

        retval, rotationAsVec, tr, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # check if we got any problem caclulate the absolute pose
        if not retval:
            print(f"Pose estimation failed for image {next_view}")
            # exit()
            continue


        # getting the RT matrix of the new image and assign it to the camera_poses
        R = cv2.Rodrigues(rotationAsVec)[0]
        t = tr.reshape(3, 1)
        RT = np.hstack((R, t))
        camera_poses[next_view] = RT
        

        ### new code 

        
        for unique_id,observations in tracks.items():

            amIinthetrack= next_view in [obs[0] for obs in observations]
            registered_observations=[]
            for obs in observations:
                if obs[0] in registered_images:
                    registered_observations.append(obs)
            
            if amIinthetrack and len(registered_observations)>=2:
                X_refined,size_of_observations=multiViewTriangulation(observations,camera_poses,all_kps,K)
                if X_refined is not  None:
                    

                    previous_maximum_set_size=0
                    if unique_id in unique_id2_set_size:
                        previous_maximum_set_size=unique_id2_set_size[unique_id]

                    if size_of_observations>previous_maximum_set_size:
                        unique_id2reconstructed[unique_id]=X_refined
                        visited_tracks[unique_id]=True
                        unique_id2_set_size[unique_id]=size_of_observations
        

        # if flag%5==0:
        #     #logger.info("we started bundle adjustment for flag")
        #     mean_error_before = calculate_reprojection_error(
        #     tracks, camera_poses, keypoint_id_map, all_kps, K, unique_id2reconstructed
        #     )
        #     #logger.info(f"Error before bundle adjustment: {mean_error_before}")

            
        #     print("Starting bundle adjustment")
        #     camera_poses, unique_id2reconstructed = bundle_adjustment(
        #         camera_poses, unique_id2reconstructed, tracks, keypoint_id_map, all_kps, K
        #     )

        #     # Update reconstructed points after bundle adjustment
        #     reconstructed_points = []
        #     colors = []

        #     # iterative over all the 3D Points
        #     for unique_id in unique_id2reconstructed:
        #         point = unique_id2reconstructed[unique_id]
        #         reconstructed_points.append(point)
        #         idx = list(unique_id2reconstructed.keys()).index(unique_id)
        #         # colors.append(colors_bgr[idx])
        #     reconstructed_points = np.array(reconstructed_points)
       


    unique_ids=[]
    reconstructed_points=[]

    colors=[]
    for unique_id,X_refined in unique_id2reconstructed.items():
        reconstructed_points.append(X_refined)
        unique_ids.append(unique_id)
        if unique_id not in tracks:
            print("not good at all")
            continue 

        observations = tracks[unique_id]
        color_found = False

        for obs in observations:
            img_idx, kp_idx = obs

            
            if img_idx not in images:
                continue

            
            kp = all_kps[img_idx][kp_idx]
            x = int(round(kp[1]))  
            y = int(round(kp[0]))  

            
            img = images[img_idx]
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                color = img[y, x, :]
                colors.append(color)
                color_found = True
                break  

        if not color_found:
            colors.append([0, 0, 0])  

                


    print(f"Total registered images: {len(registered_images)}")
    print(f"Total constructed points till now: {len(reconstructed_points)}")

    
    

    reconstructed_points = np.array(reconstructed_points)
    print(f"number of points beforoe going to bundle adjustment is {len(reconstructed_points)}")
    colors_bgr = np.array(colors)

    # Calculate mean reprojection error before bundle adjustment
    print("Calculating mean reprojection error before bundle adjustment:")
    mean_error_before = calculate_reprojection_error(
        tracks, camera_poses, keypoint_id_map, all_kps, K, unique_id2reconstructed,visited_tracks
    )
    print(f"Error before bundle adjustment: {mean_error_before}")

    
    print("Starting bundle adjustment")
    camera_poses, unique_id2reconstructed = bundle_adjustment(
        camera_poses, unique_id2reconstructed, tracks, keypoint_id_map, all_kps, K,visited_tracks
    )

    # Update reconstructed points after bundle adjustment
    reconstructed_points = []
    colors = []

    # iterative over all the 3D Points
    for unique_id in unique_id2reconstructed:
        point = unique_id2reconstructed[unique_id]
        reconstructed_points.append(point)
        idx = list(unique_id2reconstructed.keys()).index(unique_id)
        # colors.append(colors_bgr[idx])
    reconstructed_points = np.array(reconstructed_points)
    # colors_bgr = np.array(colors)

    # 
    # calculating the reprojection error
    mean_error_after = calculate_reprojection_error(
        tracks, camera_poses, keypoint_id_map, all_kps, K, unique_id2reconstructed,visited_tracks
    )
    print(f"Mean Reprojection Error after bundle adjustment: {mean_error_after}")

    print("SFM done")
    print(f"Number of 3D points reconstructed: {len(reconstructed_points)}")

    
    np.save("reconstructed_points.npy", reconstructed_points)
    with open("camera_poses.pkl", "wb") as f:
        pickle.dump(camera_poses, f)
    np.save("colors.npy", colors_bgr)
    write_ply('reconstructed_points_colored.ply', reconstructed_points, colors_bgr)
    visualize_with_open3d(reconstructed_points, colors_bgr)
    points3D=reconstructed_points
    z_negative = np.sum(points3D[:, 2] < 0)
    
    print(f"number of points with z <0: {z_negative}")
    # exit()

    print(f"Max Z: {np.max(points3D[:, 2])}")
    print(f"Min Z: {np.min(points3D[:, 2])}")
    print(f"Max Y: {np.max(points3D[:, 1])}")
    print(f"Min Y: {np.min(points3D[:, 1])}")
    print(f"Max X: {np.max(points3D[:, 0])}")
    print(f"Min X: {np.min(points3D[:, 0])}")

    std_dev = np.std(points3D, axis=0)
    mean = np.mean(points3D, axis=0)
    print(f"Standard deviation: {std_dev}")
    print(f"Mean: {mean}")

    # Filtering points within a threshold of std and see what will happen (bad way)
    threshold = 0.9
    valid_mask=np.all(np.abs(points3D-mean)<=threshold*std_dev,axis=1)
    points3D =points3D[valid_mask]
    colors_bgr2= colors_bgr[valid_mask]
    
    print(f"Number of valid points after filtering: {len(points3D)}")
    print(f"Max Z: {np.max(points3D[:, 2])}")
    print(f"Min Z: {np.min(points3D[:, 2])}")
    print(f"Max Y: {np.max(points3D[:, 1])}")
    print(f"Min Y: {np.min(points3D[:, 1])}")
    print(f"Max X: {np.max(points3D[:, 0])}")
    print(f"Min X: {np.min(points3D[:, 0])}")
    print(f"std: {std_dev}")
    print(f"mean:{mean}")

    
    
    print(f"Max Z: {np.max(points3D[:, 2])}")
    print(f"Min Z: {np.min(points3D[:, 2])}")
    print(f"Max Y: {np.max(points3D[:, 1])}")
    print(f"Min Y: {np.min(points3D[:, 1])}")
    print(f"Max X: {np.max(points3D[:, 0])}")
    print(f"Min X: {np.min(points3D[:, 0])}")
    visualize_with_open3d(points3D, colors_bgr2)


    for unique_id, point in unique_id2reconstructed.items():
        if unique_id in visited_tracks:
            if unique_id2reconstructed[unique_id] not in points3D:
                visited_tracks[unique_id] = False
            else:
                visited_tracks[unique_id] = True



    mean_error_afterFiltering = calculate_reprojection_error(
        tracks, camera_poses, keypoint_id_map, all_kps, K, unique_id2reconstructed,visited_tracks
    )
    print(f"reprojection error after filtering {mean_error_afterFiltering}")

    bundle_adjustment(camera_poses,unique_id2reconstructed,tracks,keypoint_id_map,all_kps,K,visited_tracks)

    mean_error_afterFilteringAndBA = calculate_reprojection_error(
        tracks, camera_poses, keypoint_id_map, all_kps, K, unique_id2reconstructed,visited_tracks
    )
    print(f"reprojection error after filtering and BA {mean_error_afterFilteringAndBA}")

    reconstructed_points = []
    colors = []

    # iterative over all the 3D Points
    for unique_id in unique_id2reconstructed:
        if unique_id in visited_tracks and visited_tracks[unique_id]==True:
            point = unique_id2reconstructed[unique_id]
            reconstructed_points.append(point)
            idx = list(unique_id2reconstructed.keys()).index(unique_id)
            colors.append(colors_bgr[idx])

    reconstructed_points = np.array(reconstructed_points)
    visualize_with_open3d(reconstructed_points,np.array(colors))
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


    filename = f"{folder_path}_{current_time}.npy"
    np.save(filename, reconstructed_points)
    print(f"Reconstructed points saved as {filename}")
    np.save(f"{folder_path}_colors_{current_time}.npy", np.array(colors))




if __name__ == "__main__":
    
    SFM()
    # points,colors=load_reconstructed_data("cupcake_final_last_one_20241203_140458.npy","cupcake_final_last_one_colors_20241203_140458.npy")
    # threshold = 2
    # std_dev = np.std(points, axis=0)
    # mean = np.mean(points, axis=0)

    # valid_mask=np.all(np.abs(points-mean)<=threshold*std_dev,axis=1)
    # points_3d =points[valid_mask]
    # colors_bgr2= colors[valid_mask]
    # visualize_with_open3d(points_3d,colors_bgr2)
    
