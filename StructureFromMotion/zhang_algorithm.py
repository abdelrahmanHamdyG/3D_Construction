import cv2
import numpy as np 
import glob
import scipy.optimize

imgpoints = []
# Zeyad 10/28/2024

x_inner_corners = 3
y_inner_corners = 4
# real points as square size is 1.27 cm

all_points = []

def pts_3D(x_inner_corners, y_inner_corners, square_size):
    
    pts = np.array([
        [i * square_size, j * square_size, 0]
        for j in range(y_inner_corners)
        for i in range(x_inner_corners)
    ], dtype=np.float64)
    return pts



real_points = pts_3D(x_inner_corners, y_inner_corners, 1.27)
# print(real_points)

def projectPoints2(objpoints, rot, t, K, dist_coeffs):
    real_points_np = np.array(objpoints)
    
    rot= np.asarray(rot, dtype=np.float64).flatten()
    t = np.asarray(t, dtype=np.float64).flatten()

    # Convert rotation vec to rotation matrix
    R_mat = cv2.Rodrigues(rot)[0]

    
    objpoints_cam = np.dot(R_mat, real_points_np.T) +t.reshape(3, 1)  # Shape (3, N)

    # Normalize coordinates
    x = objpoints_cam[0 ,:] / objpoints_cam[2,:]
    y = objpoints_cam[1,:] / objpoints_cam[2,:]

    # Apply distortion
    k1, k2, p1, p2, k3 = dist_coeffs.flatten()
    r2 = x**2 + y**2
    radial = 1+ k1* r2+ k2 * r2**2 + k3 * r2**3
    x_tangential = 2* p1 * x * y + p2 * (r2 + 2 * x**2)
    y_tangential = p1* (r2 + 2 * y**2) + 2 * p2 * x * y
    x_distorted = x* radial + x_tangential
    y_distorted = y *radial + y_tangential

    
    u = K[0,0] * x_distorted + K[0,2]
    v = K[1,1] * y_distorted + K[1,2]

    imgpoints_proj = np.vstack((u, v)).T  
    return imgpoints_proj

def compute_Vij_Transpose(H, i, j):
    v_ij = np.array([
        H[0, i] * H[0, j],  
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j], 
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],  
        H[2, i] * H[2, j]   
    ])
    return v_ij;


# computing B 
def compute_B(all_h):

    V=[]
    for h in all_h:
        v12=compute_Vij_Transpose(h,0,1)
        v11=compute_Vij_Transpose(h,0,0)
        v22=compute_Vij_Transpose(h,1,1)
        V.append(v12)
        V.append(v11-v22)
    
    V=np.array(V)
    # getting the last component of svd and the solution is the last row in the last component
    b=np.linalg.svd(V)[2][-1]
    b/=b[-1]
    
    
    b[3],b[2]=b[2],b[3]

    
    B=[[b[0],b[1],b[2]],
       [b[1],b[3],b[4]],
       [b[2],b[4],b[5]]]
    
    B=np.array(B)
    

    return  B



# extracting the intriniscs 
def compute_K(B):
        # print(B)
        # exit()
    # from professor Kak's lectures 
    # Equation:
    #     K = [[alpha_x, s, x_0],
    #          [0,     alpha_y,  y0],
    #          [0,     0,      1]]
    # y0=v0  x0=u0  s=gamma alpha_x=alpha alpha_y=beta

        denom = B[0, 0] * B[1, 1] - B[0, 1] ** 2
        
        y0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / denom
        

        

        lambda_ = B[2, 2] - (B[0, 2] ** 2 + y0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]

        alpha_x = np.sqrt(lambda_ / B[0, 0])
        alpha_y = np.sqrt(lambda_ * B[0, 0] / denom)

        
        s = 0

        
        x0 = -B[0][2]/B[0][0]

        
        K = np.array([[alpha_x, s, x0],
                      [0,      alpha_y, y0],
                      [0,      0,     1]])

        
        return K


# Shebl 10/23/2024
# projecting 3D points back using the projection matrix 
    


def compute_H(image,  num):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # finding corners of the checkerboard 
    ret, corners = cv2.findChessboardCorners(gray, (3, 4), )
    
    if ret:
        
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        squeezed_corners=corners_refined.squeeze()
        imgpoints.append(squeezed_corners)
        all_points.append(squeezed_corners)
        num_of_corners=corners_refined.shape[0]
        
        # adding one extra dimension for the homogonoius part
        corners_homogeneous = np.hstack([squeezed_corners, np.ones((num_of_corners, 1))])
        
        
        # solving for H 
        A = np.empty((0, 9))  

        for i in range(num_of_corners):
            x_2d = corners_homogeneous[i][0]
            y_2d = corners_homogeneous[i][1]
            x_3d = real_points[i][0]
            y_3d = real_points[i][1]

            axi = [-x_3d, -y_3d, -1, 0, 0, 0, x_2d * x_3d, x_2d * y_3d, x_2d]
            ayi = [0, 0, 0, -x_3d, -y_3d, -1, y_2d * x_3d, y_2d * y_3d, y_2d]
            
            A=np.vstack([A,axi])
            A=np.vstack([A,ayi])
        
        last_component_of_svd=np.linalg.svd(A)[2]
        
        last_row=last_component_of_svd[-1]
        H=last_row.reshape(3,3)
        H=H/H[2][2]
        # print(H)
        
        return H
    else:
        print(f"Chessboard corners not found in image {num}.")
        return None
    
# Ghania 10/23/2024
# we compute the extrinsic of each image  because they differ for each image unlike the intrinsics

def loss(params,num_images):
    
    intrinsic_params = params[:4]
    K=np.array([
        [intrinsic_params[0],0,intrinsic_params[2]]
        ,[0,intrinsic_params[1],intrinsic_params[3]]
        ,[0,0,1]])
    
    distorion_params = np.array(params[4:9])
    start=9
    

    total_projected_points = []
    for i in range(num_images):
        rot = np.array(params[start:start + 3]).flatten()
        t=np.array(params[start+3:start + 6]).flatten()
        start += 6
        projected_points = projectPoints2(real_points,rot,t,K,distorion_params)
        # print(projected_points.shape)
        total_projected_points.append(projected_points)
        


    # print("total_projected_points shape :",len(total_projected_points), len(total_projected_points[0]))
    
    residuals = []
    for i in range(num_images):
        residual = imgpoints[i] - total_projected_points[i]
        residuals.extend(residual.ravel())
    # print(np.array(residuals).shape)

    
    return np.array(residuals)



def compute_extrinsic_for_each_image(H,K):
    
    
    K_inv = np.linalg.inv(K)
    
    # print("H is ",H)

    RT = K_inv @ H
    # print(f"RT is {RT}")
    
    r1 = RT[:, 0]  # First column of K^-1 * H
    
    r2 = RT[:, 1]  # Second column of K^-1 * H
    t = RT[:, 2]   # Third column of K^-1 * H translation vector
    
    r1_norm=np.linalg.norm(r1)
    # print(f"r1_norm is {r1_norm}")
    r1 = r1 /r1_norm
    r2 = r2 / r1_norm
    t/=r1_norm

    r3=np.cross(r1,r2)


    R = np.column_stack((r1, r2, r3))
    U, sigma, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)
    R_det=np.linalg.det(R)
    
    if R_det<0:
        R=-R
    # print(f"R is {R}")
    
    # print(f"t is {t}")    
    rot_as_vec=cv2.Rodrigues(R)[0].flatten()
    t_as_vec=t.flatten()
    return np.column_stack((R, t)),rot_as_vec,t_as_vec



def get_intrinsic_params(img="rightcamera/*.png"):
    np.set_printoptions(precision=5, suppress=True)
    num=0

    real_points_list=[]
    all_H=[]
    all_images=[]
    # reading the images calibration_images
    for image_path in sorted(glob.glob(img)):
        img = cv2.imread(image_path)
        all_images.append(img)
        H=compute_H(img,num)
        all_H.append(H)
        real_points_list.append(real_points)
        num+=1

    print("all_H is ",all_H[0])
    
    B=compute_B(all_H)
    # print(B)
    K=compute_K(B)
    # print(f"K is {K}")
    # exit()
    all_RT=[]
    rot_as_vecs=[]
    t_as_vecs=[]
    for h in all_H:
        RT,rot_as_vec,t_as_vec=compute_extrinsic_for_each_image(h,K)
        all_RT.append(RT)
        
        rot_as_vecs.append(rot_as_vec)
        t_as_vecs.append(t_as_vec)  
        


    k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0

    alpha_x, alpha_y, x0, y0 = K[0, 0], K[1, 1], K[0, 2],K[1, 2]
    initial_params=[alpha_x, alpha_y, x0, y0, k1, k2, p1, p2, k3]

    extrinsic_param = []
    for rot_as_vec,t_as_vec in zip(rot_as_vecs, t_as_vecs):
        extrinsic_param.extend(rot_as_vec)
        extrinsic_param.extend(t_as_vec)
        
    initial_params.extend(extrinsic_param)

    initial_params = np.array(initial_params)


    result = scipy.optimize.least_squares(
            loss,
            initial_params,
            method='lm',
            args=( num,),
            max_nfev=7000  
        )
    total_params = result.x

    # Extract intrinsic and distortion parameters
    intrinsic_params = total_params[:4]
    distortion_params = total_params[4:9]
    K = np.array([
        [intrinsic_params[0], 0, intrinsic_params[2]],
        [0, intrinsic_params[1], intrinsic_params[3]],
        [0, 0, 1]
    ])
    # print(f"K is {K}")
    # print(f"distortion_params is {distortion_params}")

    # Extract optimized extrinsic parameters
    start = 9
    rot_as_vecs_opt = []
    t_as_vecs_opt = []
    for i in range(num):
        rot_vec = total_params[start:start + 3]
        t_vec = total_params[start + 3:start + 6]
        rot_as_vecs_opt.append(rot_vec)
        t_as_vecs_opt.append(t_vec)
        start += 6

    # Calculate the total reprojection error using optimized extrinsics
    total_error = 0
    nPoints = sum(len(imgp) for imgp in imgpoints)
    # print("nPoints is ", nPoints)

    for i in range(num):
        project_points = projectPoints2(real_points, rot_as_vecs_opt[i], t_as_vecs_opt[i], K, distortion_params)
        error = project_points - imgpoints[i]
        total_error += np.sum(error ** 2)

    rms_error = np.sqrt(total_error / nPoints)
    # print("RMS error is ", rms_error)
    return K,distortion_params
