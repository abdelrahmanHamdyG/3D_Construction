import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy import cos, sin, deg2rad, exp, floor, sqrt, array


SIGMA = 1.6
INTERVALS = 6
K = 2**(1/3)

#used https://github.com/rmislam/PythonSIFT/blob/master/pysift.py  as a refrence for some parts in the code

# Ghania And Shebl 31/10  

# calculating sigmas beween images in the same octav
def get_sigma_diff(wanted, prev):
    return np.sqrt(max((wanted ** 2) - ((prev) ** 2), 0.01))


# downsampling images to accomodate multiple size scales
def generate_octaves(img, sigmas, octaves):
    all_images = []
    current_octave = []
    # print(octaves)
    for i in range(octaves):
        current_octave.append(img)
        for sigma in sigmas:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
            current_octave.append(img)

        all_images.append(current_octave)
        # using the second last element in an octav for the next octav 
        img = cv2.resize(current_octave[-3], (int(current_octave[-3].shape[1] // 2), int(current_octave[-3].shape[0] // 2)), interpolation=cv2.INTER_NEAREST)
        current_octave = []
    return all_images

# generating Difference of Gaussians
def generate_dogs(imgs, octaves):
    dogs = []
    for i in range(octaves):
        DoGs_per_octav = []
        for j in range(5):
            first = imgs[i][j]
            second = imgs[i][j+1]

            current_dog = cv2.subtract(second ,first)
            DoGs_per_octav.append(current_dog)
        dogs.append(DoGs_per_octav)
    return dogs






# using numerical diff to find the the gradient
def compute_gradient(window):
    # f' = (f(x+1)-f(x-1))/2
    dx=0.5*(window[1,1,2]-window[1,1,0])
    dy=0.5*(window[1,2,1]-window[1,0,1])
    dScale=0.5*(window[2,1,1]-window[0,1,1])
    return np.array([dx,dy,dScale])

# using numerical diff to find the hessian (second derivative)
def compute_hessian(window):

    # F''(x)=f(x+1)-2*f(x)+f(x-1)
    
    dxx = window[1, 1, 2] - 2 * window[1, 1, 1] + window[1, 1, 0]
    dyy = window[1, 2, 1] - 2 * window[1, 1, 1] + window[1, 0, 1]
    dss = window[2, 1, 1] - 2 * window[1, 1, 1] + window[0, 1, 1]
    dxy = 0.25 * (window[1, 2, 2] - window[1, 2, 0] - window[1, 0, 2] + window[1, 0, 0])
    dxs = 0.25 * (window[2, 1, 2] - window[2, 1, 0] - window[0, 1, 2] + window[0, 1, 0])
    dys = 0.25 * (window[2, 2, 1] - window[2, 0, 1] - window[0, 2, 1] + window[0, 0, 1])
    return np.array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])



def get_keypoints(dogs, octaves,sigmas,all_imgs):
    keypoints = []
    l=[]
    for i in range(octaves):
        for j in range(1, len(dogs[i]) - 1):
            prev, cur, next = dogs[i][j - 1], dogs[i][j], dogs[i][j + 1]
            for x in range(5, cur.shape[0] - 5):
                for y in range(5, cur.shape[1] - 5):
                    #eliminating DoGs with pixel value less than 1  
                    if abs(cur[x, y]) < 1:
                        continue

                    # getting the 26 neighbours beside you (9 above )(9 below) and (8 on the same level)
                    neighbors = np.concatenate([
                        prev[x-1:x+2, y-1:y+2].flatten(),
                        cur[x-1:x+2, y-1:y+2].flatten(),
                        next[x-1:x+2, y-1:y+2].flatten()
                    ])

                    #checking if pixel is maximum or minimum 
                    value = cur[x, y]
                    if value > 0:
                        if np.abs((value -np.max(neighbors)))>0.000001: # !=
                            continue
                    else:
                        if np.abs((value -np.min(neighbors)))>0.000001: #!=
                            continue
                    l.append((x,y))
                    # refining Key points using quadratic fit 
                    kp=refine_key_points(x,y,j,i,dogs[i],sigmas)
                    if kp is not None:
                        
                        
                        kp=compute_orientation(kp,all_imgs[i][kp[3]])
                        keypoints.extend(kp)

    return keypoints



def refine_key_points(x,y,s,octv_idx,dogs_in_octav,sigmas):
    for attempt in range(5):
        prev,cur,next=dogs_in_octav[s-1],dogs_in_octav[s],dogs_in_octav[s+1]
        window = np.stack([prev[x-1:x+2, y-1:y+2],
                            cur[x-1:x+2, y-1:y+2],
                            next[x-1:x+2, y-1:y+2]])
        window=window.astype('float32')/255.0

        
        gradient=compute_gradient(window)
        hessian=compute_hessian(window)

        # solving for x_hat (offset) using least square 
        x_hat=-np.linalg.lstsq(hessian,gradient,rcond=None)[0]



        # checking if the point is close enough 
        if np.all(np.abs(x_hat[:3]) < 0.5):
            break
        
        # adjusting values of the pixel based on the quadratic fit
        y+=int(round(x_hat[0]))
        x+=int(round(x_hat[1]))
        s+=int(round(x_hat[2]))

        if s<1 or s>3:
            return None 

        if not isValidPoint((x,y),dogs_in_octav[s]) or attempt==4:
            return None
        
    contrast=window[1,1,1]+0.5*((gradient.T)@x_hat)
    
    if np.abs(contrast)*3 >=0.04:

        trace=hessian[0,0]+hessian[1,1]
        det=hessian[0,0]*hessian[1,1]-hessian[1,0]**2

        
        # eliminate the edges using r=10 as in the paper
        if det>0 and ((trace*trace)/det)<12.1:
            size = (2 ** ((s + x_hat[2]) / 3.0)) * (2 ** (octv_idx+1))
            return ((x+x_hat[1])*2**(octv_idx), (y+x_hat[0])*2**(octv_idx), octv_idx, s, size*1.6)
        
h2=0
aaaa=[]
# shebl, Ghania 11/2/2024
def compute_orientation(point,img):
    
    global aaaaa
    aaaa.append(point)
    x, y, octave, img_idx,size = point 

    scale=1.5*(size)/np.float32(2**(octave+1))
    window_size=int(round(3*scale))

    

    key_points_with_dominant_orientation = []
    orientation_histogram = np.zeros(36)  

    for i in range(-window_size, window_size + 1):
        new_x = i + int(round(x/np.float32(2**octave)))
        if new_x > 0 and new_x < img.shape[0] - 1:
            for j in range(-window_size, window_size + 1):
                
                new_y = j + int(round(y/np.float32(2**octave)))
                if new_y > 0 and new_y < img.shape[1] - 1:
                

                    mag, dir = mag_and_direction(img, (new_x, new_y))
                    
                    weight = np.exp( (i**2 + j**2)*(-0.5/(scale**2)))  
                    specific_bin = int(round(dir / 10)) % 36 
                    orientation_histogram[specific_bin] += mag * weight
    

    smooth_histogram=np.zeros(36)
    for n in range(36):
        smooth_histogram[n] = (6 * orientation_histogram[n] + 4 * (orientation_histogram[n - 1] + orientation_histogram[(n + 1) % 36]) + orientation_histogram[n - 2] + orientation_histogram[(n + 2) % 36]) / 16.0
    dominant_orientation=np.max(smooth_histogram)
    local_peaks=get_local_peaks(smooth_histogram)
    
    
    for peak_idx in local_peaks:
        local_peak_value=smooth_histogram[peak_idx]
        if local_peak_value >= 0.8 * dominant_orientation:            
            # indices=[(peak_idx-1)%36, peak_idx,(peak_idx+1)%36]
            # values=[smooth_histogram[i] for i in indices]
            # coef=np.polyfit(values,indices,2)
            # a,b,c=coef
            # values_peak=-b/(2*a)
            # index_peak=np.polyval(coef,values_peak)
            left_value = smooth_histogram[(peak_idx - 1) % 36]
            right_value = smooth_histogram[(peak_idx + 1) % 36]
            index_peak = (peak_idx + 0.5 * (left_value - right_value) / (left_value - 2 * local_peak_value + right_value)) % 36
            orientation=360.-index_peak*360./36
            if abs(orientation-360.)<( 1e-7):
                orientation=0    
            key_points_with_dominant_orientation.append((x,y,octave,img_idx,size,orientation))
    
    # if len(key_points_with_dominant_orientation)>1:
    #     global h2
    #     h2+=1
    #     print(f"x {new_x} y {new_y}    {len(key_points_with_dominant_orientation)}")

    return key_points_with_dominant_orientation

                

            



    
    








#Ahmed Shebl 10/21
#Zeyad 10/21



def save_keypoints(keypoints,file_name="first"):
    # Save the keypoints list using pickle
    with open(file_name, 'wb') as f:
        pickle.dump(keypoints, f)

def load_keypoints(file_name="first"):
    
    with open(file_name, 'rb') as f:
        return pickle.load(f)



def plot_keypoints_on_image(image, keypoints, flag=True):
    print("Plotting keypoints on image")
    
    # Display the original grayscale image
    plt.imshow(image, cmap="gray")
    
    for kp in keypoints:
    
        x, y, octave, img_idx, size, orientation = kp
        if flag:
            # Plot the keypoint as a red circle
            plt.plot(y, x, 'ro', markersize=2)  # 'ro' for red circles
        else:
            # Plot red circle for keypoint
            # plt.plot(y, x, 'ro', markersize=4)
            
            # Calculate the end point for the orientation arrow
            orientation_rad = np.deg2rad(orientation)
            arrow_length = size / 2.0
            arrow_x = x + arrow_length * np.cos(orientation_rad)
            arrow_y = y + arrow_length * np.sin(orientation_rad)
            
            # Draw arrow showing the orientation
            plt.arrow(y, x, arrow_y - y, arrow_x - x, color='blue', head_width=2, head_length=3, width=0.7)

    # Set the title based on the flag
    
    plt.title("Keypoints on Image" if flag else "Keypoints with Orientation")
    plt.axis("off")
    plt.show()
        


def mag_and_direction(img,point):
    x=point[0]
    y=point[1]
    
    dx=img[x][y+1]-img[x][y-1]
    dy=img[x-1][y]-img[x+1][y]
    
    mag=np.sqrt(dx*dx+dy*dy)
    dir = np.arctan2(dy, dx) * (180.0 / np.pi)
    
    if dir< 0:
        dir += 360.0
    return mag,dir




# checking if a point is in the boundaries 
# Ghania 10/22
def isValidPoint(point,img):
    if point[0] <= 5 or point[0] >= img.shape[0] - 5 or point[1] <= 5 or point[1] >= img.shape[1] - 5:
        return False
    
    
    return True


def get_local_peaks(l):
    peaks=[]

    for i in range(0,len(l)):
        if l[i]>l[(i-1)%36] and l[i]>l[(i+1)%36]:
            peaks.append(i)
    
    return peaks







# computing the dominant orientation


def plot_keypoints_with_orientations(image, keypoints):

    print("we started plotting now ")
    output_image = image.copy()
    
    for point in keypoints:
        x, y, octave, val,img , orientation = point
        
        
        scale_factor = 2 ** (octave)
        new_x = int(x * scale_factor)
        new_y = int(y * scale_factor)
        
        
        cv2.circle(output_image, (new_y, new_x), 5, (0, 0, 255), 2)  
        
        
        orientation_rad = np.deg2rad(orientation)
        arrow_length = 20  
        arrow_x = int(new_x + arrow_length * np.sin(orientation_rad))
        arrow_y = int(new_y + arrow_length * np.cos(orientation_rad))
        
        
        cv2.arrowedLine(output_image, (new_y, new_x), (arrow_y, arrow_x), (0, 255, 0),5 , tipLength=0.4)  
    
    
    plt.imshow(output_image)
    plt.show()

# find the distance descriptors using L2 norm  
# Shebl 10/22
def match_descriptors(descriptors1, descriptors2):
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    matches = bf.knnMatch(np.array(descriptors1, dtype=np.float32), np.array(descriptors2, dtype=np.float32), k=2)

    
    good_matches = []
    for best, secondBest in matches:
        if best.distance < 0.8 * secondBest.distance:
            good_matches.append(best)
    
    return good_matches



import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_matches(img1, img2, kp1, kp2, good_matches, min_matches=10):
    
    print(len(good_matches))
    good_matches=good_matches[1:300]
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Check if there are at least one match
    if len(good_matches) >= 1:
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        n_width = w1 + w2
        n_height = max(h1, h2)
        hdif = int((h2 - h1) / 2) if h1 < h2 else 0

        
        new_img = np.zeros((n_height, n_width, 3), dtype=np.uint8)
        new_img[hdif:hdif + h1, :w1] = img1_color
        new_img[:h2, w1:w1 + w2] = img2_color

        
        for m in good_matches:
            
            pt1 = (int(kp1[m.queryIdx][1]), int(kp1[m.queryIdx][0] + hdif))
            pt2 = (int(kp2[m.trainIdx][1] + w1), int(kp2[m.trainIdx][0]))

            
            cv2.circle(new_img, pt1, 2, (0, 255, 0), -1)  
            cv2.circle(new_img, pt2, 2, (0, 0, 255), -1)  

            
            cv2.line(new_img, pt1, pt2, (255, 0, 0), 1)  

        plt.figure(figsize=(15, 10))
        plt.imshow(new_img)
        plt.axis('off')
        plt.title(f"Matchs Found")
        plt.show()
    else:
        print(f"No matches")
    

# generating descriptors for the keypoints
def generate_descriptors(keypoints, all_imgs,):
    descriptors = []
    print("Generating descriptors...")
    
    for point in keypoints:
        
        # extracting the original attributes of the key point 

        x, y, octave, img_idx, size, orientation = point

        # Getting at which size the image was blurred 
        scale = 1 / 2**(octave - 1) if octave >= 0 else 2**(octave + 1)

        img = all_imgs[octave][img_idx]
        rows, cols = img.shape

        
        # getting the new position of the point with respect to the octav it was in it
        new_x_pt = int(round(scale * x))
        new_y_pt = int(round(scale * y))

        bins_per_degree = 1 / 45
        new_orientation = 360.0 - orientation
        sin_d = sin(deg2rad(new_orientation))
        cos_d = cos(deg2rad(new_orientation))
        hist_width = 1.5 * scale * size
        half_width = min(int(round(hist_width * sqrt(2) * 2.5)), int(sqrt(rows**2 + cols**2)))
        weight_multiplier = -0.5 / (0.5 * 4) ** 2

        # Histogram for each keypoint that will generate the descriptor 
        descriptor_hist = np.zeros((6, 6, 8))

        # Loop through pixels within half-width
        for i in range(-half_width, half_width + 1):
            for j in range(-half_width, half_width + 1):

                # Rotate the coordinates
                new_i = j * sin_d + i * cos_d
                new_j = j * cos_d - i * sin_d

                # Compute bin positions
                row_bin = (new_i / hist_width) + 1.5
                col_bin = (new_j / hist_width) + 1.5

                
                if -1 < row_bin < 4 and -1 < col_bin < 4:
                    window_row = int(round(new_x_pt + i))
                    window_col = int(round(new_y_pt + j))

                    if 0 < window_row < rows - 1 and 0 < window_col < cols - 1:
                        

                        gradient_magnitude,dir=mag_and_direction(img,(window_row,window_col))
                        gradient_orientation = (dir % 360) - new_orientation
                        gradient_orientation = (gradient_orientation * bins_per_degree) % 8
                        weight = exp(weight_multiplier * ((new_i / hist_width) ** 2 + (new_j / hist_width) ** 2))

                        # Trilinear interpolation from opencv internal implementation and  Russ Islam Github and it is giving better results

                        r_bin_floor, c_bin_floor, o_bin_floor = floor([row_bin, col_bin, gradient_orientation]).astype(int)
                        row_fraction = row_bin - r_bin_floor
                        col_fraction = col_bin - c_bin_floor
                        orientation_fraction = gradient_orientation - o_bin_floor
                        o_bin_floor = int(o_bin_floor % 8)  

                        
                        c1 = gradient_magnitude * weight * row_fraction
                        c0 = gradient_magnitude * weight * (1 - row_fraction)
                        c11, c10 = c1 * col_fraction, c1 * (1 - col_fraction)
                        c01, c00 = c0 * col_fraction, c0 * (1 - col_fraction)
                        c111, c110 = c11 * orientation_fraction, c11 * (1 - orientation_fraction)
                        c101, c100 = c10 * orientation_fraction, c10 * (1 - orientation_fraction)
                        c011, c010 = c01 * orientation_fraction, c01 * (1 - orientation_fraction)
                        c001, c000 = c00 * orientation_fraction, c00 * (1 - orientation_fraction)


                        descriptor_hist[r_bin_floor + 1, c_bin_floor + 1, o_bin_floor] += c000
                        descriptor_hist[r_bin_floor + 1, c_bin_floor + 1, (o_bin_floor + 1) % 8] += c001
                        descriptor_hist[r_bin_floor + 1, c_bin_floor + 2, o_bin_floor] += c010
                        descriptor_hist[r_bin_floor + 1, c_bin_floor + 2, (o_bin_floor + 1) % 8] += c011
                        descriptor_hist[r_bin_floor + 2, c_bin_floor + 1, o_bin_floor] += c100
                        descriptor_hist[r_bin_floor + 2, c_bin_floor + 1, (o_bin_floor + 1) % 8] += c101
                        descriptor_hist[r_bin_floor + 2, c_bin_floor + 2, o_bin_floor] += c110
                        descriptor_hist[r_bin_floor + 2, c_bin_floor + 2, (o_bin_floor + 1) % 8] += c111

        
        descriptor = descriptor_hist[1:-1, 1:-1, :].flatten()  # Remove borders to make it again 4*4*8
        threshold = np.linalg.norm(descriptor) * 0.2
        descriptor = np.clip(descriptor, 0, threshold)  
        descriptor /= max(np.linalg.norm(descriptor), 1e-7)
        descriptor = np.clip(np.round(512 * descriptor), 0, 255).astype(np.uint8)
        descriptors.append(descriptor)

    return np.array(descriptors, dtype='float32')

# comparing float to avoid precision problems 
def cmpFloat(f1,f2):

    return abs(f1-f2)<0.00001


    


descriptors1=[]
descriptors2=[]

# remove duplicates 
def filter_points(l):

    sz=len(l)
    new_key_points=[]
    
    
    for i in range(sz):
        flag=True
        first=l[i]
        # print(first)
        
        for j in range(i+1,sz):
            
            second=l[j]
            # print(first)

            if cmpFloat(first[0],second[0]) and cmpFloat(first[1],second[1]) and cmpFloat(first[2],second[2])  and cmpFloat(first[4],second[4]) and cmpFloat(first[5],second[5]) :
                flag=False
                break
        if  flag:
            new_key_points.append(first)
    return new_key_points



# convert the scale of the points to the size of the original one we uploaded
def convert_to_orignial_image(keypoints):

    converted=[]
    for p in keypoints:

        converted.append((p[0]*0.5,p[1]*0.5,p[2],p[3],p[4]*0.5,p[5]))
    return converted


def convert_to_8bit(image):
    return cv2.convertScaleAbs(image, alpha=255.0 / image.max())



# normal mode (True) and if False the program will  use the prevoiusly calculated keypoints and descriptors
if True:


    img1 = cv2.imread("tacos_segmented_ransac/198.jpg",0)
    img2 = cv2.imread("tacos_segmented_ransac/195.jpg",0)  

    img1 = img1.astype('float32')
    height1, width1 = img1.shape[:2]
    img1 = cv2.resize(img1, (width1 * 2,height1 * 2), interpolation=cv2.INTER_LINEAR)
    img1=cv2.GaussianBlur(img1,(0,0),sigmaX=1.24899,sigmaY=1.24899)


    img2 = img2.astype('float32')
    height2, width2 = img2.shape[:2]
    img2 = cv2.resize(img2, (width2 * 2,height2 * 2), interpolation=cv2.INTER_LINEAR)
    img2=cv2.GaussianBlur(img2,(0,0),sigmaX=1.24899,sigmaY=1.24899)


    
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # height2, width2 = img2.shape[:2]
    # img2 = cv2.resize(img2, (height2 * 2, width2 * 2), interpolation=cv2.INTER_LINEAR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    
    octaves1 =  int(round(np.log(min(img1.shape)) / np.log(2) - 1))
    sigmas1 = [get_sigma_diff(SIGMA * (K ** i), SIGMA * (K ** (i - 1))) for i in range(1, 6)]
    all_imgs1 = generate_octaves(img1, sigmas1, octaves1)
    dogs1 = generate_dogs(all_imgs1, octaves1)


    key_points1 = get_keypoints(dogs1, octaves1,sigmas1,all_imgs1)
    print("before flatten")
    print(len(key_points1))

    print("filtered")
    key_points1=filter_points(key_points1)
    print(len(key_points1))
    key_points1=convert_to_orignial_image(key_points1)
    print(key_points1[0][0])
    save_keypoints(key_points1, "keypoints1_ours.pkl")      
    descriptors1=generate_descriptors(key_points1,all_imgs1)
    save_keypoints(descriptors1,"descriptors1.pkl")




    
    octaves2 =  int(round(np.log(min(img2.shape)) / np.log(2) - 1))
    sigmas2 = [get_sigma_diff(SIGMA * (K ** i), SIGMA * (K ** (i - 1))) for i in range(1, 6)]
    all_imgs2 = generate_octaves(img2, sigmas2, octaves2)
    dogs2 = generate_dogs(all_imgs2, octaves2)

    
    key_points2 = get_keypoints(dogs2, octaves2,sigmas2,all_imgs2)
    print("before flatten 2")
    print(len(key_points2))

    print("filtered")
    key_points2=filter_points(key_points2)
    print(len(key_points2))
    key_points2=convert_to_orignial_image(key_points2)
    print(key_points2[0][0])
    save_keypoints(key_points2, "keypoints2_ours.pkl")      
    descriptors2=generate_descriptors(key_points2,all_imgs2)
    save_keypoints(descriptors2,"descriptors2.pkl")
    

    
    
    matches=match_descriptors(descriptors1,descriptors2)
    
    
    
    
    
    img1_8bit = convert_to_8bit( cv2.imread("tacos_segmented_ransac/198.jpg",0))
    img2_8bit = convert_to_8bit( cv2.imread("tacos_segmented_ransac/195.jpg",0) )

    
    
    
    draw_matches(img1_8bit,img2_8bit,key_points1,key_points2,matches)
    


    # key_points1_flatten=list(set(key_points1_flatten))
    # print(len(key_points1))
    # exit()


    plot_keypoints_on_image(img1,key_points1,False)
else:
    
    img1 = cv2.imread("tacos_segmented_ransac/198.jpg",0)
    img2 = cv2.imread("tacos_segmented_ransac/195.jpg",0)  

    img1 = img1.astype('float32')
    height1, width1 = img1.shape[:2]
    img1 = cv2.resize(img1, (width1 * 2,height1 * 2), interpolation=cv2.INTER_LINEAR)
    img1=cv2.GaussianBlur(img1,(0,0),sigmaX=1.24899,sigmaY=1.24899)


    img2 = img2.astype('float32')
    height2, width2 = img2.shape[:2]
    img2 = cv2.resize(img2, (width2 * 2,height2 * 2), interpolation=cv2.INTER_LINEAR)
    img2=cv2.GaussianBlur(img2,(0,0),sigmaX=1.24899,sigmaY=1.24899)


    #                        new_width, new_height
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # height2, width2 = img2.shape[:2]
    # img2 = cv2.resize(img2, (height2 * 2, width2 * 2), interpolation=cv2.INTER_LINEAR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    
    octaves1 =  int(round(np.log(min(img1.shape)) / np.log(2) - 1))
    sigmas1 = [get_sigma_diff(SIGMA * (K ** i), SIGMA * (K ** (i - 1))) for i in range(1, 6)]
    all_imgs1 = generate_octaves(img1, sigmas1, octaves1)
    # dogs1 = generate_dogs(all_imgs1, octaves1)


    key_points1 = load_keypoints("keypoints1_ours.pkl")

    
    descriptors1=load_keypoints("descriptors1.pkl")





    
    octaves2 =  int(round(np.log(min(img2.shape)) / np.log(2) - 1))
    sigmas2 = [get_sigma_diff(SIGMA * (K ** i), SIGMA * (K ** (i - 1))) for i in range(1, 6)]
    all_imgs2 = generate_octaves(img2, sigmas2, octaves2)
    

    
    
    
    key_points2=load_keypoints("keypoints2_ours.pkl")      
    
    descriptors2=load_keypoints("descriptors2.pkl")
    
    img1_8bit = convert_to_8bit( cv2.imread("tacos_segmented_ransac/198.jpg",0))
    img2_8bit = convert_to_8bit( cv2.imread("tacos_segmented_ransac/195.jpg",0) )

    
    plot_keypoints_on_image(img1_8bit,key_points1,False)
    matches=match_descriptors(descriptors1,descriptors2)
    draw_matches(img1_8bit,img2_8bit,key_points1,key_points2,matches)
