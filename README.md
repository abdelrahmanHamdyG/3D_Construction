# 3D Food Reconstruction Pipeline

A computer vision pipeline that reconstructs detailed 3D models of food objects from multiple RGB images captured around the object. This project combines modern deep learning techniques with classical computer vision methods to create accurate 3D representations of food items.

<p align="center">
  <img src="https://github.com/abdelrahmanHamdyG/3D_Construction/blob/master/github_assets/22_images_allignment-ezgif.com-video-to-gif-converter.gif?raw=true" width="400"/>
</p>


*Example: 3D reconstruction of a burger from multiple viewpoints*

## What This Project Does

This pipeline transforms a collection of photographs taken around a food object into a complete 3D digital model. The process mimics how humans perceive depth and shape by combining information from multiple viewpoints, but uses advanced algorithms to achieve precise reconstruction.



**Multi-View 3D Reconstruction** is a fundamental problem in computer vision. When we take a photo, we lose the depth information - a 3D world gets projected onto a 2D image plane. This pipeline recovers that lost depth information by:

1. **Analyzing Multiple Perspectives**: Just like how our two eyes give us depth perception, multiple camera angles provide the geometric constraints needed to reconstruct 3D structure.

2. **Deep Learning Depth Estimation**: neural networks can predict how far each pixel is from the camera, even from a single image. This provides dense depth information that traditional methods struggle to achieve.

3. **Feature Correspondence**: The system identifies the same physical points across different images, creating a network of 3D constraints that guide the reconstruction process.

4. **Point Cloud Generation**: Each image contributes a "point cloud" - a collection of 3D points that represent the visible surface of the object from that viewpoint.

5. **Geometric Alignment**: All individual point clouds are aligned in a common 3D coordinate system, combining information from all viewpoints.

6. **Surface Reconstruction**: The final step converts the point cloud data into a smooth, continuous 3D surface using mathematical techniques like Poisson reconstruction.


## Sample Dataset

Here are examples of the input images used for reconstruction:

<table>
  <tr>
    <td><img src="https://github.com/abdelrahmanHamdyG/3D_Construction/blob/master/github_assets/Dataset-Burger_1.jpg?raw=true" width="300"/></td>
    <td><img src="https://github.com/abdelrahmanHamdyG/3D_Construction/blob/master/github_assets/Dataset-Burger_2.jpg?raw=true" width="300"/></td>
  </tr>
</table>

*Sample images showing different viewpoints of the food object*

## Installation
1. **Clone the repository**:
```bash
git clone (https://github.com/abdelrahmanHamdyG/3D_Construction)
cd 3d_construction
pip install -r requirements.txt
```

2. **Install SAM (Segment Anything Model)**:
Follow the installation guide: https://github.com/facebookresearch/segment-anything

3. **Install ML-Depth-Pro**:
Follow the installation guide: https://github.com/apple/ml-depth-pro

## Configuration

Create a `.env` file in the project root:
put your RGB images folder name 
and put other values based on your installation 
```env
MODEL_TYPE=vit_h
CHECKPOINT_PATH=models/SAM/sam_vit_h_4b8939.pth
DEVICE=cuda
IMAGE_FOLDER=assets/YourFoodName
MASK_OUTPUT_DIR=masks
DISPLAY_MAX_W=1024
```


### Run the Pipeline
```bash
cd 3d_construction
python 3d_construction_pipeline/main.py
```

The system will automatically process your images through all reconstruction stages and generate the final 3D model.

## How It Works

###  Image Preprocessing
Images are resized and prepared for optimal processing while maintaining quality and aspect ratios.

###  Object Segmentation
SAM  isolates the food object from the background in each image, creating  masks that focus reconstruction on the object of interest.

### Depth Estimation
ML-Depth-Pro analyzes each image to predict the distance of every pixel from the camera, creating detailed depth maps that capture the 3D structure.

###  Feature Matching
SuperGlue finds corresponding points between different images, establishing geometric relationships that constrain the 3D reconstruction.

###  Point Cloud Generation
Each image and its depth map are combined to create a 3D point cloud representing the visible surface from that viewpoint.

### : Multi-View Alignment
All point clouds are aligned in a common coordinate system using the feature correspondences and robust geometric algorithms.

###  Surface Reconstruction
The aligned point clouds are converted into a smooth 3D mesh using Poisson surface reconstruction, creating the final 3D model.


## Detailed Technical Report

For comprehensive technical details, methodology, and experimental results, please refer to our complete project report:

**[📄 Full Project Report](https://drive.google.com/file/d/19eQZyZglpFhFLolu9x1D3EyblrnUnhlh/view?usp=sharing)**

The report includes:
- Detailed mathematical formulations
- Experimental methodology and validation
- Comparison with alternative approaches
- Performance analysis and limitations


## Acknowledgments

This project builds upon 
- Segment Anything Model
- ML-Depth-Pro
- SuperGlue
- The broader computer vision research community

