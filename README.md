# 3D Food Reconstruction from 2D Images

## Project Overview

**VIP-IPA: 3D Food Reconstruction From 2D Images** is a project aimed at reconstructing 3D models of food items by leveraging classical geometric techniques. The project was developed **from scratch** and focuses on **sparse reconstruction**, meaning the generated 3D models are built from key points rather than dense surfaces. This approach is computationally efficient and provides a foundational understanding of 3D reconstruction pipelines.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Camera Calibration](#camera-calibration)
  - [Feature Matching](#feature-matching)
  - [Structure From Motion (SfM)](#structure-from-motion-sfm)
- [References](#references)

## Introduction

The project applies **incremental structure from motion (SfM)** to construct 3D models from sets of 2D images taken from different perspectives. By gradually adding images to the reconstruction process, the model is refined and enhanced, leading to more accurate 3D representations of the target objects. 

This method, though sparse, lays the groundwork for future enhancements, such as dense reconstruction or texture mapping, to create more detailed models.

## Dataset

The dataset for this project is sourced from the [CVPR MetaFood 3D Challenge (Kaggle)](https://www.kaggle.com/competitions/cvpr-metafood-3d-food-reconstruction-challenge/data) and other sources. The dataset consists of 200-image sets per food item, captured from various angles, ensuring thorough coverage for the 3D reconstruction process. Each image set includes:

- **Checkerboard Calibration**: A checkerboard with known dimensions placed next to the food object, essential for accurate scale calibration.
- **Masked Images**: Highlighting the food object, which aids in separating the object from the background during reconstruction.
- **Depth Data**: While depth images are provided, the current project focuses on sparse 3D reconstruction and does not incorporate depth data for dense modeling.

Sample images from the dataset are shown below:

<p align="center">
  <img src="https://github.com/abdelrahmanHamdyG/Structure-From-motion-/blob/master/Dataset%20sample-Nachos.jpg?raw=true" width="45%" />
  <img src="https://github.com/abdelrahmanHamdyG/Structure-From-motion-/blob/master/Dataset-cupcake.jpg?raw=true" width="45%" />
</p>

A full detailed report of this project, along with the **3D reconstruction results**, is available at the following link:  
**[3D Food Reconstruction Report and Results](https://drive.google.com/drive/u/1/folders/1LpuLYpan7PQoMBKjhf73XtZdkul945AO)**.

## Methodology

### Camera Calibration

Camera calibration is a crucial step in the 3D reconstruction pipeline, as it ensures accurate mapping between the 3D world and the 2D image plane. The project employs **Zhang's Method** for calibration, a widely-used algorithm that estimates the intrinsic and extrinsic parameters of the camera by analyzing images of a checkerboard pattern.

#### Key Steps:

1. **Capture Images**: Images of a checkerboard pattern are captured at different angles.
2. **Corner Detection**: The **Harris corner detection** algorithm identifies key points on the checkerboard.
3. **Homography Calculation**: The relationship between the 3D points and their 2D projections is computed.
4. **Estimate Intrinsics**: The intrinsic parameters (focal length, optical center, skew) are estimated to finalize calibration.

### Feature Matching

Accurate feature matching is essential for aligning multiple images in the reconstruction process. This project employs **SIFT (Scale Invariant Feature Transform)**, known for its robustness and accuracy, especially in 3D reconstruction tasks.

#### SIFT Pipeline:

1. **Keypoint Detection**: Identify distinctive keypoints across the image at different scales.
2. **Descriptor Generation**: Generate descriptors to characterize each keypoint.
3. **Keypoint Matching**: Match keypoints across images based on descriptor similarity.
4. **Outlier Removal**: Apply **RANSAC (Random Sample Consensus)** to filter out erroneous matches and ensure geometric consistency.

### Structure From Motion (SfM)

SfM constructs a sparse 3D model by incrementally adding images and triangulating points from multiple viewpoints. The core steps include:

- **Initial Pair Selection**: The process begins by selecting two images with maximum feature overlap.
- **Incremental Addition**: New images are iteratively added, refining the 3D model by triangulating new keypoints.
- **Bundle Adjustment**: A non-linear optimization method that minimizes reprojection error, ensuring the accuracy of the reconstructed model.

SfM, although sparse, provides a reliable framework to understand 3D reconstruction principles and serves as a foundation for future dense modeling projects.

## References

1. Lowe, D. G. "Distinctive Image Features from Scale-Invariant Keypoints." 2004.  
2. Zhang, Z. "A Flexible New Technique for Camera Calibration." IEEE PAMI, 2000.  
3. Wu, C. C. "VisualSFM: A Visual Structure from Motion System."  
4. CVPR MetaFood 3D Challenge (Kaggle).  
5. Schöenberger, J. L., and Frahm, J. "Structure-from-Motion Revisited." CVPR 2016.
