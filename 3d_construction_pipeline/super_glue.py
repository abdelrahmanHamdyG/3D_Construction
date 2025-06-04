# generate_matches.py

import os
from natsort import natsorted
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm

def run_keypoint_matching():
    IMAGE_FOLDER = os.getenv("IMAGE_FOLDER")
    if not IMAGE_FOLDER:
        raise EnvironmentError("IMAGE_FOLDER environment variable not set.")

    image_dir = f"{IMAGE_FOLDER}_processed"
    mask_dir = f"{IMAGE_FOLDER}_masks"
    match_dir = f"{IMAGE_FOLDER}_matches"
    os.makedirs(match_dir, exist_ok=True)

    print("Loading SuperGlue model...")
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor", use_fast=False)
    model = AutoModel.from_pretrained("magic-leap-community/superglue_outdoor")
    model.eval()

    filenames = natsorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for i in tqdm(range(0,len(filenames)-1)):
        name0 = filenames[i]
        name1 = filenames[i + 1]

        image0 = Image.open(os.path.join(image_dir, name0)).convert("RGB")
        image1 = Image.open(os.path.join(image_dir, name1)).convert("RGB")
        images = [image0, image1]

        inputs = processor(images, return_tensors="pt", do_resize=False)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = [[(image.height, image.width) for image in images]]
        outputs = processor.post_process_keypoint_matching(outputs, target_sizes, threshold=0.3)
        
        output = outputs[0]

        kp0 = output["keypoints0"].numpy()
        kp1 = output["keypoints1"].numpy()
        scores = output["matching_scores"]
        top_k = 150
        if len(scores) > top_k:
            top_indices = np.argsort(scores)[-top_k:]  # get indices of top 200 scores
            kp0 = kp0[top_indices]
            kp1 = kp1[top_indices]
            scores = scores[top_indices]


        mask0_path = os.path.join(mask_dir, name0)
        mask1_path = os.path.join(mask_dir, name1)

        mask0 = np.array(Image.open(mask0_path).convert("L"))
        mask1 = np.array(Image.open(mask1_path).convert("L"))
        valid_mask0 = mask0 > 128
        valid_mask1 = mask1 > 128

        filtered_kp0, filtered_kp1, filtered_scores = [], [], []

        for pt0, pt1, s in zip(kp0, kp1, scores):
            x0, y0 = int(round(pt0[0])), int(round(pt0[1]))
            x1, y1 = int(round(pt1[0])), int(round(pt1[1]))

            if (0 <= x0 < valid_mask0.shape[1] and 0 <= y0 < valid_mask0.shape[0] and
                0 <= x1 < valid_mask1.shape[1] and 0 <= y1 < valid_mask1.shape[0]):

                if valid_mask0[y0, x0] and valid_mask1[y1, x1]:
                    filtered_kp0.append(pt0)
                    filtered_kp1.append(pt1)
                    filtered_scores.append(s)

        kp0 = np.array(filtered_kp0)
        kp1 = np.array(filtered_kp1)
        scores = np.array(filtered_scores)

        # Save visualization
        canvas = np.zeros((max(image0.height, image1.height), image0.width + image1.width, 3))
        canvas[:image0.height, :image0.width] = np.array(image0) / 255.0
        canvas[:image1.height, image0.width:] = np.array(image1) / 255.0

        plt.figure(figsize=(12, 6))
        plt.imshow(canvas)
        for (x0, y0), (x1, y1), s in zip(kp0, kp1, scores):
            color = plt.get_cmap("cool")(s)
            plt.plot([x0, x1 + image0.width], [y0, y1], c=color, linewidth=0.5)
            plt.scatter(x0, y0, c='yellow', s=5)
            plt.scatter(x1 + image0.width, y1, c='yellow', s=5)
        plt.axis("off")
        plt.tight_layout()
        
        # plt.show()

        base0 = os.path.splitext(name0)[0]
        base1 = os.path.splitext(name1)[0]

        # Save matches
        match_path = os.path.join(match_dir, f"matches_{base0}_{base1}.npy")
        np.save(match_path, {
            "keypoints0": kp0,
            "keypoints1": kp1,
            "scores": scores
        })

        