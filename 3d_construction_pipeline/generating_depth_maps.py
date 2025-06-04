# generate_depth.py

import os
import numpy as np
from PIL import Image
import torch
import depth_pro
from torchvision import transforms
from tqdm import tqdm

def run_depth_generation():
    IMAGE_FOLDER = os.getenv("IMAGE_FOLDER")
    if not IMAGE_FOLDER:
        raise EnvironmentError("IMAGE_FOLDER environment variable not set.")

    input_dir = f"{IMAGE_FOLDER}_processed"
    output_dir = f"{IMAGE_FOLDER}_depths"
    os.makedirs(output_dir, exist_ok=True)

    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    fixed_f_px = torch.tensor(755.1956588142441)

    for filename in tqdm(sorted(os.listdir(input_dir))):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"depth_{os.path.splitext(filename)[0]}.npy")

            image = Image.open(input_path).convert("RGB")
            image_tensor = transform(image)

            with torch.no_grad():
                prediction = model.infer(image_tensor, f_px=fixed_f_px)
                depth_np = prediction["depth"].cpu().numpy()

            np.save(output_path, depth_np)
            print(f"Saved: {output_path}")
