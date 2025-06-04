import os
from PIL import Image
from torchvision import transforms

# Define directories
IMAGE_FOLDER = os.getenv("IMAGE_FOLDER")
output_dir = f"{IMAGE_FOLDER}_processed"
os.makedirs(output_dir, exist_ok=True)

# Resize to 720x960

def resize_images():
    transform = transforms.Compose([
        transforms.Resize((720, 960)),  # height, width
    ])

    processed_files = []
    for idx,filename in enumerate(os.listdir(IMAGE_FOLDER)):    
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(IMAGE_FOLDER, filename)
            output_path = os.path.join(output_dir, f"{idx}.jpg")

            image = Image.open(input_path).convert("RGB")
            processed_image = transform(image)
            processed_image.save(output_path)
            