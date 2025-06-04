# generate_masks.py
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from natsort import natsorted
from dotenv import load_dotenv
from segment_anything import sam_model_registry, SamPredictor

load_dotenv()

# Load configs from .env
MODEL_TYPE = os.getenv("MODEL_TYPE")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
DEVICE = os.getenv("DEVICE")
IMAGE_FOLDER = os.getenv("IMAGE_FOLDER")
DISPLAY_MAX_W = int(os.getenv("DISPLAY_MAX_W", "1024"))
MASK_OUTPUT_DIR = f"{IMAGE_FOLDER}_masks"




def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()

def iou_mask(m1, m2):
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return intersection / union if union != 0 else 0.0

def masked_ncc(img_patch, template, mask):
    img_patch_f, template_f = img_patch.astype(np.float32), template.astype(np.float32)
    img_vals, templ_vals = img_patch_f[mask], template_f[mask]
    img_norm = img_vals - img_vals.mean(axis=0)
    templ_norm = templ_vals - templ_vals.mean(axis=0)
    numerator = np.sum(img_norm * templ_norm)
    denominator = np.sqrt(np.sum(img_norm**2) * np.sum(templ_norm**2) + 1e-6)
    return numerator / denominator if denominator != 0 else -1.0

def run_mask_generation():

    if os.path.exists(MASK_OUTPUT_DIR):
        print(f"Mask output directory {MASK_OUTPUT_DIR} already exists")
        return
    os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)



    print(f"Loading SAM model on {DEVICE}...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)

    # Load all images
    image_files = natsorted([
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    images = [cv2.cvtColor(cv2.imread(os.path.join(IMAGE_FOLDER, f)), cv2.COLOR_BGR2RGB)
              for f in image_files]
    print(f"Loaded {len(images)} images.")

    first_img = images[0]
    h0, w0 = first_img.shape[:2]
    display_scale = min(1.0, DISPLAY_MAX_W / w0)
    disp_img = cv2.resize(first_img, None, fx=display_scale, fy=display_scale)

    clicked_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = int(x / display_scale), int(y / display_scale)
            clicked_points.append([ox, oy])
            print(f"Clicked at: ({ox}, {oy})")
            cv2.destroyAllWindows()

    cv2.imshow("Click the object to segment", cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback("Click the object to segment", click_event)
    cv2.waitKey(0)

    if not clicked_points:
        raise RuntimeError("No point clicked!")
    start_point = np.array([clicked_points[0]])
    start_label = np.array([1])

    predictor.set_image(first_img)
    masks, scores, _ = predictor.predict(
        point_coords=start_point,
        point_labels=start_label,
        multimask_output=True
    )

    print("Showing 3 masks, please choose the best one (0, 1, or 2).")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axs[i].imshow(first_img)
        axs[i].imshow(masks[i], alpha=0.5)
        axs[i].set_title(f"Mask {i} Score: {scores[i]:.3f}")
        axs[i].axis('off')
    plt.show()

    chosen_index = int(input("Enter mask index (0, 1, or 2): "))
    chosen_mask = masks[chosen_index].astype(np.uint8)

    prev_mask = chosen_mask
    # Save the initial mask
    save_path = os.path.join(MASK_OUTPUT_DIR, "0.jpg")
    cv2.imwrite(save_path, prev_mask * 255)
    prev_img = first_img

    for idx in tqdm(range(1, len(images))):
        img = images[idx]
        # print(f"\nProcessing image {idx+1}/{len(images)}: {image_files[idx]}")
        x1, y1, x2, y2 = mask_to_bbox(prev_mask)
        prev_patch = prev_img[y1:y2+1, x1:x2+1]
        prev_mask_crop = prev_mask[y1:y2+1, x1:x2+1].astype(bool)
        H, W = img.shape[:2]
        h, w = prev_patch.shape[:2]

        max_score = -2
        best_pos = (0, 0)
        # print("Running masked similarity search...")
        for y in range(0, H - h, 32):
            for x in range(0, W - w, 32):
                window = img[y:y+h, x:x+w]
                score = masked_ncc(window, prev_patch, prev_mask_crop)
                if score > max_score:
                    max_score, best_pos = score, (x, y)

        # print(f"Best similarity score: {max_score:.4f} at {best_pos}")
        cx, cy = best_pos[0] + w // 2, best_pos[1] + h // 2

        predictor.set_image(img)
        candidate_masks, _, _ = predictor.predict(
            point_coords=np.array([[cx, cy]]),
            point_labels=np.array([1]),
            multimask_output=True
        )

        ious = []
        for m in candidate_masks:
            candidate_mask_crop = m[best_pos[1]:best_pos[1]+h, best_pos[0]:best_pos[0]+w]
            prev_mask_resized = cv2.resize(
                prev_mask_crop.astype(np.uint8),
                (candidate_mask_crop.shape[1], candidate_mask_crop.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            ious.append(iou_mask(candidate_mask_crop, prev_mask_resized))

        best_mask_idx = int(np.argmax(ious))
        best_mask_iou = ious[best_mask_idx]
        # print(f"Best mask idx: {best_mask_idx}, IoU: {best_mask_iou:.4f}")
        prev_mask = candidate_masks[best_mask_idx].astype(np.uint8)
        prev_img = img

        save_path = os.path.join(MASK_OUTPUT_DIR, f"{idx}.jpg")
        cv2.imwrite(save_path, prev_mask * 255)
        # print(f"Saved to {save_path}")

    # print("\nDone generating masks.")
