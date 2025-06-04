import os
import numpy as np
import open3d as o3d
from PIL import Image
import re

# === CONFIGURATION ===
IMAGE_FOLDER = os.getenv("IMAGE_FOLDER")

MATCH_DIR = f"{IMAGE_FOLDER}_matches"
DEPTH_DIR = f"{IMAGE_FOLDER}_depths"
RGB_DIR = f"{IMAGE_FOLDER}_processed"
MASK_DIR = f"{IMAGE_FOLDER}_masks"

# Intrinsics
fx = 818.88865126988162
cx = 467.37764908767036
cy = 519.95480857428674

# RANSAC Parameters
RANSAC_ITERS = 10000
RANSAC_THRESH = 0.009
MIN_RANSAC_INLIERS = 7


def get_image_ids(rgb_dir):
    ids = []
    for file in sorted(os.listdir(rgb_dir)):
        if file.endswith(".jpg"):
            name = os.path.splitext(file)[0]
            if name.isdigit():
                ids.append(int(name))
    return sorted(ids)


def load_match_data(src, tgt):
    path = os.path.join(MATCH_DIR, f"matches_{src}_{tgt}.npy")
    data = np.load(path, allow_pickle=True).item()
    idx = np.argsort(data["scores"])
    return data["keypoints0"][idx], data["keypoints1"][idx]


def load_frame_data(frame_id):
    depth_path = os.path.join(DEPTH_DIR, f"depth_{frame_id}.npy")
    rgb_path = os.path.join(RGB_DIR, f"{frame_id}.jpg")
    mask_path = os.path.join(MASK_DIR, f"{frame_id}.jpg")

    depth = np.load(depth_path).astype(np.float32)
    rgb = np.array(Image.open(rgb_path))
    mask = np.array(Image.open(mask_path).convert("L"))

    return depth, rgb, mask


def project_to_3d(x, y, Z):
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fx
    return X, Y, Z


def project_matched_keypoints_to_3d(kpA, kpB, depthA, depthB):
    H, W = depthA.shape
    ptsA_3d, ptsB_3d = [], []
    for (xA, yA), (xB, yB) in zip(kpA, kpB):
        uA, vA = int(round(xA)), int(round(yA))
        uB, vB = int(round(xB)), int(round(yB))
        if 0 <= uA < W and 0 <= vA < H and 0 <= uB < W and 0 <= vB < H:
            zA, zB = depthA[vA, uA], depthB[vB, uB]
            if zA > 0 and zB > 0:
                ptA = project_to_3d(uA, vA, zA)
                ptB = project_to_3d(uB, vB, zB)
                ptsA_3d.append(ptA)
                ptsB_3d.append(ptB)
    return np.array(ptsA_3d), np.array(ptsB_3d)


def sim3(A, B):
    centroidA = A.mean(axis=0)
    centroidB = B.mean(axis=0)
    AA = A - centroidA
    BB = B - centroidB

    s = np.sqrt((BB ** 2).sum() / (AA ** 2).sum())
    U, _, Vt = np.linalg.svd(AA.T @ BB)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = centroidB - s * R @ centroidA
    return T


def ransac_sim3(ptsA, ptsB, inlier_thresh=0.01, max_iters=5000, min_inliers=3):
    best_inliers = np.zeros(ptsA.shape[0], dtype=bool)
    best_T = np.eye(4)
    n = ptsA.shape[0]
    if n < 3:
        return best_T, best_inliers
    indices = np.arange(n)
    for _ in range(max_iters):
        try:
            sample = np.random.choice(indices, 8, replace=False)
            T_candidate = sim3(ptsA[sample], ptsB[sample])
        except:
            continue
        A_hom = np.hstack([ptsA, np.ones((n, 1))])
        A_trans = (T_candidate @ A_hom.T).T[:, :3]
        dists = np.linalg.norm(A_trans - ptsB, axis=1)
        inliers = dists < inlier_thresh
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_T = T_candidate
    if best_inliers.sum() < min_inliers:
        print(" RANSAC found fewer inliers than min_inliers")
        return np.eye(4), np.zeros(n, dtype=bool)
    return best_T, best_inliers


def create_pointcloud(depth, rgb, mask):
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    X = (us - cx) * depth / fx
    Y = (vs - cy) * depth / fx
    Z = depth

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0
    valid = (Z.flatten() > 0) & (mask.flatten() > 128)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[valid])
    pcd.colors = o3d.utility.Vector3dVector(colors[valid])
    return pcd.voxel_down_sample(voxel_size=0.0033)


# === MAIN PIPELINE ===
def run_construction():
    image_ids = get_image_ids(RGB_DIR)
    transform_chain = {image_ids[0]: np.eye(4)}

    for i in range(len(image_ids) - 1):
        src, tgt = image_ids[i], image_ids[i + 1]
        print(f"\nAligning {src} -> {tgt}")

        try:
            kp0, kp1 = load_match_data(src, tgt)
        except FileNotFoundError:
            print(f" No match file for {src}->{tgt}")
            continue

        depth0, rgb0, mask0 = load_frame_data(src)
        depth1, rgb1, mask1 = load_frame_data(tgt)

        pts0, pts1 = project_matched_keypoints_to_3d(kp0, kp1, depth0, depth1)
        if len(pts0) < MIN_RANSAC_INLIERS:
            print(f" Not enough valid 3D matches between {src} and {tgt}")
            continue

        T_ransac, inliers = ransac_sim3(pts0, pts1, inlier_thresh=RANSAC_THRESH,
                                        max_iters=RANSAC_ITERS, min_inliers=MIN_RANSAC_INLIERS)

        if inliers.sum() >= MIN_RANSAC_INLIERS:
            T_sim3 = sim3(pts0[inliers], pts1[inliers])
        else:
            T_sim3 = T_ransac

        T_tgt_to_src = np.linalg.inv(T_sim3)
        T_tgt_global = transform_chain[src] @ T_tgt_to_src
        transform_chain[tgt] = T_tgt_global

    # === Merge Transformed Point Clouds ===
    merged = o3d.geometry.PointCloud()
    for frame_id in image_ids:
        depth, rgb, mask = load_frame_data(frame_id)
        pcd = create_pointcloud(depth, rgb, mask)
        pcd.transform(transform_chain[frame_id])
        merged += pcd

    o3d.io.write_point_cloud("merged_multiview_fullscene.ply", merged)
    o3d.visualization.draw_geometries([merged], window_name="Merged Scene")


# Run pipeline
if __name__ == "__main__":
    run_construction()
