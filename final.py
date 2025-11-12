#!/usr/bin/env python3
"""
Cuboid Rotation Estimator - Solution for Perception Assignment
This script estimates:
1. Normal angle and visible area of the largest face at each timestamp
2. Axis of rotation vector about which the box is rotating
Based on validated plane detection algorithm provided.
"""
import os
import sys
import sqlite3
import struct
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cv2
# CONFIGURATION
DB_PATH = "depth/depth.db3"
OUTPUT_DIR = "results"
# RANSAC parameters (from validated code)
MIN_INLIERS_FOR_PLANE = 100
RANSAC_ITERS = 2000
RANSAC_DIST_THRESH = 0.05 # 5 cm
MAX_PLANES = 6
CAMERA_FOV_DEG = 60.0
# Selection parameters
SCORE_WEIGHTS = (0.5, 0.2, 0.2, 0.1) # (pixel_area, solidity, center_proximity, normal_facing)
MIN_SELECTION_SCORE = 0.02
MIN_COMPONENT_PIXELS = 50
MAX_HULL_POINTS = 10000  # Increased for better hull accuracy
SAFE_SAMPLE_MAX = 50000  # Increased for better sampling
SEED = 42
np.random.seed(SEED)
# MESSAGE PARSING
def parse_image_message_bytes(data):
    """Parse sensor_msgs/Image from CDR bytes"""
    try:
        offset = 0
        if len(data) >= 4:
            hdr = struct.unpack_from('<I', data, 0)[0]
            if hdr in (0x00000000, 0x00000100):
                offset = 4
        sec = struct.unpack_from('<I', data, offset)[0]; offset += 4
        nanosec = struct.unpack_from('<I', data, offset)[0]; offset += 4
        frame_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
        frame_id = data[offset:offset+frame_len].decode('utf-8', errors='ignore').rstrip('\x00')
        offset += frame_len
        offset = (offset + 3) & ~3
        height = struct.unpack_from('<I', data, offset)[0]; offset += 4
        width = struct.unpack_from('<I', data, offset)[0]; offset += 4
        enc_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
        encoding = data[offset:offset+enc_len].decode('utf-8', errors='ignore').rstrip('\x00')
        offset += enc_len
        offset = (offset + 3) & ~3
        _is_big = struct.unpack_from('<B', data, offset)[0]; offset += 1
        offset = (offset + 3) & ~3
        _step = struct.unpack_from('<I', data, offset)[0]; offset += 4
        _data_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
        img_data = data[offset:]
        if '16UC1' in encoding or 'uint16' in encoding.lower():
            arr = np.frombuffer(img_data, dtype=np.uint16).astype(np.float32) / 1000.0
        elif '32FC1' in encoding or 'float32' in encoding.lower():
            arr = np.frombuffer(img_data, dtype=np.float32)
        else:
            arr = np.frombuffer(img_data, dtype=np.uint16).astype(np.float32) / 1000.0
        expected = height * width
        if arr.size < expected:
            arr = np.pad(arr, (0, expected - arr.size), constant_values=np.nan)
        elif arr.size > expected:
            arr = arr[:expected]
        arr = arr.reshape((height, width))
        arr[arr <= 0] = np.nan
        arr[np.isinf(arr)] = np.nan
       
        metadata = {'height': height, 'width': width, 'encoding': encoding,
                   'sec': sec, 'nanosec': nanosec, 'frame_id': frame_id}
        return arr.astype(np.float32), metadata
    except Exception as e:
        print(f"[parse_image_message_bytes] failed: {e}")
        return None, None
# POINT CLOUD CONVERSION
class PointCloudConverter:
    def __init__(self, width, height, fx=None, fy=None, cx=None, cy=None):
        self.width = width
        self.height = height
        if fx is None:
            self.fx = width / (2 * math.tan(math.radians(CAMERA_FOV_DEG / 2)))
        else:
            self.fx = fx
        if fy is None:
            self.fy = height / (2 * math.tan(math.radians(CAMERA_FOV_DEG / 2)))
        else:
            self.fy = fy
        self.cx = cx if cx is not None else width / 2.0
        self.cy = cy if cy is not None else height / 2.0
    def depth_to_points(self, depth_img):
        """Convert depth image to 3D point cloud"""
        h, w = depth_img.shape
        u = np.arange(w)
        v = np.arange(h)
        uu, vv = np.meshgrid(u, v)
        z = depth_img
        x = (uu - self.cx) * z / self.fx
        y = (vv - self.cy) * z / self.fy
        pts = np.stack([x, y, z], axis=-1)
        mask = ~np.isnan(z)
        points = pts[mask]
        return points, mask
# UTILITY FUNCTIONS
def safe_sample(pts, max_pts=SAFE_SAMPLE_MAX):
    """Sample points if too many"""
    if pts is None:
        return np.zeros((0, 3), dtype=np.float32)
    n = len(pts)
    if n <= max_pts:
        return pts
    idx = np.random.choice(n, max_pts, replace=False)
    return pts[idx]
def robust_depth_crop(points, z_clip_sigma=2.0, min_z=0.05):
    """Conservative outlier removal using median + IQR"""
    if points is None or len(points) == 0:
        return np.zeros(0, dtype=bool)
    z = points[:, 2]
    valid = np.isfinite(z) & (z > min_z)
    if np.sum(valid) < 10:
        return valid
    zv = z[valid]
    med = np.median(zv)
    q1 = np.percentile(zv, 25)
    q3 = np.percentile(zv, 75)
    iqr = max(1e-6, q3 - q1)
    low = max(min_z, med - z_clip_sigma * iqr)
    high = med + z_clip_sigma * iqr
    mask = (z >= low) & (z <= high)
    return mask
def project_points_to_image(pts3, width, height, fx, fy, cx, cy):
    """Project 3D points to 2D image coordinates"""
    if len(pts3) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    z = pts3[:, 2]
    mask = z > 0
    u = np.zeros(len(pts3))
    v = np.zeros(len(pts3))
    if np.any(mask):
        u[mask] = (pts3[mask, 0] * fx / pts3[mask, 2]) + cx
        v[mask] = (pts3[mask, 1] * fy / pts3[mask, 2]) + cy
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    return np.column_stack([u, v]), mask
def shoelace_area(pts2):
    """Calculate 2D polygon area using shoelace formula"""
    if pts2 is None or len(pts2) < 3:
        return 0.0
    x = pts2[:, 0]
    y = pts2[:, 1]
    if not (x[0] == x[-1] and y[0] == y[-1]):
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
# PLANE DETECTION (RANSAC)
def fit_plane_from_3pts(p):
    """Fit plane from 3 points"""
    v1 = p[1] - p[0]
    v2 = p[2] - p[0]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-8:
        return None, None
    n = n / norm
    d = -np.dot(n, p[0])
    return n, d
def ransac_planes(points, distance_threshold=RANSAC_DIST_THRESH,
                 num_iters=RANSAC_ITERS, max_planes=MAX_PLANES):
    """RANSAC plane detection with refinement"""
    remaining = points.copy()
    remaining_idx = np.arange(len(points))
    planes = []
    for plane_idx in range(max_planes):
        n_pts = len(remaining)
        if n_pts < MIN_INLIERS_FOR_PLANE:
            break
           
        best_inliers = None
        best_count = 0
        best_plane = None
       
        for it in range(num_iters):
            if n_pts < 3:
                break
            tri_idx = np.random.choice(n_pts, 3, replace=False)
            sample = remaining[tri_idx]
            normal, d = fit_plane_from_3pts(sample)
            if normal is None:
                continue
               
            dists = np.abs(remaining @ normal + d)
            inliers_mask = dists < distance_threshold
            count = int(np.sum(inliers_mask))
           
            if count > best_count:
                best_count = count
                best_inliers = inliers_mask.copy()
                best_plane = (normal.copy(), float(d))
               
            if best_count > 0.45 * n_pts:
                break
        if best_inliers is None or best_count < MIN_INLIERS_FOR_PLANE:
            break
        # Refine using covariance
        inlier_pts = remaining[best_inliers]
        centroid = np.mean(inlier_pts, axis=0)
        centered = inlier_pts - centroid
        cov = (centered.T @ centered) / max(1, (len(centered) - 1))
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        normal_refined = eigvecs[:, 2]
       
        # Ensure normal points towards camera
        if normal_refined[2] > 0:
            normal_refined = -normal_refined
           
        d_refined = -np.dot(normal_refined, centroid)
        dists_refined = np.abs(remaining @ normal_refined + d_refined)
        final_inliers = dists_refined < distance_threshold
        num_final = int(np.sum(final_inliers))
       
        if num_final < MIN_INLIERS_FOR_PLANE:
            break
        actual_indices = remaining_idx[final_inliers]
        planes.append({
            'normal': normal_refined,
            'd': d_refined,
            'inlier_mask_global': actual_indices,
            'inlier_pts': remaining[final_inliers],
            'num_inliers': num_final
        })
        keep_mask = ~final_inliers
        remaining = remaining[keep_mask]
        remaining_idx = remaining_idx[keep_mask]
    return planes
# CONTIGUOUS CLUSTER EXTRACTION
def largest_contiguous_inlier_cluster(inlier_pts3, image_width, image_height,
                                     fx, fy, cx, cy, min_component_pixels=MIN_COMPONENT_PIXELS):
    """Extract largest contiguous pixel cluster from 3D points"""
    if inlier_pts3 is None or len(inlier_pts3) == 0:
        return None, None, None
    z = inlier_pts3[:, 2]
    valid_idx = np.isfinite(z) & (z > 0)
    if not np.any(valid_idx):
        return None, None, None
    pts3 = inlier_pts3[valid_idx]
    # Project to image
    proj_uv, _ = project_points_to_image(pts3, image_width, image_height, fx, fy, cx, cy)
    ui = np.round(proj_uv[:, 0]).astype(np.int32)
    vi = np.round(proj_uv[:, 1]).astype(np.int32)
    ui = np.clip(ui, 0, image_width - 1)
    vi = np.clip(vi, 0, image_height - 1)
    # Create mask
    pix_coords = np.column_stack([vi, ui])
    uniq_pixels, inv = np.unique(pix_coords, axis=0, return_inverse=True)
    mask_img = np.zeros((image_height, image_width), dtype=np.uint8)
    mask_img[uniq_pixels[:, 0], uniq_pixels[:, 1]] = 255
    # Morphological operations - Increased kernel and iterations for better connectivity
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel, iterations=2)
    # Connected components
    num_labels, labels = cv2.connectedComponents(mask_img, connectivity=8)
   
    if num_labels <= 1:
        pixel_count = int(np.count_nonzero(mask_img))
        if pixel_count < min_component_pixels:
            return None, None, mask_img
        sel_mask = mask_img[vi, ui] > 0
        cluster_pts3 = pts3[sel_mask]
    else:
        max_label = 1
        max_count = 0
        for lbl in range(1, num_labels):
            cnt = int(np.sum(labels == lbl))
            if cnt > max_count:
                max_count = cnt
                max_label = lbl
        if max_count < min_component_pixels:
            return None, None, mask_img
        sel_mask = (labels[vi, ui] == max_label)
        cluster_pts3 = pts3[sel_mask]
    if cluster_pts3 is None or len(cluster_pts3) < 3:
        return None, None, mask_img
    # Compute hull in 3D plane coordinates
    pts_for_basis = safe_sample(cluster_pts3, max_pts=SAFE_SAMPLE_MAX)
    centroid = np.mean(pts_for_basis, axis=0).astype(np.float64)
    centered = (pts_for_basis - centroid).astype(np.float64)
    cov = (centered.T @ centered) / max(1, (len(centered) - 1))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    b1 = eigvecs[:, order[0]]
    b2 = eigvecs[:, order[1]]
    # Project all cluster points to 2D plane
    centered_full = (cluster_pts3 - centroid).astype(np.float64)
    pts2 = np.column_stack([centered_full @ b1, centered_full @ b2])
    # Compute hull
    if len(pts2) > MAX_HULL_POINTS:
        hull_pts2_candidates = safe_sample(pts2, max_pts=MAX_HULL_POINTS)
    else:
        hull_pts2_candidates = pts2
    try:
        ch = ConvexHull(hull_pts2_candidates, qhull_options='QJ Pp')  # Added options for robustness
        hull_pts2_ordered = hull_pts2_candidates[ch.vertices]
        hull_pts3 = centroid + np.outer(hull_pts2_ordered[:, 0], b1) + np.outer(hull_pts2_ordered[:, 1], b2)
    except Exception:
        mins = np.min(pts2, axis=0)
        maxs = np.max(pts2, axis=0)
        rect2 = np.array([[mins[0], mins[1]], [maxs[0], mins[1]],
                         [maxs[0], maxs[1]], [mins[0], maxs[1]]])
        hull_pts3 = centroid + np.outer(rect2[:, 0], b1) + np.outer(rect2[:, 1], b2)
    return cluster_pts3, hull_pts3, mask_img
# AREA COMPUTATION
def compute_3d_area_from_hull(hull_pts3):
    """Compute actual 3D surface area from convex hull vertices"""
    if hull_pts3 is None or len(hull_pts3) < 3:
        return 0.0
   
    # Fit plane to hull points
    centroid = np.mean(hull_pts3, axis=0)
    centered = hull_pts3 - centroid
    cov = (centered.T @ centered) / max(1, len(centered) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    b1 = eigvecs[:, order[0]]
    b2 = eigvecs[:, order[1]]
   
    # Project to 2D
    pts2 = np.column_stack([centered @ b1, centered @ b2])
   
    # Use shoelace for ordered hull vertices
    return shoelace_area(pts2)
# PLANE SCORING AND SELECTION
def score_and_pick_best_plane(usable_planes, image_width, image_height,
                              fx, fy, cx, cy, weights=SCORE_WEIGHTS,
                              min_score=MIN_SELECTION_SCORE):
    """Score planes and select best one (largest visible face)"""
    if not usable_planes:
        return None
    img_cx = image_width / 2.0
    img_cy = image_height / 2.0
    raw_pixel_areas = []
    raw_hull_pixel_areas = []
    raw_centroid_dists = []
    raw_normal_facing = []
    for p in usable_planes:
        cluster = p.get('cluster_pts3')
        if cluster is None or len(cluster) == 0:
            cluster = p.get('inlier_pts')
            if cluster is None or len(cluster) == 0:
                raw_pixel_areas.append(0)
                raw_hull_pixel_areas.append(1e-6)
                raw_centroid_dists.append(math.hypot(img_cx, img_cy))
                raw_normal_facing.append(0.0)
                continue
        pts2d, _ = project_points_to_image(cluster, image_width, image_height, fx, fy, cx, cy)
        pix = np.round(pts2d).astype(int)
        pix[:, 0] = np.clip(pix[:, 0], 0, image_width - 1)
        pix[:, 1] = np.clip(pix[:, 1], 0, image_height - 1)
        uniq = np.unique(pix, axis=0)
        pixel_area = int(len(uniq))
        hull_pix_area = 1e-6
        if p.get('cluster_hull3') is not None:
            hull_uv, _ = project_points_to_image(np.asarray(p['cluster_hull3']),
                                                image_width, image_height, fx, fy, cx, cy)
            try:
                hull_pix_area = float(shoelace_area(hull_uv))
            except Exception:
                hull_pix_area = max(1.0, pixel_area)
        else:
            if len(uniq) >= 3:
                try:
                    ch = ConvexHull(uniq)
                    hull_pts = uniq[ch.vertices]
                    hull_pix_area = float(shoelace_area(hull_pts))
                except Exception:
                    hull_pix_area = max(1.0, pixel_area)
        centroid_uv = np.mean(uniq, axis=0) if len(uniq) > 0 else np.array([img_cx, img_cy])
        dist = float(math.hypot(centroid_uv[0] - img_cx, centroid_uv[1] - img_cy))
        n = np.array(p['normal']).astype(float)
        if n[2] > 0:
            n = -n
        normal_facing = max(0.0, -n[2])
        raw_pixel_areas.append(pixel_area)
        raw_hull_pixel_areas.append(hull_pix_area)
        raw_centroid_dists.append(dist)
        raw_normal_facing.append(normal_facing)
    pa = np.array(raw_pixel_areas, dtype=float)
    ha = np.array(raw_hull_pixel_areas, dtype=float)
    cd = np.array(raw_centroid_dists, dtype=float)
    nf = np.array(raw_normal_facing, dtype=float)
    def normalize(x):
        if np.ptp(x) < 1e-6:
            return np.ones_like(x) * 0.5
        return (x - np.min(x)) / np.ptp(x)
    n_pa = normalize(pa)
    solidity = pa / (ha + 1e-9)
    n_sol = normalize(solidity)
    n_cd = 1.0 - normalize(cd)
    n_nf = normalize(nf)
    w1, w2, w3, w4 = weights
    scores = []
   
    for i, p in enumerate(usable_planes):
        score = (w1 * n_pa[i]) + (w2 * n_sol[i]) + (w3 * n_cd[i]) + (w4 * n_nf[i])
        scores.append(score)
        p['_selection_score'] = float(score)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
   
    if best_score < min_score:
        return None
       
    return usable_planes[best_idx]
# ROTATION AXIS ESTIMATION
def estimate_rotation_axis_from_normals(normals):
    """Estimate rotation axis from face normals across frames"""
    normals = np.array(normals)
   
    # Ensure normals point towards camera
    for i in range(len(normals)):
        if normals[i][2] > 0:
            normals[i] = -normals[i]
   
    # Compute differences between consecutive normals
    diffs = []
    for i in range(len(normals) - 1):
        d = normals[i + 1] - normals[i]
        nrm = np.linalg.norm(d)
        if nrm > 1e-6:
            diffs.append(d / nrm)
   
    if len(diffs) < 2:
        return None
   
    # SVD to find principal direction of change
    M = np.array(diffs)
    _, _, vh = np.linalg.svd(M, full_matrices=False)
    rotation_axis = vh[-1]
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
   
    # Cross product method for validation
    crosses = []
    for i in range(len(normals) - 1):
        c = np.cross(normals[i], normals[i + 1])
        nrm = np.linalg.norm(c)
        if nrm > 1e-6:
            crosses.append(c / nrm)
   
    if crosses:
        avg_cross = np.mean(crosses, axis=0)
        avg_cross = avg_cross / np.linalg.norm(avg_cross)
        # Use cross product if more consistent
        if abs(np.dot(rotation_axis, avg_cross)) < 0.5:
            rotation_axis = avg_cross
   
    return rotation_axis
# MAIN PROCESSING
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
   

   
    # Load depth images from database
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")
   
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
   
    cur.execute("""
        SELECT m.timestamp, m.data, t.name, t.type
        FROM messages m
        JOIN topics t ON m.topic_id = t.id
        WHERE t.name = '/depth'
        ORDER BY m.timestamp
    """)
    rows = cur.fetchall()
    conn.close()
   
    depth_images = []
    timestamps = []
    metadata = None
   
    print(f"\nLoading {len(rows)} depth images...")
    for i, (timestamp, data, name, ttype) in enumerate(rows):
        img, meta = parse_image_message_bytes(data)
        if img is not None:
            depth_images.append(img)
            timestamps.append(timestamp)
            if metadata is None:
                metadata = meta
            print(f" Frame {i+1}: {img.shape}, depth range: {np.nanmin(img):.3f}-{np.nanmax(img):.3f}m")
   
    if not depth_images:
        raise RuntimeError("No valid depth images found")
   
    # Initialize point cloud converter
    H = metadata['height']
    W = metadata['width']
    pc_conv = PointCloudConverter(width=W, height=H)
    fx, fy, cx, cy = pc_conv.fx, pc_conv.fy, pc_conv.cx, pc_conv.cy
   
    print(f"\nCamera parameters:")
    print(f" Resolution: {W}x{H}")
    print(f" Focal length: fx={fx:.2f}, fy={fy:.2f}")
    print(f" Principal point: cx={cx:.2f}, cy={cy:.2f}")
   
    # Process each frame
    results_table = []
    all_normals = []
      
    for idx, depth_img in enumerate(depth_images):
        print(f"\n--- Frame {idx+1}/{len(depth_images)} ---")
       
        # Convert to point cloud
        points, valid_mask = pc_conv.depth_to_points(depth_img)
        print(f" Point cloud: {len(points)} points")
       
        if len(points) < 50:
            print(" ERROR: Too few points, skipping")
            results_table.append({
                'image_number': idx + 1,
                'normal_angle_deg': None,
                'visible_area_m2': None,
                'error': 'Insufficient points'
            })
            all_normals.append(None)
            continue
       
        # Crop outliers
        crop_mask = robust_depth_crop(points, z_clip_sigma=2.0)
        cropped_pts = points[crop_mask]
        print(f" After outlier removal: {len(cropped_pts)} points")
       
        if len(cropped_pts) < 50:
            cropped_pts = points.copy()
       
        # Detect planes using RANSAC
        planes = ransac_planes(cropped_pts, distance_threshold=RANSAC_DIST_THRESH,
                              num_iters=RANSAC_ITERS, max_planes=MAX_PLANES)
        print(f" Detected {len(planes)} planes")
       
        # Process each plane
        usable_planes = []
        for pi, p in enumerate(planes):
            # Extract contiguous cluster
            cluster_pts3, cluster_hull3, mask_img = largest_contiguous_inlier_cluster(
                p['inlier_pts'], W, H, fx, fy, cx, cy,
                min_component_pixels=MIN_COMPONENT_PIXELS
            )
           
            p['cluster_pts3'] = cluster_pts3
            p['cluster_hull3'] = cluster_hull3
           
            # Compute 3D area
            if cluster_hull3 is not None:
                area_3d = compute_3d_area_from_hull(cluster_hull3)
                p['area_m2'] = float(area_3d)
            else:
                p['area_m2'] = 0.0
           
            # Compute normal angle
            n = p['normal']
            if n[2] > 0:
                n = -n
            camera_normal = np.array([0, 0, -1])
            cos_angle = np.clip(np.dot(n, camera_normal), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            p['normal_angle_deg'] = angle_deg
           
            print(f" Plane {pi+1}: {p['num_inliers']} inliers, "
                  f"area={p['area_m2']:.4f}m², angle={angle_deg:.1f}°")
           
            usable_planes.append(p)
       
        # Select best plane (largest visible face)
        selected = score_and_pick_best_plane(usable_planes, W, H, fx, fy, cx, cy,
                                            weights=SCORE_WEIGHTS,
                                            min_score=MIN_SELECTION_SCORE)
       
        if selected is None:
            # Fallback to largest area
            if usable_planes:
                selected = max(usable_planes, key=lambda x: x.get('area_m2', 0.0))
                print(" Using fallback selection (largest area)")
            else:
                print(" ERROR: No valid planes found")
                results_table.append({
                    'image_number': idx + 1,
                    'normal_angle_deg': None,
                    'visible_area_m2': None,
                    'error': 'No valid planes'
                })
                all_normals.append(None)
                continue
       
        # Extract results
        normal_angle = selected['normal_angle_deg']
        visible_area = selected['area_m2']
       
        print(f"\n SELECTED FACE:")
        print(f" Normal angle: {normal_angle:.2f}°")
        print(f" Visible area: {visible_area:.6f} m²")
       
        results_table.append({
            'image_number': idx + 1,
            'normal_angle_deg': round(normal_angle, 2),
            'visible_area_m2': round(visible_area, 6)
        })
        all_normals.append(selected['normal'])
       
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
       
        # Left: depth image with selected face hull
        ax1 = axes[0]
        im1 = ax1.imshow(depth_img, cmap='viridis')
        ax1.set_title(f'Frame {idx+1} - Selected Face')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
       
        if selected.get('cluster_hull3') is not None:
            hull3 = np.asarray(selected['cluster_hull3'])
            imgcoords, _ = project_points_to_image(hull3, W, H, fx, fy, cx, cy)
            poly = np.vstack([imgcoords, imgcoords[0]])
            ax1.plot(poly[:, 0], poly[:, 1], '-r', linewidth=2, label='Selected face')
            ax1.scatter(poly[:, 0], poly[:, 1], c='r', s=30)
            ax1.legend()
       
        ax1.text(10, 30, f"Angle: {normal_angle:.1f}°\nArea: {visible_area:.4f} m²",
                color='white', fontsize=11, bbox=dict(facecolor='black', alpha=0.7))
       
        # Right: all detected planes
        ax2 = axes[1]
        ax2.imshow(depth_img, cmap='gray', alpha=0.3)
        ax2.set_title(f'Frame {idx+1} - All Detected Planes')
       
        colors = plt.cm.tab10(np.linspace(0, 1, len(usable_planes)))
        for pi, p in enumerate(usable_planes):
            if p.get('cluster_pts3') is not None:
                proj2, _ = project_points_to_image(p['cluster_pts3'], W, H, fx, fy, cx, cy)
                is_selected = (p is selected)
                alpha = 0.9 if is_selected else 0.4
                size = 2 if is_selected else 1
                ax2.scatter(proj2[:, 0], proj2[:, 1], s=size,
                          c=[colors[pi]], alpha=alpha,
                          label=f"Plane {pi+1}" + (" (selected)" if is_selected else ""))
        ax2.legend(fontsize=8)
       
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'frame_{idx+1}_analysis.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
   
    # ========================================================================
    # ESTIMATE ROTATION AXIS
    # ========================================================================
    print("\n" + "="*70)
    print("ROTATION AXIS ESTIMATION")
    print("="*70)
   
    valid_normals = [n for n in all_normals if n is not None]
    print(f"\nUsing {len(valid_normals)}/{len(all_normals)} valid normals")
   
    if len(valid_normals) < 3:
        print("ERROR: Insufficient valid normals to estimate rotation axis")
        rotation_axis = None
    else:
        rotation_axis = estimate_rotation_axis_from_normals(valid_normals)
        if rotation_axis is not None:
            print(f"\nEstimated rotation axis (camera frame):")
            print(f" x: {rotation_axis[0]:.6f}")
            print(f" y: {rotation_axis[1]:.6f}")
            print(f" z: {rotation_axis[2]:.6f}")
            print(f" Magnitude: {np.linalg.norm(rotation_axis):.6f}")
   
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
   
    # Save table
    table_file = os.path.join(OUTPUT_DIR, 'results_table.txt')
    with open(table_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CUBOID FACE ANALYSIS RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Image':<10} {'Normal Angle (°)':<20} {'Visible Area (m²)':<20}\n")
        f.write("-"*70 + "\n")
        for r in results_table:
            img_num = r['image_number']
            angle = f"{r['normal_angle_deg']:.2f}" if r['normal_angle_deg'] is not None else "N/A"
            area = f"{r['visible_area_m2']:.6f}" if r['visible_area_m2'] is not None else "N/A"
            f.write(f"{img_num:<10} {angle:<20} {area:<20}\n")
   
    # Save rotation axis
    axis_file = os.path.join(OUTPUT_DIR, 'rotation_axis.txt')
    with open(axis_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ROTATION AXIS VECTOR (Camera Frame)\n")
        f.write("="*70 + "\n\n")
        if rotation_axis is not None:
            f.write(f"x: {rotation_axis[0]:.6f}\n")
            f.write(f"y: {rotation_axis[1]:.6f}\n")
            f.write(f"z: {rotation_axis[2]:.6f}\n")
            f.write(f"\nMagnitude: {np.linalg.norm(rotation_axis):.6f}\n")
        else:
            f.write("ERROR: Could not estimate rotation axis\n")
   
    # Save JSON summary
    summary = {
        'camera_parameters': {
            'width': W,
            'height': H,
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy),
            'fov_deg': CAMERA_FOV_DEG
        },
        'processing_parameters': {
            'ransac_iterations': RANSAC_ITERS,
            'distance_threshold_m': RANSAC_DIST_THRESH,
            'min_inliers': MIN_INLIERS_FOR_PLANE,
            'max_planes': MAX_PLANES
        },
        'results': results_table,
        'rotation_axis': {
            'x': float(rotation_axis[0]) if rotation_axis is not None else None,
            'y': float(rotation_axis[1]) if rotation_axis is not None else None,
            'z': float(rotation_axis[2]) if rotation_axis is not None else None
        } if rotation_axis is not None else None
    }
   
    json_file = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
   
    # Print summary table
    print(f"\n{'Image':<10} {'Normal Angle (°)':<20} {'Visible Area (m²)':<20}")
    for r in results_table:
        img_num = r['image_number']
        angle = f"{r['normal_angle_deg']:.2f}" if r['normal_angle_deg'] is not None else "N/A"
        area = f"{r['visible_area_m2']:.6f}" if r['visible_area_m2'] is not None else "N/A"
        print(f"{img_num:<10} {angle:<20} {area:<20}")
   
    if rotation_axis is not None:
        print(f"\nRotation Axis: [{rotation_axis[0]:.6f}, {rotation_axis[1]:.6f}, {rotation_axis[2]:.6f}]")
   
if __name__ == "__main__":
    main()