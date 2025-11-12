# Take Home Assigment
---

## 1. Depth Data Extraction  
- Depth images are decoded from ROS 2 `sensor_msgs/Image` byte messages in the `.db3` file.  
- Each image is converted into a floating-point depth map (in meters).  
- Metadata such as resolution, encoding, and timestamps are parsed for each frame.

---

## 2. Point Cloud Generation  
Using pinhole camera geometry, each depth map is projected into 3D coordinates.  
Focal lengths are computed from the horizontal field of view (60°):  

$$ f_x = \frac{W}{2 \tan(\text{FOV}/2)}, \quad f_y = \frac{H}{2 \tan(\text{FOV}/2)} $$

Each pixel \((u, v)\) is mapped to 3D space as:  

$$ x = (u - c_x) \frac{z}{f_x}, \quad y = (v - c_y) \frac{z}{f_y} $$

---

## 3. Outlier Removal  
- Conservative z-clipping based on **median** and **IQR** removes depth outliers and sensor noise.  
- Ensures stable and robust plane fitting.

---

## 4. Plane Detection (RANSAC)  
- Iterative RANSAC is applied to extract planar surfaces from 3D points.  
- Each plane is represented as:  

$$ n \cdot x + d = 0 $$

where **n** is the unit normal vector.  
- Plane normals are refined using **PCA** on inliers.  
- Up to **6 dominant planes** are detected per frame.

---

## 5. Contiguous Face Extraction  
For each detected plane:  
- Inlier points are projected back to 2D image space.  
- **Morphological close–open operations** and **connected components** isolate the largest contiguous region.  
- The **Convex Hull** (`scipy.spatial.ConvexHull`) defines the face boundary for area estimation.

---

## 6. Visible Area Computation  
- The 3D area of the planar hull is computed by projecting points onto two principal basis vectors \((b_1, b_2)\).  
- The **shoelace formula** is then applied on the 2D projection:  

$$ A = \frac{1}{2} \left| \sum_{i=1}^{n} (x_i y_{i+1} - y_i x_{i+1}) \right| $$

---

## 7. Plane Scoring and Selection  
Each detected plane is scored using a weighted composite metric:  
- Pixel area  
- Hull solidity  
- Center proximity  
- Normal facing the camera  

The plane with the highest score is selected as the **dominant visible face**.

---

## 8. Normal Angle Estimation  
The **normal angle** is computed between the plane normal and the camera optical axis:  

$$ \theta = \cos^{-1}(-n_z) $$

---

## 9. Rotation Axis Estimation  
- Plane normals across frames are aligned toward the camera.  
- Differences between consecutive normals are analyzed using **SVD** to extract the **dominant rotation axis**.  
- A **cross-product validation** ensures consistency across frames.  
- The result is a normalized **3D rotation vector** in camera coordinates.

---

## Output  

### Per-Frame Results  

| Image | Normal Angle (degrees) | Visible Area (m²) |
|-------|------------------------|-------------------|
| 1     | 57.78                  | 4.189572          |
| 2     | 13.03                  | 1.008394          |
| 3     | 28.45                  | 1.273574          |
| 4     | 47.40                  | 1.665369          |
| 5     | 24.86                  | 0.859226          |
| 6     | 37.09                  | 1.236707          |
| 7     | 42.31                  | 0.727463          |

---

### Files Saved  
- `results_table.txt`  
- `rotation_axis.txt`  

All image results saved in: `results/`  
