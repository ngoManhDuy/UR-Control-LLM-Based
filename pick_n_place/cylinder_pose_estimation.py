import numpy as np
import cv2
import torch
import sys
import os

# Fix PyTorch 2.6+ compatibility issue with YOLO model loading
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import YOLO with proper PyTorch compatibility
try:
    # For PyTorch 2.6+, we need to handle the weights_only security feature
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    
    # Add safe globals for PyTorch 2.6+
    torch.serialization.add_safe_globals([DetectionModel])
    
except ImportError as e:
    print(f"Error importing ultralytics: {e}")
    raise

from camera import RealSenseCamera
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CylinderPoseEstimator:
    def __init__(self, model_path='best.pt'):
        """
        Initialize the cylinder pose estimator
        
        Args:
            model_path (str): Path to the YOLO model weights
        """
        # Initialize YOLO model with PyTorch 2.6+ compatibility
        try:
            # First, try standard loading
            self.model = YOLO(model_path)
            print(f"✓ YOLO model loaded successfully from {model_path}")
        except Exception as e:
            if "weights_only" in str(e) or "WeightsUnpickler" in str(e) or "__module__" in str(e):
                print(f"PyTorch security issue detected. Attempting alternative loading method...")
                
                # Try with weights_only=False by temporarily patching torch.load
                original_torch_load = torch.load
                
                def patched_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
                    return original_torch_load(f, map_location=map_location, 
                                             pickle_module=pickle_module, 
                                             weights_only=False, **kwargs)
                
                # Temporarily replace torch.load
                torch.load = patched_load
                
                try:
                    self.model = YOLO(model_path)
                    print(f"✓ YOLO model loaded successfully from {model_path} (with weights_only=False)")
                finally:
                    # Restore original torch.load
                    torch.load = original_torch_load
            else:
                print(f"Error loading YOLO model: {e}")
                print("Make sure the model file 'best.pt' exists in the current directory")
                raise
        
        # Initialize camera
        self.camera = RealSenseCamera()
        
        # Cylinder parameters
        self.cylinder_radius = 0.006  # 6mm as corrected by user
        self.cylinder_height = 0.01   # 10mm as specified
        
    def connect(self):
        """Connect to the camera"""
        return self.camera.connect()
    
    def disconnect(self):
        """Disconnect from the camera"""
        self.camera.disconnect()
    
    def detect_cylinder_bbox(self, confidence_threshold=0.5):
        """
        Detect cylinder using YOLO and return bounding box
        
        Args:
            confidence_threshold (float): Minimum confidence score for detection
            
        Returns:
            tuple: (success, bbox, confidence, color_image, depth_image)
                - success: Boolean indicating if detection was successful
                - bbox: Bounding box [x_min, y_min, x_max, y_max]
                - confidence: Detection confidence score
                - color_image: RGB image
                - depth_image: Depth image in mm
        """
        # Get frames from RealSense
        color_image, depth_image = self.camera.get_frames()
        if color_image is None or depth_image is None:
            return False, None, None, None, None
        
        # Run YOLO detection
        results = self.model(color_image)[0]
        
        # Process results to find best detection
        best_detection = None
        best_confidence = 0
        
        for detection in results.boxes.data:
            # detection format: [x1, y1, x2, y2, confidence, class_id]
            if detection[5] == 0:  # class_id 0 is "product"
                confidence = float(detection[4])
                if confidence > confidence_threshold and confidence > best_confidence:
                    best_detection = detection
                    best_confidence = confidence
        
        if best_detection is None:
            return False, None, None, color_image, depth_image
        
        # Get bounding box [x_min, y_min, x_max, y_max]
        bbox = best_detection[:4].cpu().numpy()
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        
        return True, bbox, best_confidence, color_image, depth_image
    
    def extract_point_cloud_from_bbox(self, bbox, depth_image, downsample_factor=1):
        """
        Extract 3D point cloud from bounding box region
        
        Args:
            bbox: Bounding box [x_min, y_min, x_max, y_max]
            depth_image: Depth image in mm from RealSense
            downsample_factor (int): Factor to downsample points (1 = no downsampling)
            
        Returns:
            numpy.ndarray: Point cloud as Nx3 array [X, Y, Z] in meters
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Get camera intrinsics
        intrinsics = self.camera.intrinsics
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy
        
        # Initialize list to store 3D points
        points_3d = []
        
        # Iterate through pixels in bounding box
        for v in range(y_min, y_max, downsample_factor):  # v is row (y-coordinate)
            for u in range(x_min, x_max, downsample_factor):  # u is column (x-coordinate)
                # Get depth value at this pixel
                depth_mm = depth_image[v, u]
                
                # Skip invalid depth values
                if depth_mm <= 0:
                    continue
                
                # Convert depth from mm to meters
                Z = depth_mm / 1000.0
                
                # Back-project to 3D using pinhole camera model
                # Following the equations from your PDF:
                # X = (u - cx) * Z / fx
                # Y = (v - cy) * Z / fy
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                
                # Add point to list
                points_3d.append([X, Y, Z])
        
        # Convert to numpy array
        if len(points_3d) == 0:
            return np.array([]).reshape(0, 3)
        
        point_cloud = np.array(points_3d)
        
        print(f"Extracted {len(point_cloud)} points from bounding box")
        print(f"Point cloud bounds:")
        print(f"  X: [{point_cloud[:, 0].min():.4f}, {point_cloud[:, 0].max():.4f}]")
        print(f"  Y: [{point_cloud[:, 1].min():.4f}, {point_cloud[:, 1].max():.4f}]")
        print(f"  Z: [{point_cloud[:, 2].min():.4f}, {point_cloud[:, 2].max():.4f}]")
        
        return point_cloud
    
    def visualize_detection_and_pointcloud(self, bbox, color_image, point_cloud):
        """
        Visualize the detection and extracted point cloud
        
        Args:
            bbox: Bounding box [x_min, y_min, x_max, y_max]
            color_image: RGB image
            point_cloud: Nx3 point cloud array
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Original detection
        ax1 = fig.add_subplot(131)
        ax1.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               fill=False, color='red', linewidth=2)
            ax1.add_patch(rect)
        ax1.set_title('YOLO Detection')
        ax1.axis('off')
        
        # Plot 2: Point cloud - Top view (X-Y)
        ax2 = fig.add_subplot(132)
        if len(point_cloud) > 0:
            ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], 
                       c=point_cloud[:, 2], cmap='viridis', s=1)
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Point Cloud - Top View')
            ax2.axis('equal')
            plt.colorbar(ax2.collections[0], ax=ax2, label='Z (m)')
        
        # Plot 3: Point cloud - 3D view
        ax3 = fig.add_subplot(133, projection='3d')
        if len(point_cloud) > 0:
            ax3.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                       c=point_cloud[:, 2], cmap='viridis', s=1)
            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Y (m)')
            ax3.set_zlabel('Z (m)')
            ax3.set_title('Point Cloud - 3D View')
        
        plt.tight_layout()
        plt.show()

    def get_cylinder_point_cloud(self, confidence_threshold=0.5, downsample_factor=1, visualize=False):
        """
        Complete pipeline: Detect cylinder and extract point cloud
        
        Args:
            confidence_threshold (float): Minimum confidence for YOLO detection
            downsample_factor (int): Factor to downsample point cloud
            visualize (bool): Whether to show visualization
            
        Returns:
            tuple: (success, point_cloud, bbox, confidence)
                - success: Boolean indicating if extraction was successful
                - point_cloud: Nx3 numpy array of 3D points
                - bbox: Bounding box used for extraction
                - confidence: YOLO detection confidence
        """
        # Step 1: Detect cylinder with YOLO
        success, bbox, confidence, color_image, depth_image = self.detect_cylinder_bbox(confidence_threshold)
        
        if not success:
            print("No cylinder detected")
            return False, None, None, None
        
        print(f"Cylinder detected with confidence: {confidence:.3f}")
        print(f"Bounding box: {bbox}")
        
        # Step 2: Extract point cloud from bounding box
        point_cloud = self.extract_point_cloud_from_bbox(bbox, depth_image, downsample_factor)
        
        if len(point_cloud) == 0:
            print("No valid points extracted from bounding box")
            return False, None, bbox, confidence
        
        # Step 3: Visualize if requested
        if visualize:
            self.visualize_detection_and_pointcloud(bbox, color_image, point_cloud)
        
        return True, point_cloud, bbox, confidence

    # ===== UR3 OPTIMIZED METHODS =====
    
    def fit_circle_from_three_points_2d(self, p1, p2, p3):
        """
        Fit a circle from three 2D points using circumcenter calculation
        Optimized for UR3 top-down view setup
        
        Args:
            p1, p2, p3: Three 2D points as numpy arrays [x, y]
            
        Returns:
            tuple: (center_x, center_y, radius) or (None, None, None) if invalid
        """
        # Calculate circumcenter of triangle formed by three points
        ax, ay = p1[0], p1[1]
        bx, by = p2[0], p2[1]
        cx, cy = p3[0], p3[1]
        
        # Calculate determinant
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        
        # Check if points are collinear
        if abs(d) < 1e-10:
            return None, None, None
        
        # Calculate circumcenter coordinates
        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
        
        # Calculate radius
        radius = np.sqrt((ax - ux)**2 + (ay - uy)**2)
        
        # Verify all three points are equidistant (sanity check)
        r1 = np.sqrt((ax - ux)**2 + (ay - uy)**2)
        r2 = np.sqrt((bx - ux)**2 + (by - uy)**2)
        r3 = np.sqrt((cx - ux)**2 + (cy - uy)**2)
        
        max_error = max(abs(r1-r2), abs(r2-r3), abs(r1-r3))
        if max_error > 0.001:  # 1mm tolerance
            return None, None, None
        
        return ux, uy, radius
    
    def point_to_circle_distance_2d(self, point, center_x, center_y, radius):
        """
        Compute distance from a 2D point to a circle
        
        Args:
            point: 2D point as numpy array [x, y]
            center_x, center_y: Circle center coordinates
            radius: Circle radius
            
        Returns:
            float: Distance from point to circle circumference
        """
        distance_to_center = np.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
        return abs(distance_to_center - radius)
    
    def count_circle_inliers_2d(self, points_2d, center_x, center_y, radius, epsilon=0.0005):
        """
        Count inliers for a 2D circle hypothesis
        Optimized for UR3 setup with tight tolerances
        
        Args:
            points_2d: Nx2 numpy array of 2D points
            center_x, center_y: Circle center coordinates  
            radius: Circle radius
            epsilon: Inlier threshold in meters (default 0.5mm)
            
        Returns:
            tuple: (inlier_count, inlier_mask)
        """
        inlier_mask = np.zeros(len(points_2d), dtype=bool)
        
        for i, point in enumerate(points_2d):
            distance = self.point_to_circle_distance_2d(point, center_x, center_y, radius)
            if distance <= epsilon:
                inlier_mask[i] = True
        
        return np.sum(inlier_mask), inlier_mask
    
    def ransac_circle_fitting_2d(self, points_2d, max_iterations=500, epsilon=0.0005, min_inliers=30):
        """
        RANSAC 2D circle fitting optimized for UR3 constrained setup
        
        Args:
            points_2d: Nx2 numpy array of 2D points [x, y]
            max_iterations: Maximum RANSAC iterations
            epsilon: Inlier threshold in meters (0.5mm default)
            min_inliers: Minimum number of inliers required
            
        Returns:
            tuple: (success, center_x, center_y, radius, inlier_mask)
        """
        if len(points_2d) < 3:
            print("Need at least 3 points for 2D circle fitting")
            return False, None, None, None, None
        
        best_inlier_count = 0
        best_center_x = None
        best_center_y = None
        best_radius = None
        best_inlier_mask = None
        
        print(f"Starting 2D circle RANSAC with {len(points_2d)} points")
        print(f"Expected radius: {self.cylinder_radius:.4f}m, tolerance: {epsilon:.4f}m")
        
        # Track radius statistics for debugging
        all_radii = []
        good_radii = []
        
        for iteration in range(max_iterations):
            # Step 1: Randomly select 3 points
            indices = np.random.choice(len(points_2d), 3, replace=False)
            p1, p2, p3 = points_2d[indices[0]], points_2d[indices[1]], points_2d[indices[2]]
            
            # Step 2: Fit circle from three points
            center_x, center_y, radius = self.fit_circle_from_three_points_2d(p1, p2, p3)
            
            if center_x is None:
                continue  # Points were collinear or invalid
            
            # Track all radii
            all_radii.append(radius)
            
            # Filter out obviously bad radii (should be close to expected)
            radius_error = abs(radius - self.cylinder_radius) / self.cylinder_radius
            if radius_error > 0.5:  # Reject if more than 50% off
                continue
            
            good_radii.append(radius)
            
            # Step 3: Count inliers
            inlier_count, inlier_mask = self.count_circle_inliers_2d(
                points_2d, center_x, center_y, radius, epsilon
            )
            
            # Step 4: Update best model if this is better
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_center_x = center_x
                best_center_y = center_y
                best_radius = radius
                best_inlier_mask = inlier_mask.copy()
                
                print(f"Iteration {iteration}: New best with {inlier_count} inliers, radius={radius:.4f}m")
        
        # Print radius statistics
        if len(all_radii) > 0:
            print(f"\n2D Circle Radius Statistics:")
            print(f"  All radii: min={min(all_radii):.4f}m, max={max(all_radii):.4f}m, mean={np.mean(all_radii):.4f}m")
            if len(good_radii) > 0:
                print(f"  Good radii: min={min(good_radii):.4f}m, max={max(good_radii):.4f}m, mean={np.mean(good_radii):.4f}m")
                print(f"  Good radius attempts: {len(good_radii)}/{len(all_radii)} ({len(good_radii)/len(all_radii)*100:.1f}%)")
        
        # Check if we found a good solution
        success = best_inlier_count >= min_inliers
        
        if success:
            inlier_percentage = (best_inlier_count / len(points_2d)) * 100
            print(f"✓ 2D Circle RANSAC succeeded: {best_inlier_count}/{len(points_2d)} inliers ({inlier_percentage:.1f}%)")
            print(f"  Final circle: center=({best_center_x:.4f}, {best_center_y:.4f}), radius={best_radius:.4f}m")
        else:
            print(f"✗ 2D Circle RANSAC failed: only {best_inlier_count} inliers (need {min_inliers})")
        
        return success, best_center_x, best_center_y, best_radius, best_inlier_mask
    
    def fit_constrained_cylinder_ur3(self, points_3d, max_iterations=500, epsilon=0.0005, min_inliers=30):
        """
        Constrained cylinder fitting optimized for UR3 "Finding Pose" setup
        
        Assumptions:
        - Camera is perpendicular to work plane (top-down view)
        - Cylinders are always vertical (axis = [0, 0, -1] toward camera)
        - Known cylinder dimensions
        
        Args:
            points_3d: Nx3 numpy array of 3D points
            max_iterations: Maximum RANSAC iterations  
            epsilon: Inlier threshold in meters
            min_inliers: Minimum number of inliers
            
        Returns:
            tuple: (success, cylinder_params, fit_quality)
        """
        if len(points_3d) < 3:
            print("Need at least 3 points for constrained cylinder fitting")
            return False, None, None
        
        print(f"\nStarting UR3 constrained cylinder fitting...")
        print(f"Input: {len(points_3d)} 3D points")
        
        # Step 1: Project to 2D (drop Z coordinate for circle fitting)
        points_2d = points_3d[:, :2]  # Keep only X, Y coordinates
        print(f"Projected to {len(points_2d)} 2D points for circle fitting")
        
        # Step 2: Fit 2D circle using RANSAC
        circle_success, center_x, center_y, radius_2d, inlier_mask_2d = self.ransac_circle_fitting_2d(
            points_2d, max_iterations, epsilon, min_inliers
        )
        
        if not circle_success:
            return False, None, None
        
        # Step 3: Determine cylinder height and Z position
        inlier_points_3d = points_3d[inlier_mask_2d]
        z_values = inlier_points_3d[:, 2]
        
        z_top = np.min(z_values)      # Closest to camera (top surface)
        z_bottom = np.max(z_values)   # Furthest from camera (could be noise or actual bottom)
        z_mean = np.mean(z_values)
        z_std = np.std(z_values)
        
        # For top-down view, most points should be at the top surface
        # Use the mode/median Z value as the cylinder top
        z_cylinder_top = np.median(z_values)
        z_cylinder_bottom = z_cylinder_top + self.cylinder_height  # Known height
        z_cylinder_center = (z_cylinder_top + z_cylinder_bottom) / 2
        
        print(f"Z analysis: top={z_top:.4f}m, bottom={z_bottom:.4f}m, mean={z_mean:.4f}m, std={z_std:.4f}m")
        print(f"Cylinder Z: top={z_cylinder_top:.4f}m, center={z_cylinder_center:.4f}m, bottom={z_cylinder_bottom:.4f}m")
        
        # Step 4: Create 3D cylinder parameters
        cylinder_center = np.array([center_x, center_y, z_cylinder_center])
        cylinder_top_center = np.array([center_x, center_y, z_cylinder_top])
        cylinder_bottom_center = np.array([center_x, center_y, z_cylinder_bottom])
        cylinder_axis = np.array([0.0, 0.0, -1.0])  # Always pointing toward camera (down)
        
        # Step 5: Validate against known constraints
        radius_error = abs(radius_2d - self.cylinder_radius)
        radius_error_percentage = (radius_error / self.cylinder_radius) * 100
        
        # Step 6: Compute quality metrics
        inlier_count = np.sum(inlier_mask_2d)
        inlier_percentage = (inlier_count / len(points_3d)) * 100
        
        cylinder_params = {
            'center': cylinder_center,
            'top_center': cylinder_top_center,
            'bottom_center': cylinder_bottom_center,
            'axis': cylinder_axis,
            'radius': radius_2d,
            'height': self.cylinder_height,
            'inliers': inlier_points_3d,
            'all_points': points_3d,
            'inlier_mask': inlier_mask_2d
        }
        
        fit_quality = {
            'inlier_count': inlier_count,
            'total_points': len(points_3d),
            'inlier_percentage': inlier_percentage,
            'radius_error': radius_error,
            'radius_error_percentage': radius_error_percentage,
            'z_variation': z_std,
            'method': 'constrained_2d_circle'
        }
        
        print(f"✓ Constrained cylinder fitting completed:")
        print(f"  Center: [{cylinder_center[0]:.4f}, {cylinder_center[1]:.4f}, {cylinder_center[2]:.4f}]")
        print(f"  Radius: {radius_2d:.4f}m (error: {radius_error_percentage:.1f}%)")
        print(f"  Inliers: {inlier_count}/{len(points_3d)} ({inlier_percentage:.1f}%)")
        
        return True, cylinder_params, fit_quality

    def visualize_constrained_cylinder_fit(self, cylinder_params, fit_quality, bbox, color_image):
        """
        Visualize the constrained cylinder fitting results for UR3 setup
        """
        fig = plt.figure(figsize=(20, 5))
        
        points_3d = cylinder_params['all_points']
        inlier_mask = cylinder_params['inlier_mask']
        center = cylinder_params['center']
        radius = cylinder_params['radius']
        
        # Plot 1: Original detection
        ax1 = fig.add_subplot(141)
        ax1.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               fill=False, color='red', linewidth=2)
            ax1.add_patch(rect)
        ax1.set_title('YOLO Detection')
        ax1.axis('off')
        
        # Plot 2: 2D Circle Fit - Top view
        ax2 = fig.add_subplot(142)
        if len(points_3d) > 0:
            # Project to 2D
            points_2d = points_3d[:, :2]
            
            # Plot outliers in red
            outliers_2d = points_2d[~inlier_mask]
            if len(outliers_2d) > 0:
                ax2.scatter(outliers_2d[:, 0], outliers_2d[:, 1], c='red', s=3, alpha=0.6, label='Outliers')
            
            # Plot inliers in blue
            inliers_2d = points_2d[inlier_mask]
            if len(inliers_2d) > 0:
                ax2.scatter(inliers_2d[:, 0], inliers_2d[:, 1], c='blue', s=3, label='Inliers')
            
            # Draw fitted circle
            circle = plt.Circle((center[0], center[1]), radius, fill=False, color='green', linewidth=2, label='Fitted Circle')
            ax2.add_patch(circle)
            
            # Draw expected circle for comparison
            expected_circle = plt.Circle((center[0], center[1]), self.cylinder_radius, 
                                       fill=False, color='orange', linewidth=2, linestyle='--', label='Expected Circle')
            ax2.add_patch(expected_circle)
            
            # Mark center
            ax2.plot(center[0], center[1], 'g+', markersize=10, markeredgewidth=2, label='Center')
            
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('2D Circle Fit (Top View)')
            ax2.legend()
            ax2.axis('equal')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: 3D view
        ax3 = fig.add_subplot(143, projection='3d')
        if len(points_3d) > 0:
            # Plot outliers
            outliers_3d = points_3d[~inlier_mask]
            if len(outliers_3d) > 0:
                ax3.scatter(outliers_3d[:, 0], outliers_3d[:, 1], outliers_3d[:, 2], 
                           c='red', s=3, alpha=0.6, label='Outliers')
            
            # Plot inliers
            inliers_3d = points_3d[inlier_mask]
            if len(inliers_3d) > 0:
                ax3.scatter(inliers_3d[:, 0], inliers_3d[:, 1], inliers_3d[:, 2], 
                           c='blue', s=3, label='Inliers')
            
            # Draw cylinder axis
            top_center = cylinder_params['top_center']
            bottom_center = cylinder_params['bottom_center']
            ax3.plot([top_center[0], bottom_center[0]], 
                    [top_center[1], bottom_center[1]], 
                    [top_center[2], bottom_center[2]], 
                    'g-', linewidth=3, label='Cylinder Axis')
            
            # Mark centers
            ax3.scatter(*top_center, c='green', s=50, marker='^', label='Top Center')
            ax3.scatter(*bottom_center, c='green', s=50, marker='v', label='Bottom Center')
            
            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Y (m)')
            ax3.set_zlabel('Z (m)')
            ax3.set_title('3D View with Constrained Cylinder')
            ax3.legend()
        
        # Plot 4: Results summary
        ax4 = fig.add_subplot(144)
        if len(points_3d) > 0:
            ax4.text(0.05, 0.9, 'UR3 Constrained Cylinder Fit', fontsize=14, fontweight='bold')
            ax4.text(0.05, 0.8, f'Method: {fit_quality["method"]}', fontsize=10)
            ax4.text(0.05, 0.75, f'Fitted Radius: {radius:.4f} m', fontsize=10)
            ax4.text(0.05, 0.7, f'Expected Radius: {self.cylinder_radius:.4f} m', fontsize=10, color='green')
            ax4.text(0.05, 0.65, f'Radius Error: {fit_quality["radius_error"]:.4f} m ({fit_quality["radius_error_percentage"]:.1f}%)', 
                    fontsize=10, color='red' if fit_quality["radius_error_percentage"] > 5 else 'green')
            
            ax4.text(0.05, 0.55, f'Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]', fontsize=10)
            ax4.text(0.05, 0.5, f'Top Center: [{top_center[0]:.4f}, {top_center[1]:.4f}, {top_center[2]:.4f}]', fontsize=10)
            ax4.text(0.05, 0.45, f'Axis: [{cylinder_params["axis"][0]:.1f}, {cylinder_params["axis"][1]:.1f}, {cylinder_params["axis"][2]:.1f}]', fontsize=10)
            
            ax4.text(0.05, 0.35, f'Inliers: {fit_quality["inlier_count"]}/{fit_quality["total_points"]} ({fit_quality["inlier_percentage"]:.1f}%)', fontsize=10)
            ax4.text(0.05, 0.3, f'Z Variation: {fit_quality["z_variation"]/.001:.4f} mm', fontsize=10)
            
            # Quality assessment
            if fit_quality['radius_error_percentage'] < 5:
                quality_text = "🟢 Excellent accuracy!"
                quality_color = 'green'
            elif fit_quality['radius_error_percentage'] < 15:
                quality_text = "🟡 Good accuracy"
                quality_color = 'orange'
            else:
                quality_text = "🔴 Poor accuracy"
                quality_color = 'red'
            
            ax4.text(0.05, 0.2, quality_text, fontsize=12, color=quality_color, fontweight='bold')
            
            # Improvement vs original method
            if fit_quality['radius_error_percentage'] < 20:  # Much better than 150%+ we had before
                ax4.text(0.05, 0.1, "✅ Major improvement over 3D RANSAC!", fontsize=10, color='green', fontweight='bold')
            
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_title('UR3 Optimized Results')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.show()

    def fit_cylinder_with_ur3_optimization(self, confidence_threshold=0.3, downsample_factor=2, 
                                          max_iterations=500, epsilon=0.0005, min_inliers=20, visualize=False):
        """
        Complete UR3-optimized pipeline: Detect cylinder, extract point cloud, and fit with constraints
        
        Args:
            confidence_threshold (float): Minimum confidence for YOLO detection
            downsample_factor (int): Factor to downsample point cloud
            max_iterations (int): Maximum RANSAC iterations
            epsilon (float): RANSAC inlier threshold in meters (0.5mm default)
            min_inliers (int): Minimum number of inliers required
            visualize (bool): Whether to show visualization
            
        Returns:
            tuple: (success, cylinder_params, fit_quality)
        """
        # Step 1: Get point cloud
        success, point_cloud, bbox, detection_confidence = self.get_cylinder_point_cloud(
            confidence_threshold, downsample_factor, visualize=False
        )
        
        if not success or point_cloud is None:
            return False, None, None
        
        # Step 2: Apply UR3 constrained cylinder fitting
        print(f"\nStarting UR3-optimized cylinder fitting...")
        print(f"Using constrained 2D circle approach with vertical cylinder assumption")
        
        fit_success, cylinder_params, fit_quality = self.fit_constrained_cylinder_ur3(
            point_cloud, max_iterations, epsilon, min_inliers
        )
        
        if not fit_success:
            return False, None, None
        
        # Step 3: Add detection confidence to quality metrics
        fit_quality['detection_confidence'] = detection_confidence
        
        # Step 4: Visualize if requested
        if visualize:
            # Get the color image for visualization
            color_image, _ = self.camera.get_frames()
            if color_image is not None:
                self.visualize_constrained_cylinder_fit(
                    cylinder_params, fit_quality, bbox, color_image
                )
        
        return True, cylinder_params, fit_quality

    def estimate_6dof_pose_camera_frame(self, cylinder_params):
        """
        Estimate 6DOF pose of cylinder in camera coordinate frame
        
        Args:
            cylinder_params: Dictionary containing fitted cylinder parameters
            
        Returns:
            dict: 6DOF pose in camera frame:
            {
                'position': [x, y, z],           # Position in meters (camera frame)
                'rotation_euler': [rx, ry, rz],  # Euler angles in radians  
                'rotation_matrix': 3x3 matrix,   # Rotation matrix
                'pose_confidence': float         # Confidence score [0-1]
            }
        """
        # Extract cylinder center as the reference point
        center_x, center_y, center_z = cylinder_params['center']
        
        # Position: Use cylinder center as the object coordinate
        position = np.array([center_x, center_y, center_z])
        
        # Rotation: Canonical orientation for cylinder
        # Z-axis points along cylinder axis (toward camera) = [0, 0, -1]
        # X-axis and Y-axis are in the plane perpendicular to cylinder
        rotation_matrix = np.array([
            [1.0, 0.0, 0.0],    # X-axis
            [0.0, 1.0, 0.0],    # Y-axis  
            [0.0, 0.0, -1.0]    # Z-axis (cylinder axis)
        ])
        
        # Convert to Euler angles (XYZ convention)
        rotation_euler = np.array([0.0, 0.0, 0.0])  # [rx, ry, rz] in radians
        
        # Calculate pose confidence
        pose_confidence = self._calculate_pose_confidence(cylinder_params)
        
        pose_6dof = {
            'position': position,
            'rotation_euler': rotation_euler, 
            'rotation_matrix': rotation_matrix,
            'pose_confidence': pose_confidence
        }
        
        return pose_6dof
                                                                
    def _calculate_pose_confidence(self, cylinder_params):
        """Calculate confidence score for the estimated pose - optimized for UR3 top-down grasping"""
        # Get fitting quality metrics
        if 'inlier_mask' in cylinder_params:
            inlier_count = np.sum(cylinder_params['inlier_mask'])
            total_points = len(cylinder_params['all_points'])
            inlier_ratio = inlier_count / total_points if total_points > 0 else 0
        else:
            inlier_ratio = 0.7
        
        # Position stability - how well do inliers agree on center position
        if 'inliers' in cylinder_params and len(cylinder_params['inliers']) > 10:
            inlier_points = cylinder_params['inliers']
            
            # Check X,Y position consistency (most important for grasping)
            xy_points = inlier_points[:, :2]  # Get X,Y coordinates
            xy_center = np.mean(xy_points, axis=0)
            xy_distances = np.linalg.norm(xy_points - xy_center, axis=1)
            xy_std = np.std(xy_distances)
            
            # Check Z position consistency  
            z_values = inlier_points[:, 2]
            z_std = np.std(z_values)
            
            # Position stability score (lower std = higher stability)
            # Use exponential decay: stability = exp(-std/tolerance)
            xy_stability = np.exp(-xy_std / 0.001)  # 1mm tolerance for XY
            z_stability = np.exp(-z_std / 0.002)    # 2mm tolerance for Z
            position_stability = (xy_stability + z_stability) / 2
        else:
            position_stability = 0.5  # Default if not enough inliers
        
        # Point density score - more points = higher confidence
        point_density_score = min(1.0, inlier_count / 50.0)  # Saturate at 50 inliers
        
        # Detection consistency - radius should be reasonable (but not heavily penalized)
        fitted_radius = cylinder_params['radius']
        expected_radius = self.cylinder_radius
        radius_error = abs(fitted_radius - expected_radius) / expected_radius
        # Gentle penalty: only heavily penalize if radius is >100% off (clearly wrong)
        radius_reasonableness = 1.0 if radius_error < 1.0 else max(0.3, 1.0 - radius_error)
        
        # Final confidence: Focus on position and inliers
        # 50% inlier ratio + 30% position stability + 15% point density + 5% radius check
        confidence = (0.50 * inlier_ratio + 
                     0.30 * position_stability +
                     0.15 * point_density_score +
                     0.05 * radius_reasonableness)
        
        return max(0.0, min(1.0, confidence))

    def get_cylinder_6dof_pose(self, confidence_threshold=0.3, downsample_factor=2, 
                              max_iterations=500, epsilon=0.0005, min_inliers=20, visualize=False):
        """
        Complete pipeline: Detect cylinder and estimate 6DOF pose in camera frame
        
        Returns:
            tuple: (success, pose_6dof)
            - success: Boolean indicating if pose estimation was successful
            - pose_6dof: Dictionary with position, rotation, and confidence
        """
        # Step 1: Run cylinder fitting
        fit_success, cylinder_params, fit_quality = self.fit_cylinder_with_ur3_optimization(
            confidence_threshold, downsample_factor, max_iterations, epsilon, min_inliers, visualize=False
        )
        
        if not fit_success:
            return False, None
        
        # Step 2: Estimate 6DOF pose
        pose_6dof = self.estimate_6dof_pose_camera_frame(cylinder_params)
        
        # Step 3: Visualize if requested
        if visualize:
            self._visualize_6dof_pose_simple(cylinder_params, pose_6dof)
        
        return True, pose_6dof

    def _visualize_6dof_pose_simple(self, cylinder_params, pose_6dof):
        """Simple visualization focused on pose estimation results"""
        fig = plt.figure(figsize=(15, 5))
        
        points_3d = cylinder_params['all_points']
        inlier_mask = cylinder_params.get('inlier_mask', np.ones(len(points_3d), dtype=bool))
        position = pose_6dof['position']
        rotation_matrix = pose_6dof['rotation_matrix']
        
        # Plot 1: 3D point cloud with pose coordinate frame
        ax1 = fig.add_subplot(131, projection='3d')
        if len(points_3d) > 0:
            # Plot points
            outliers_3d = points_3d[~inlier_mask]
            inliers_3d = points_3d[inlier_mask]
            
            if len(outliers_3d) > 0:
                ax1.scatter(outliers_3d[:, 0], outliers_3d[:, 1], outliers_3d[:, 2], 
                           c='red', s=2, alpha=0.4, label='Outliers')
            if len(inliers_3d) > 0:
                ax1.scatter(inliers_3d[:, 0], inliers_3d[:, 1], inliers_3d[:, 2], 
                           c='blue', s=2, label='Inliers')
            
            # Draw coordinate frame at estimated pose
            origin = position
            scale = 0.01  # 1cm arrows
            
            # X-axis (red), Y-axis (green), Z-axis (blue)
            for i, color in enumerate(['r', 'g', 'b']):
                end = origin + scale * rotation_matrix[:, i]
                ax1.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], 
                        f'{color}-', linewidth=3, label=f'{["X","Y","Z"][i]}-axis')
            
            ax1.scatter(*origin, c='yellow', s=100, marker='*', label='Pose Origin')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D Point Cloud + 6DOF Pose')
            ax1.legend()
        
        # Plot 2: Top view with position
        ax2 = fig.add_subplot(132)
        if len(points_3d) > 0:
            points_2d = points_3d[:, :2]
            inliers_2d = points_2d[inlier_mask]
            outliers_2d = points_2d[~inlier_mask]
            
            if len(outliers_2d) > 0:
                ax2.scatter(outliers_2d[:, 0], outliers_2d[:, 1], c='red', s=3, alpha=0.6, label='Outliers')
            if len(inliers_2d) > 0:
                ax2.scatter(inliers_2d[:, 0], inliers_2d[:, 1], c='blue', s=3, label='Inliers')
            
            # Draw fitted circle and pose position
            circle = plt.Circle((position[0], position[1]), cylinder_params['radius'], 
                              fill=False, color='green', linewidth=2, label='Fitted Circle')
            ax2.add_patch(circle)
            ax2.plot(position[0], position[1], 'y*', markersize=15, label='6DOF Position')
            
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Top View - Position')
            ax2.legend()
            ax2.axis('equal')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Pose information
        ax3 = fig.add_subplot(133)
        ax3.text(0.05, 0.9, '6DOF POSE (Camera Frame)', fontsize=14, fontweight='bold')
        
        ax3.text(0.05, 0.75, 'POSITION:', fontsize=12, fontweight='bold', color='blue')
        ax3.text(0.05, 0.7, f'  X: {position[0]:+.4f} m', fontsize=10)
        ax3.text(0.05, 0.65, f'  Y: {position[1]:+.4f} m', fontsize=10)
        ax3.text(0.05, 0.6, f'  Z: {position[2]:+.4f} m', fontsize=10)
        
        ax3.text(0.05, 0.45, 'ROTATION (Euler XYZ):', fontsize=12, fontweight='bold', color='green')
        euler = pose_6dof['rotation_euler']
        ax3.text(0.05, 0.4, f'  RX: {np.degrees(euler[0]):+.1f}°', fontsize=10)
        ax3.text(0.05, 0.35, f'  RY: {np.degrees(euler[1]):+.1f}°', fontsize=10)
        ax3.text(0.05, 0.3, f'  RZ: {np.degrees(euler[2]):+.1f}°', fontsize=10)
        
        confidence = pose_6dof['pose_confidence']
        ax3.text(0.05, 0.15, f'CONFIDENCE: {confidence:.3f}', 
                fontsize=12, fontweight='bold', 
                color='green' if confidence > 0.8 else 'orange' if confidence > 0.6 else 'red')
        
        # Show confidence breakdown for debugging
        if 'inlier_mask' in cylinder_params and len(cylinder_params.get('inliers', [])) > 10:
            ax3.text(0.55, 0.9, 'CONFIDENCE BREAKDOWN:', fontsize=10, fontweight='bold')
            
            # Calculate individual components (copy from confidence method)
            inlier_count = np.sum(cylinder_params['inlier_mask'])
            total_points = len(cylinder_params['all_points'])
            inlier_ratio = inlier_count / total_points
            
            inlier_points = cylinder_params['inliers']
            xy_points = inlier_points[:, :2]
            xy_center = np.mean(xy_points, axis=0)
            xy_distances = np.linalg.norm(xy_points - xy_center, axis=1)
            xy_std = np.std(xy_distances)
            z_std = np.std(inlier_points[:, 2])
            
            xy_stability = np.exp(-xy_std / 0.001)
            z_stability = np.exp(-z_std / 0.002)
            position_stability = (xy_stability + z_stability) / 2
            point_density_score = min(1.0, inlier_count / 50.0)
            
            fitted_radius = cylinder_params['radius']
            expected_radius = self.cylinder_radius  
            radius_error = abs(fitted_radius - expected_radius) / expected_radius
            radius_reasonableness = 1.0 if radius_error < 1.0 else max(0.3, 1.0 - radius_error)
            
            ax3.text(0.55, 0.8, f'Inlier Ratio: {inlier_ratio:.3f} (50%)', fontsize=8)
            ax3.text(0.55, 0.75, f'Position Stability: {position_stability:.3f} (30%)', fontsize=8)
            ax3.text(0.55, 0.7, f'Point Density: {point_density_score:.3f} (15%)', fontsize=8)
            ax3.text(0.55, 0.65, f'Radius Check: {radius_reasonableness:.3f} (5%)', fontsize=8)
            
            ax3.text(0.55, 0.55, f'XY Std: {xy_std*1000:.2f}mm', fontsize=8)
            ax3.text(0.55, 0.5, f'Z Std: {z_std*1000:.2f}mm', fontsize=8)
            ax3.text(0.55, 0.45, f'Radius Error: {radius_error*100:.1f}%', fontsize=8)
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()

def test_6dof_pose_estimation():
    """Test the 6DOF pose estimation in camera coordinates"""
    estimator = CylinderPoseEstimator()
    
    if not estimator.connect():
        print("Failed to connect to camera")
        return
    
    print("\n" + "="*60)
    print("🎯 6DOF CYLINDER POSE ESTIMATION")
    print("="*60)
    print("Pipeline: YOLO → Point Cloud → Cylinder Fitting → 6DOF Pose")
    print("Output: Position + Rotation in camera coordinate frame")
    print("\nControls:")
    print("  'p' - Estimate 6DOF pose")
    print("  'q' - Quit")
    
    try:
        while True:
            # Get current frame for preview
            color_image, depth_image = estimator.camera.get_frames()
            if color_image is not None:
                cv2.imshow('6DOF Pose Estimation', color_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                print("\n🚀 Estimating 6DOF pose...")
                print("-" * 40)
                
                # Run complete pipeline
                success, pose_6dof = estimator.get_cylinder_6dof_pose(
                    confidence_threshold=0.3,
                    downsample_factor=2,
                    max_iterations=500,
                    epsilon=0.0005,  # 0.5mm tolerance
                    min_inliers=20,
                    visualize=True
                )
                
                if success:
                    print("\n✅ 6DOF POSE ESTIMATED SUCCESSFULLY!")
                    print("="*50)
                    
                    position = pose_6dof['position']
                    rotation = pose_6dof['rotation_euler']
                    confidence = pose_6dof['pose_confidence']
                    
                    print(f"📍 POSITION (Camera Frame):")
                    print(f"   X: {position[0]:+.4f} m")
                    print(f"   Y: {position[1]:+.4f} m")
                    print(f"   Z: {position[2]:+.4f} m")
                    
                    print(f"\n🔄 ROTATION (Euler XYZ):")
                    print(f"   RX: {np.degrees(rotation[0]):+.1f}°")
                    print(f"   RY: {np.degrees(rotation[1]):+.1f}°")
                    print(f"   RZ: {np.degrees(rotation[2]):+.1f}°")
                    
                    print(f"\n📊 CONFIDENCE: {confidence:.3f}")
                    
                    if confidence > 0.8:
                        print("   ✅ HIGH QUALITY pose estimation")
                    elif confidence > 0.6:
                        print("   🟡 MEDIUM QUALITY pose estimation")
                    else:
                        print("   ❌ LOW QUALITY pose estimation")
                        
                else:
                    print("\n❌ 6DOF POSE ESTIMATION FAILED!")
                    print("Check: YOLO detection, point cloud quality, cylinder visibility")
                
                print(f"\nPress 'p' for another pose estimation...")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        estimator.disconnect()
        cv2.destroyAllWindows()

def test_point_cloud_extraction():
    """Test basic point cloud extraction"""
    estimator = CylinderPoseEstimator()
    
    if not estimator.connect():
        print("Failed to connect to camera")
        return
    
    print("\nBasic point cloud extraction test...")
    print("Press 'c' to capture and process, 'q' to quit")
    
    try:
        while True:
            color_image, depth_image = estimator.camera.get_frames()
            if color_image is not None:
                cv2.imshow('Camera Feed', color_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("\nProcessing current frame...")
                success, point_cloud, bbox, confidence = estimator.get_cylinder_point_cloud(
                    confidence_threshold=0.3, downsample_factor=2, visualize=True
                )
                
                if success:
                    print(f"✅ Extracted {len(point_cloud)} points")
                else:
                    print("❌ Failed to extract point cloud")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        estimator.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Choose which test to run
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "basic":
            test_point_cloud_extraction()
        elif sys.argv[1] == "6dof" or sys.argv[1] == "pose":
            test_6dof_pose_estimation()
        else:
            print("Usage: python test_point_cloud_extraction.py [basic|6dof|pose]")
            print("  (no args)  - 6DOF pose estimation (default)")
            print("  basic      - Basic point cloud extraction test")
            print("  6dof|pose  - 6DOF pose estimation test")
    else:
        # Default to 6DOF pose estimation
        test_6dof_pose_estimation()