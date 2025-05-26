import numpy as np
import cv2
import torch
from ultralytics import YOLO
from pick_n_place.camera import RealSenseCamera
from pick_n_place.hand_eye_calibration import HandEyeCalibrator

class ProductDetector:
    def __init__(self, model_path='best.pt', robot_ip=None, calibration_file=None):
        """
        Initialize the product detector
        
        Args:
            model_path (str): Path to the YOLO model weights
            robot_ip (str): IP address of the UR robot
            calibration_file (str): Path to the hand-eye calibration file
        """
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Initialize camera
        self.camera = RealSenseCamera()
        
        # Initialize calibration if provided
        self.calibrator = None
        if robot_ip and calibration_file:
            self.calibrator = HandEyeCalibrator(robot_ip)
            self.calibrator.load_calibration(calibration_file)
    
    def connect(self):
        """Connect to the camera"""
        return self.camera.connect()
    
    def disconnect(self):
        """Disconnect from the camera"""
        self.camera.disconnect()
    
    def detect_product(self, confidence_threshold=0.5):
        """
        Detect product in the current camera frame
        
        Args:
            confidence_threshold (float): Minimum confidence score for detection
            
        Returns:
            tuple: (success, bbox, confidence, depth, image)
                - success: Boolean indicating if detection was successful
                - bbox: Bounding box [x1, y1, x2, y2]
                - confidence: Detection confidence score
                - depth: Depth value at the center of bbox (in meters)
                - image: Annotated RGB image
        """
        # Get frames from RealSense
        color_image, depth_image = self.camera.get_frames()
        if color_image is None or depth_image is None:
            return False, None, None, None, None
        
        # Run YOLO detection
        results = self.model(color_image)[0]
        
        # Process results
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
            return False, None, None, None, color_image
        
        # Get bounding box
        bbox = best_detection[:4].cpu().numpy()
        
        # Get depth at center of bounding box
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        depth_value = depth_image[center_y, center_x] * self.camera.depth_scale
        
        # Draw detection on image
        annotated_image = color_image.copy()
        cv2.rectangle(annotated_image, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     (0, 255, 0), 2)
        
        # Add text with confidence and depth
        text = f"Product: {best_confidence:.2f}, Depth: {depth_value:.3f}m"
        cv2.putText(annotated_image, text, 
                   (int(bbox[0]), int(bbox[1] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return True, bbox, best_confidence, depth_value, annotated_image, depth_image
    
    def estimate_orientation(self, bbox, depth_image):
        """
        Estimate object orientation using depth information and bounding box
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            depth_image: Depth image from RealSense
            
        Returns:
            tuple: (rx, ry, rz) Euler angles in radians
        """
        # Extract region of interest from depth image
        x1, y1, x2, y2 = map(int, bbox)
        roi_depth = depth_image[y1:y2, x1:x2] * self.camera.depth_scale
        
        # Calculate surface normal using depth gradients
        gy, gx = np.gradient(roi_depth)
        
        # Average gradients to get dominant surface normal
        normal = np.array([-gx.mean(), -gy.mean(), 1.0])
        normal = normal / np.linalg.norm(normal)
        
        # Calculate rotation matrix that aligns z-axis with surface normal
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, normal)
        rotation_angle = np.arccos(np.dot(z_axis, normal))
        
        # Convert axis-angle to rotation matrix
        if np.linalg.norm(rotation_axis) > 0:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            R = cv2.Rodrigues(rotation_axis * rotation_angle)[0]
        else:
            R = np.eye(3)
        
        # Get principal orientation from bounding box
        bbox_angle = np.arctan2(y2 - y1, x2 - x1)
        Rz = cv2.Rodrigues(np.array([0, 0, bbox_angle]))[0]
        
        # Combine rotations
        R = R @ Rz
        
        # Convert rotation matrix to euler angles
        euler_angles = cv2.RQDecomp3x3(R)[0]
        
        return euler_angles * np.pi / 180.0  # Convert to radians
    
    def get_product_pose_camera_frame(self):
        """
        Get the 6D pose of the detected product in camera frame
        
        Returns:
            tuple: (success, pose, image)
                - success: Boolean indicating if pose was obtained
                - pose: [x, y, z, rx, ry, rz] in camera frame (meters, radians)
                - image: Annotated RGB image
        """
        success, bbox, conf, depth, image, depth_image = self.detect_product()
        if not success:
            return False, None, image
        
        # Get center point of bounding box
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Get 3D position
        point_3d = self.camera.get_3d_point(int(center_x), int(center_y), depth)
        if point_3d is None:
            return False, None, image
        
        # Get orientation
        orientation = self.estimate_orientation(bbox, depth_image)
        
        # Combine position and orientation
        pose = np.concatenate([point_3d, orientation])
        
        return True, pose, image
    
    def get_product_pose_robot_frame(self):
        """
        Get the 6D pose of the detected product in robot base frame
        
        Returns:
            tuple: (success, pose, image)
                - success: Boolean indicating if pose was obtained
                - pose: [x, y, z, rx, ry, rz] in robot base frame (meters, radians)
                - image: Annotated RGB image
        """
        if self.calibrator is None:
            raise ValueError("Calibration not loaded. Initialize with robot_ip and calibration_file")
        
        # Get pose in camera frame
        success, camera_pose, image = self.get_product_pose_camera_frame()
        if not success:
            return False, None, image
        
        # Convert to homogeneous transformation matrix
        camera_matrix = np.eye(4)
        camera_matrix[:3, 3] = camera_pose[:3]
        camera_matrix[:3, :3] = cv2.Rodrigues(camera_pose[3:])[0]
        
        # Get current robot pose
        robot_pose = self.calibrator.robot.get_tcp_pose()
        if robot_pose is None:
            return False, None, image
            
        robot_matrix = self.calibrator._pose_to_matrix(robot_pose)
        
        # Transform to robot base coordinates
        pose_in_base = robot_matrix @ self.calibrator.eye_hand_matrix @ camera_matrix
        
        # Extract position and orientation
        position = pose_in_base[:3, 3]
        orientation, _ = cv2.Rodrigues(pose_in_base[:3, :3])
        
        return True, np.concatenate([position, orientation.flatten()]), image

def test_detection():
    """Test the product detection"""
    detector = ProductDetector()
    
    if not detector.connect():
        print("Failed to connect to camera")
        return
    
    print("\nStarting product detection test...")
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        while True:
            # Get detection with pose
            success, pose, image = detector.get_product_pose_camera_frame()
            
            if image is not None:
                # Show image
                cv2.imshow('Product Detection', image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and success:
                    cv2.imwrite('product_detection.jpg', image)
                    print(f"\nDetection saved:")
                    print(f"Position (x,y,z): {pose[:3]}")
                    print(f"Orientation (rx,ry,rz): {pose[3:]}")
    
    except KeyboardInterrupt:
        print("\nStopping detection...")
    
    finally:
        detector.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_detection() 