import numpy as np
import cv2
import time
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pick_n_place.camera import RealSenseCamera
from URRobotControl import URRobotControl

class CameraPoseEstimator:
    def __init__(self, marker_size=0.1, marker_id=7):
        """
        Initialize the camera pose estimator for ArUco marker detection
        
        Args:
            marker_size (float): Size of the ArUco marker in meters
            marker_id (int): ID of the ArUco marker to detect
        """
        self.camera = RealSenseCamera()
        self.marker_size = marker_size
        self.marker_id = marker_id
        
        # ArUco dictionary and parameters - handle different OpenCV versions
        try:
            # OpenCV 4.7+
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            try:
                # OpenCV 4.0-4.6
                self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            except AttributeError:
                # Very old OpenCV versions
                raise ImportError("OpenCV version too old for ArUco support")
                
    def connect(self):
        """Connect to the camera"""
        return self.camera.connect()
    
    def disconnect(self):
        """Disconnect from the camera"""
        self.camera.disconnect()
    
    def detect_marker_pose(self):
        """
        Detect ArUco marker and estimate its pose
        
        Returns:
            tuple: (success, rvec, tvec, image)
                - success: Boolean indicating if marker was detected
                - rvec: Rotation vector of the marker relative to camera
                - tvec: Translation vector of the marker relative to camera
                - image: Current camera image with marker detection drawn
        """
        # Get camera frame
        color_image, _ = self.camera.get_frames()
        if color_image is None:
            return False, None, None, None
        
        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers - handle different OpenCV versions
        try:
            # OpenCV 4.7+
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            # OpenCV 4.0-4.6
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        # Check if our target marker was detected
        if ids is not None and self.marker_id in ids:
            # Find the index of our target marker
            marker_index = np.where(ids == self.marker_id)[0][0]
            marker_corners = corners[marker_index]
            
            # Define 3D object points for the ArUco marker
            # ArUco markers are defined with origin at center, with corners at:
            # Top-left, Top-right, Bottom-right, Bottom-left
            half_size = self.marker_size / 2.0
            object_points = np.array([
                [-half_size, -half_size, 0],  # Top-left
                [ half_size, -half_size, 0],  # Top-right
                [ half_size,  half_size, 0],  # Bottom-right
                [-half_size,  half_size, 0]   # Bottom-left
            ], dtype=np.float32)
            
            # Get 2D image points (corners of detected marker)
            image_points = marker_corners[0].astype(np.float32)
            
            # Solve PnP to get pose
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                self.camera.camera_matrix, self.camera.dist_coeffs
            )
            
            if not success:
                return False, None, None, color_image
            
            # Draw detected marker and pose axis
            image_with_detection = color_image.copy()
            
            # Draw markers
            cv2.aruco.drawDetectedMarkers(image_with_detection, corners, ids)
            
            # Draw coordinate axis
            try:
                cv2.drawFrameAxes(image_with_detection, self.camera.camera_matrix, 
                                self.camera.dist_coeffs, rvec, tvec, self.marker_size * 0.5)
            except AttributeError:
                # Fallback for older OpenCV versions - project and draw manually
                axis_points = np.array([
                    [0, 0, 0],                           # Origin
                    [self.marker_size * 0.5, 0, 0],      # X-axis (red)
                    [0, self.marker_size * 0.5, 0],      # Y-axis (green)
                    [0, 0, -self.marker_size * 0.5]      # Z-axis (blue)
                ], dtype=np.float32)
                
                projected_points, _ = cv2.projectPoints(
                    axis_points, rvec, tvec, 
                    self.camera.camera_matrix, self.camera.dist_coeffs
                )
                
                projected_points = np.int32(projected_points).reshape(-1, 2)
                
                # Draw axes
                cv2.arrowedLine(image_with_detection, 
                               tuple(projected_points[0]), tuple(projected_points[1]), 
                               (0, 0, 255), 3)  # X-axis in red
                cv2.arrowedLine(image_with_detection, 
                               tuple(projected_points[0]), tuple(projected_points[2]), 
                               (0, 255, 0), 3)  # Y-axis in green
                cv2.arrowedLine(image_with_detection, 
                               tuple(projected_points[0]), tuple(projected_points[3]), 
                               (255, 0, 0), 3)  # Z-axis in blue
            
            return True, rvec, tvec, image_with_detection
        
        else:
            # No marker detected
            return False, None, None, color_image

class HandEyeCalibrator:
    def __init__(self, robot_ip, marker_size=0.1):
        """
        Initialize the hand-eye calibrator
        
        Args:
            robot_ip (str): IP address of the UR robot
            marker_size (float): Size of the ArUco marker in meters
        """
        self.robot = URRobotControl(robot_ip)
        self.camera = CameraPoseEstimator(marker_size)
        
        # Lists to store calibration data
        self.robot_poses = []  # Base -> End-effector transforms
        self.camera_poses = []  # Camera -> Marker transforms
        
        # Calibration result
        self.eye_hand_matrix = None
        
    def connect(self):
        """Connect to both robot and camera"""
        if not self.robot.connect():
            print("Failed to connect to robot")
            return False
        if not self.camera.connect():
            print("Failed to connect to camera")
            return False
        return True
        
    def disconnect(self):
        """Disconnect from both robot and camera"""
        self.robot.disconnect()
        self.camera.disconnect()
        
    def collect_calibration_sample(self):
        """
        Collect a single calibration sample at current robot pose
        
        Returns:
            bool: True if sample collection was successful
        """
        # Get robot's TCP pose
        tcp_pose = self.robot.get_tcp_pose()
        if tcp_pose is None:
            print("Failed to get TCP pose")
            return False
            
        # Get marker pose from camera
        success, rvec, tvec, _ = self.camera.detect_marker_pose()
        if not success:
            print("Failed to detect marker")
            return False
            
        # Convert TCP pose to transformation matrix
        tcp_matrix = self._pose_to_matrix(tcp_pose)
        
        # Convert camera pose to transformation matrix
        camera_matrix = self._rvec_tvec_to_matrix(rvec, tvec)
        
        # Store the poses
        self.robot_poses.append(tcp_matrix)
        self.camera_poses.append(camera_matrix)
        
        print(f"Collected sample {len(self.robot_poses)}:")
        print(f"TCP pose: {tcp_pose}")
        if tvec.ndim > 1:
            print(f"Marker position: {tvec.flatten()}")
        else:
            print(f"Marker position: {tvec}")
        
        return True
        
    def calibrate(self):
        """
        Perform hand-eye calibration using collected samples
        
        Returns:
            bool: True if calibration was successful
        """
        if len(self.robot_poses) < 3:
            print("Need at least 3 poses for calibration")
            return False
            
        try:
            # Convert lists to numpy arrays
            R_gripper2base = [m[:3, :3] for m in self.robot_poses]
            t_gripper2base = [m[:3, 3] for m in self.robot_poses]
            R_target2cam = [m[:3, :3] for m in self.camera_poses]
            t_target2cam = [m[:3, 3] for m in self.camera_poses]
            
            # Perform calibration using OpenCV's hand-eye calibration
            try:
                # Try the newer API first
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=cv2.CALIB_HAND_EYE_TSAI
                )
            except AttributeError:
                # Fallback for older OpenCV versions
                print("Using alternative hand-eye calibration method...")
                # You might need to implement alternative method or use different parameters
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam
                )
            
            # Store result as 4x4 transformation matrix
            self.eye_hand_matrix = np.eye(4)
            self.eye_hand_matrix[:3, :3] = R_cam2gripper
            self.eye_hand_matrix[:3, 3] = t_cam2gripper.flatten()
            
            print("\nCalibration successful!")
            print("Camera to end-effector transformation:")
            print(self.eye_hand_matrix)
            
            return True
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return False
            
    def save_calibration(self, filename):
        """Save calibration data and results to file"""
        if self.eye_hand_matrix is None:
            print("No calibration results to save")
            return False
            
        data = {
            'timestamp': datetime.now().isoformat(),
            'eye_hand_matrix': self.eye_hand_matrix.tolist(),
            'num_samples': len(self.robot_poses),
            'marker_size': self.camera.marker_size,
            'opencv_version': cv2.__version__
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Calibration saved to {filename}")
            return True
        except Exception as e:
            print(f"Failed to save calibration: {e}")
            return False
            
    def load_calibration(self, filename):
        """Load calibration results from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.eye_hand_matrix = np.array(data['eye_hand_matrix'])
            print(f"Loaded calibration from {filename}")
            print("Camera to end-effector transformation:")
            print(self.eye_hand_matrix)
            
            # Print additional info if available
            if 'opencv_version' in data:
                print(f"Calibration was created with OpenCV version: {data['opencv_version']}")
            if 'marker_size' in data:
                print(f"Marker size used: {data['marker_size']} meters")
                
            return True
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False
            
    def verify_calibration(self):
        """
        Verify calibration accuracy by comparing predicted and actual marker positions
        
        Returns:
            float: Average error in meters
        """
        if self.eye_hand_matrix is None:
            print("No calibration data available")
            return None
            
        errors = []
        for i, (robot_pose, camera_pose) in enumerate(zip(self.robot_poses, self.camera_poses)):
            # Calculate the marker position in base frame using two different paths
            
            # Path 1: Through robot kinematics and hand-eye calibration
            marker_pos_path1 = robot_pose @ self.eye_hand_matrix @ camera_pose
            
            # Path 2: Direct calculation (should be the same for all poses if calibration is perfect)
            # For verification, we compare the consistency across all poses
            if i == 0:
                reference_marker_pos = marker_pos_path1
            
            # Compute position error relative to first pose
            error = np.linalg.norm(marker_pos_path1[:3, 3] - reference_marker_pos[:3, 3])
            errors.append(error)
            
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"\nCalibration verification:")
        print(f"Average position error: {avg_error:.4f} meters")
        print(f"Max position error: {max_error:.4f} meters")
        print(f"Number of poses: {len(self.robot_poses)}")
        
        if avg_error < 0.005:  # 5mm
            print("✓ Calibration quality: EXCELLENT")
        elif avg_error < 0.010:  # 10mm
            print("✓ Calibration quality: GOOD")
        elif avg_error < 0.020:  # 20mm
            print("⚠ Calibration quality: ACCEPTABLE")
        else:
            print("✗ Calibration quality: POOR - Consider recalibrating")
        
        return avg_error
        
    @staticmethod
    def _pose_to_matrix(pose):
        """Convert [x,y,z,rx,ry,rz] pose to 4x4 transformation matrix"""
        matrix = np.eye(4)
        
        # Extract position
        matrix[:3, 3] = pose[:3]
        
        # Convert axis-angle to rotation matrix
        rvec = np.array(pose[3:])
        matrix[:3, :3], _ = cv2.Rodrigues(rvec)
        
        return matrix
        
    @staticmethod
    def _rvec_tvec_to_matrix(rvec, tvec):
        """Convert rotation vector and translation vector to 4x4 transformation matrix"""
        matrix = np.eye(4)
        
        # Convert rotation vector to matrix
        matrix[:3, :3], _ = cv2.Rodrigues(rvec)
        
        # Set translation - handle both 1D and 2D arrays
        if tvec.ndim > 1:
            matrix[:3, 3] = tvec.flatten()
        else:
            matrix[:3, 3] = tvec
        
        return matrix

def run_calibration(robot_ip, num_poses=5):
    """
    Run the complete calibration procedure
    
    Args:
        robot_ip (str): IP address of the UR robot
        num_poses (int): Number of different poses to collect
    """
    calibrator = HandEyeCalibrator(robot_ip)
    
    if not calibrator.connect():
        print("Failed to connect to devices")
        return
        
    try:
        print(f"\nStarting hand-eye calibration procedure")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Please move the robot to {num_poses} different poses")
        print("Make sure the ArUco marker (ID 7) is visible in each pose")
        print("Press 'c' to collect sample at current pose")
        print("Press 'q' to finish collection and calculate calibration")
        print("Press 'r' to reset and start over")
        
        while len(calibrator.robot_poses) < num_poses:
            # Show current camera view
            success, _, _, image = calibrator.camera.detect_marker_pose()
            if image is not None:
                # Add status text to image
                status_text = [
                    f"Samples collected: {len(calibrator.robot_poses)}/{num_poses}",
                    f"Marker detected: {'YES' if success else 'NO'}",
                    "Press 'c' to collect, 'q' to finish, 'r' to reset"
                ]
                
                for i, text in enumerate(status_text):
                    color = (0, 255, 0) if success else (0, 0, 255)
                    cv2.putText(image, text, (10, 30 + 30*i), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow('Camera View - Hand-Eye Calibration', image)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print(f"\nCollecting sample {len(calibrator.robot_poses) + 1}/{num_poses}")
                if calibrator.collect_calibration_sample():
                    print("✓ Sample collected successfully")
                    time.sleep(1)  # Wait for robot to stabilize
                else:
                    print("✗ Failed to collect sample - make sure marker is visible")
            elif key == ord('r'):
                print("\nResetting calibration data...")
                calibrator.robot_poses.clear()
                calibrator.camera_poses.clear()
                print("Data reset. Start collecting samples again.")
                
        cv2.destroyAllWindows()
        
        if len(calibrator.robot_poses) >= 3:
            print(f"\nPerforming calibration with {len(calibrator.robot_poses)} poses...")
            if calibrator.calibrate():
                # Verify and save calibration
                calibrator.verify_calibration()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"hand_eye_calibration_{timestamp}.json"
                calibrator.save_calibration(filename)
                print(f"\n✓ Calibration completed and saved as: {filename}")
            else:
                print("\n✗ Calibration failed")
        else:
            print(f"\n✗ Not enough samples collected for calibration (need at least 3, got {len(calibrator.robot_poses)})")
            
    finally:
        calibrator.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ROBOT_IP = "169.254.200.239"  # Replace with your robot's IP
    run_calibration(ROBOT_IP, num_poses=20) 