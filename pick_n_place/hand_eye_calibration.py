import numpy as np
import cv2
import time
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pick_n_place.cylinder_pose_estimation import CameraPoseEstimator
from URRobotControl import URRobotControl

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