import numpy as np
import cv2
import time
import json
from datetime import datetime
from pick_n_place.camera_pose_estimation import CameraPoseEstimator
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
            
            # Perform calibration
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
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
            'num_samples': len(self.robot_poses)
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
        for robot_pose, camera_pose in zip(self.robot_poses, self.camera_poses):
            # Predicted marker position through robot chain
            predicted_marker = robot_pose @ self.eye_hand_matrix @ camera_pose
            
            # Actual marker position
            actual_marker = robot_pose @ self.eye_hand_matrix @ camera_pose
            
            # Compute position error
            error = np.linalg.norm(predicted_marker[:3, 3] - actual_marker[:3, 3])
            errors.append(error)
            
        avg_error = np.mean(errors)
        print(f"\nCalibration verification:")
        print(f"Average position error: {avg_error:.3f} meters")
        print(f"Max position error: {np.max(errors):.3f} meters")
        
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
        
        # Set translation
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
        print("\nStarting hand-eye calibration procedure")
        print(f"Please move the robot to {num_poses} different poses")
        print("Press 'c' to collect sample at current pose")
        print("Press 'q' to finish collection and calculate calibration")
        
        while len(calibrator.robot_poses) < num_poses:
            # Show current camera view
            success, _, _, image = calibrator.camera.detect_marker_pose()
            if image is not None:
                cv2.imshow('Camera View', image)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print(f"\nCollecting sample {len(calibrator.robot_poses) + 1}/{num_poses}")
                if calibrator.collect_calibration_sample():
                    print("Sample collected successfully")
                    time.sleep(1)  # Wait for robot to stabilize
                
        if len(calibrator.robot_poses) >= 3:
            print("\nPerforming calibration...")
            if calibrator.calibrate():
                # Verify and save calibration
                calibrator.verify_calibration()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                calibrator.save_calibration(f"hand_eye_calibration_{timestamp}.json")
        else:
            print("Not enough samples collected for calibration")
            
    finally:
        calibrator.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ROBOT_IP = "192.168.1.100"  # Replace with your robot's IP
    run_calibration(ROBOT_IP) 