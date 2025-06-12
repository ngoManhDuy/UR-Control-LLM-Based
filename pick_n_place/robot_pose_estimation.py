import numpy as np
import cv2
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from real_time_pose_estimation import RealTimePoseEstimator
from URRobotControl import URRobotControl

class RobotPoseEstimator:
    def __init__(self, robot_ip, model_path='best.pt', calibration_file='hand_eye_calibration_best.json'):
        """
        Real-time object pose estimation in robot coordinates
        
        Args:
            robot_ip (str): IP address of the UR robot
            model_path (str): Path to YOLO model
            calibration_file (str): Path to hand-eye calibration file
        """
        # Initialize camera pose estimator
        self.camera_estimator = RealTimePoseEstimator(model_path)
        
        # Initialize robot control
        self.robot = URRobotControl(robot_ip)
        
        # Load hand-eye calibration
        self.hand_eye_matrix = None
        self.load_hand_eye_calibration(calibration_file)
        
    def connect(self):
        """Connect to both camera and robot"""
        camera_ok = self.camera_estimator.connect()
        robot_ok = self.robot.connect()
        
        if not camera_ok:
            print("❌ Failed to connect to camera")
        if not robot_ok:
            print("❌ Failed to connect to robot")
            
        return camera_ok and robot_ok
    
    def disconnect(self):
        """Disconnect from both camera and robot"""
        self.camera_estimator.disconnect()
        self.robot.disconnect()
    
    def load_hand_eye_calibration(self, filename):
        """Load hand-eye calibration matrix from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.hand_eye_matrix = np.array(data['eye_hand_matrix'])
            print(f"✅ Loaded hand-eye calibration from {filename}")
            print(f"   Calibration created with {data['num_samples']} samples")
            return True
        except Exception as e:
            print(f"❌ Failed to load hand-eye calibration: {e}")
            return False
    
    def camera_to_robot_pose(self, camera_pose_6d):
        """
        Transform object pose from camera coordinates to robot base coordinates
        
        Args:
            camera_pose_6d: Pose in camera frame {position: [x,y,z], rotation_euler: [rx,ry,rz]}
            
        Returns:
            dict: Pose in robot base frame or None if transformation fails
        """
        if self.hand_eye_matrix is None:
            print("❌ No hand-eye calibration loaded")
            return None
        
        # Get current robot TCP pose
        tcp_pose = self.robot.get_tcp_pose()
        if tcp_pose is None:
            print("❌ Failed to get robot TCP pose")
            return None
        
        # Convert camera pose to 4x4 transformation matrix
        camera_position = np.array(camera_pose_6d['position'])
        camera_rotation = camera_pose_6d['rotation_euler']
        
        # Create 4x4 transformation matrix for object in camera frame
        object_in_camera = np.eye(4)
        object_in_camera[:3, 3] = camera_position
        
        # Convert Euler angles to rotation matrix
        rvec = np.array(camera_rotation)
        object_in_camera[:3, :3], _ = cv2.Rodrigues(rvec)
        
        # Convert TCP pose to 4x4 transformation matrix
        tcp_matrix = self._pose_to_matrix(tcp_pose)
        
        # Transform object pose to robot base frame
        # object_in_base = tcp_in_base @ camera_in_tcp @ object_in_camera
        object_in_base = tcp_matrix @ self.hand_eye_matrix @ object_in_camera
        
        # Extract position and rotation from result
        robot_position = object_in_base[:3, 3].tolist()
        robot_rotation_matrix = object_in_base[:3, :3]
        robot_rotation_euler, _ = cv2.Rodrigues(robot_rotation_matrix)
        robot_rotation_euler = robot_rotation_euler.flatten().tolist()
        
        return {
            'position': robot_position,
            'rotation_euler': robot_rotation_euler,
            'transformation_matrix': object_in_base.tolist()
        }
    
    def get_complete_pose_estimation(self):
        """
        Get object pose in both camera and robot coordinates
        
        Returns:
            dict: Complete pose information
        """
        # Get pose in camera coordinates
        camera_result = self.camera_estimator.get_pose_estimation()
        
        if not camera_result['success']:
            return {
                'success': False,
                'detection_box': None,
                'camera_pose': None,
                'robot_pose': None,
                'tcp_pose': None
            }
        
        # Transform to robot coordinates
        robot_pose = self.camera_to_robot_pose(camera_result['pose_6d'])
        
        # Get current TCP pose for reference
        tcp_pose = self.robot.get_tcp_pose()
        
        return {
            'success': True,
            'detection_box': camera_result['detection_box'],
            'camera_pose': camera_result['pose_6d'],
            'robot_pose': robot_pose,
            'tcp_pose': tcp_pose
        }
    
    def draw_complete_pose_info(self, image, result):
        """
        Draw both camera and robot pose information on image
        
        Args:
            image: OpenCV image
            result: Result from get_complete_pose_estimation()
            
        Returns:
            image: Annotated image
        """
        if not result['success']:
            cv2.putText(image, "NO DETECTION", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return image
        
        # Draw bounding box
        if result['detection_box'] is not None:
            bbox = result['detection_box']
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, "CYLINDER", (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Camera coordinates (left side)
        if result['camera_pose'] is not None:
            camera_pose = result['camera_pose']
            x, y, z = camera_pose['position']
            rx, ry, rz = camera_pose['rotation_euler']
            confidence = camera_pose['confidence']
            
            y_offset = 50
            cv2.putText(image, "CAMERA COORDINATES:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            y_offset += 25
            cv2.putText(image, f"X: {x:+.4f}m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(image, f"Y: {y:+.4f}m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(image, f"Z: {z:+.4f}m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            y_offset += 25
            cv2.putText(image, f"RX: {np.degrees(rx):+.1f}deg", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(image, f"RY: {np.degrees(ry):+.1f}deg", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(image, f"RZ: {np.degrees(rz):+.1f}deg", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Confidence
            confidence_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)
            y_offset += 25
            cv2.putText(image, f"Confidence: {confidence:.3f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, confidence_color, 1)
        
        # Robot coordinates (center)
        if result['robot_pose'] is not None:
            robot_pose = result['robot_pose']
            x, y, z = robot_pose['position']
            rx, ry, rz = robot_pose['rotation_euler']
            
            x_start = 250
            y_offset = 50
            cv2.putText(image, "ROBOT BASE COORDINATES:", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            y_offset += 25
            cv2.putText(image, f"X: {x:+.4f}m", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(image, f"Y: {y:+.4f}m", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(image, f"Z: {z:+.4f}m", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            y_offset += 25
            cv2.putText(image, f"RX: {np.degrees(rx):+.1f}deg", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(image, f"RY: {np.degrees(ry):+.1f}deg", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(image, f"RZ: {np.degrees(rz):+.1f}deg", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # TCP pose (right side)
        if result['tcp_pose'] is not None:
            tcp_pose = result['tcp_pose']
            x, y, z = tcp_pose[:3]
            rx, ry, rz = tcp_pose[3:]
            
            x_start = image.shape[1] - 200
            y_offset = 50
            cv2.putText(image, "TCP POSE:", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            y_offset += 20
            cv2.putText(image, f"X: {x:+.3f}m", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 15
            cv2.putText(image, f"Y: {y:+.3f}m", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 15
            cv2.putText(image, f"Z: {z:+.3f}m", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
            y_offset += 20
            cv2.putText(image, f"RX: {np.degrees(rx):+.1f}deg", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 15
            cv2.putText(image, f"RY: {np.degrees(ry):+.1f}deg", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 15
            cv2.putText(image, f"RZ: {np.degrees(rz):+.1f}deg", (x_start, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Draw coordinate frame at cylinder center
        if result['detection_box'] is not None:
            bbox = result['detection_box']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Draw small coordinate axes
            axis_length = 30
            # X-axis (red)
            cv2.arrowedLine(image, (center_x, center_y), 
                           (center_x + axis_length, center_y), (0, 0, 255), 2)
            cv2.putText(image, "X", (center_x + axis_length + 5, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Y-axis (green)
            cv2.arrowedLine(image, (center_x, center_y), 
                           (center_x, center_y + axis_length), (0, 255, 0), 2)
            cv2.putText(image, "Y", (center_x, center_y + axis_length + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Z-axis (blue)
            cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.putText(image, "Z", (center_x - 20, center_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return image
    
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

def main():
    """Real-time pose estimation in both camera and robot coordinates"""
    ROBOT_IP = "169.254.200.239"  # Replace with your robot's IP
    
    estimator = RobotPoseEstimator(
        robot_ip=ROBOT_IP,
        model_path='best.pt',
        calibration_file='pick_n_place/hand_eye_calibration_best.json'
    )
    
    if not estimator.connect():
        print("❌ Failed to connect to devices")
        return
    
    print("🎯 Real-Time Object Pose Estimation")
    print("📷 Camera Coordinates + 🤖 Robot Base Coordinates")
    print("Press 'q' to quit")
    
    try:
        while True:
            # Get camera frame
            color_image, _ = estimator.camera_estimator.estimator.camera.get_frames()
            if color_image is None:
                continue
            
            # Get complete pose estimation
            result = estimator.get_complete_pose_estimation()
            
            # Draw complete pose information
            annotated_image = estimator.draw_complete_pose_info(color_image.copy(), result)
            
            # Add title
            cv2.putText(annotated_image, "Camera + Robot Coordinates", 
                       (annotated_image.shape[1]//2 - 120, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show annotated image
            cv2.imshow('Robot Pose Estimation', annotated_image)
            
            # Print poses to console occasionally
            if result['success'] and cv2.waitKey(1) & 0xFF == ord('p'):
                print("\n" + "="*60)
                print("📊 CURRENT POSE ESTIMATION")
                print("="*60)
                
                if result['camera_pose']:
                    cam_pos = result['camera_pose']['position']
                    print(f"📷 Camera: [{cam_pos[0]:+.4f}, {cam_pos[1]:+.4f}, {cam_pos[2]:+.4f}]")
                
                if result['robot_pose']:
                    rob_pos = result['robot_pose']['position']
                    print(f"🤖 Robot:  [{rob_pos[0]:+.4f}, {rob_pos[1]:+.4f}, {rob_pos[2]:+.4f}]")
                
                if result['tcp_pose']:
                    tcp_pos = result['tcp_pose'][:3]
                    print(f"🔧 TCP:    [{tcp_pos[0]:+.4f}, {tcp_pos[1]:+.4f}, {tcp_pos[2]:+.4f}]")
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        estimator.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 