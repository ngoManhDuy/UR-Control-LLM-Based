import numpy as np
import cv2
import json
from cylinder_pose_estimation import CylinderPoseEstimator
from URRobotControl import URRobotControl
import os

# Load hand-eye calibration matrix
CALIB_PATH = os.path.join(os.path.dirname(__file__), 'hand_eye_calibration_best.json')
with open(CALIB_PATH, 'r') as f:
    calib_data = json.load(f)
EYE_HAND_MATRIX = np.array(calib_data['eye_hand_matrix'])  # 4x4

ROBOT_IP = "169.254.200.239"

class RealTimePoseRobotCoordinates:
    def __init__(self, model_path='best.pt', robot_ip=ROBOT_IP):
        self.estimator = CylinderPoseEstimator(model_path)
        self.robot = URRobotControl(robot_ip)
        self.eye_hand_matrix = EYE_HAND_MATRIX

    def connect(self):
        cam_ok = self.estimator.connect()
        rob_ok = self.robot.connect()
        return cam_ok and rob_ok

    def disconnect(self):
        self.estimator.disconnect()
        self.robot.disconnect()

    def get_pose_estimation(self):
        """
        Returns:
            dict: {
                'success': bool,
                'detection_box': [x_min, y_min, x_max, y_max] or None,
                'pose_6d_camera': {...} or None,
                'pose_6d_robot': {...} or None,
                'robot_tcp_pose': [x, y, z, rx, ry, rz] or None
            }
        """
        # Get 6D pose in camera frame
        success, pose_6dof = self.estimator.get_cylinder_6dof_pose(
            confidence_threshold=0.8,
            downsample_factor=2,
            max_iterations=500,
            epsilon=0.0005,
            min_inliers=20,
            visualize=False
        )
        if not success:
            return {
                'success': False,
                'detection_box': None,
                'pose_6d_camera': None,
                'pose_6d_robot': None,
                'robot_tcp_pose': self.robot.get_tcp_pose()
            }
        # Get detection box
        bbox_success, bbox, _, _, _ = self.estimator.detect_cylinder_bbox()
        # Transform pose to robot coordinates
        pose_cam = pose_6dof
        T_cam_obj = np.eye(4)
        T_cam_obj[:3, :3] = pose_cam['rotation_matrix']
        T_cam_obj[:3, 3] = pose_cam['position']
        # Camera->Robot: T_base_obj = T_base_ee @ T_ee_cam @ T_cam_obj
        # But for hand-eye, usually: T_base_obj = T_base_ee @ T_ee_cam @ T_cam_obj
        # Here, we assume T_ee_cam = eye_hand_matrix
        T_ee_obj = self.eye_hand_matrix @ T_cam_obj
        # If you want to show in base frame, you need T_base_ee (robot TCP pose as matrix)
        robot_tcp_pose = self.robot.get_tcp_pose()
        if robot_tcp_pose is not None:
            T_base_ee = self._pose_to_matrix(robot_tcp_pose)
            T_base_obj = T_base_ee @ T_ee_obj
            pos_robot = T_base_obj[:3, 3]
            rot_robot = T_base_obj[:3, :3]
        else:
            pos_robot = None
            rot_robot = None
        pose_6d_robot = None
        if pos_robot is not None and rot_robot is not None:
            pose_6d_robot = {
                'position': pos_robot.tolist(),
                'rotation_matrix': rot_robot.tolist(),
            }
        return {
            'success': True,
            'detection_box': bbox if bbox_success else None,
            'pose_6d_camera': {
                'position': pose_cam['position'].tolist(),
                'rotation_euler': pose_cam['rotation_euler'].tolist(),
                'confidence': pose_cam['pose_confidence']
            },
            'pose_6d_robot': pose_6d_robot,
            'robot_tcp_pose': robot_tcp_pose
        }

    @staticmethod
    def _pose_to_matrix(pose):
        import cv2
        matrix = np.eye(4)
        matrix[:3, 3] = pose[:3]
        rvec = np.array(pose[3:])
        matrix[:3, :3], _ = cv2.Rodrigues(rvec)
        return matrix

    def draw_pose_info(self, image, result):
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
        # Draw camera pose info
        pose = result['pose_6d_camera']
        y_offset = 50
        if pose is not None:
            x, y, z = pose['position']
            rx, ry, rz = pose['rotation_euler']
            confidence = pose['confidence']
            cv2.putText(image, "CAMERA 6D POSE:", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30
            cv2.putText(image, f"X: {x:+.4f}m", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(image, f"Y: {y:+.4f}m", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(image, f"Z: {z:+.4f}m", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 35
            cv2.putText(image, "ROTATION:", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            cv2.putText(image, f"RX: {np.degrees(rx):+.1f}deg", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(image, f"RY: {np.degrees(ry):+.1f}deg", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(image, f"RZ: {np.degrees(rz):+.1f}deg", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw robot pose info
        y_offset += 40
        pose_robot = result['pose_6d_robot']
        if pose_robot is not None:
            x_r, y_r, z_r = pose_robot['position']
            cv2.putText(image, "ROBOT 6D POSE:", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
            cv2.putText(image, f"X: {x_r:+.4f}m", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(image, f"Y: {y_r:+.4f}m", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(image, f"Z: {z_r:+.4f}m", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw robot TCP pose
        tcp = result['robot_tcp_pose']
        if tcp is not None:
            y_offset += 40
            cv2.putText(image, "ROBOT TCP:", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            y_offset += 30
            cv2.putText(image, f"X: {tcp[0]:+.4f}m", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(image, f"Y: {tcp[1]:+.4f}m", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(image, f"Z: {tcp[2]:+.4f}m", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image

def main():
    estimator = RealTimePoseRobotCoordinates()
    if not estimator.connect():
        print("❌ Failed to connect to camera or robot")
        return
    print("Real-Time Cylinder Pose Estimation (Camera & Robot Coordinates)")
    print("Press 'q' to quit")
    print("Live display: Bounding box + 6D pose (camera & robot) on camera feed")
    try:
        while True:
            color_image, _ = estimator.estimator.camera.get_frames()
            if color_image is None:
                continue
            result = estimator.get_pose_estimation()
            # Print to console
            if result['success']:
                print("\n--- OBJECT POSE ---")
                print("Camera frame:")
                print(f"  Position: {result['pose_6d_camera']['position']}")
                print(f"  Euler: {result['pose_6d_camera']['rotation_euler']}")
                print(f"  Confidence: {result['pose_6d_camera']['confidence']:.3f}")
                if result['pose_6d_robot'] is not None:
                    print("Robot frame:")
                    print(f"  Position: {result['pose_6d_robot']['position']}")
            annotated_image = estimator.draw_pose_info(color_image.copy(), result)
            cv2.putText(annotated_image, "Real-Time Pose Estimation (Robot Frame)", 
                        (annotated_image.shape[1]//2 - 200, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Real-Time Pose Estimation (Robot Frame)', annotated_image)
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