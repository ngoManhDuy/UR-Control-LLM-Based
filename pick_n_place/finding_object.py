#!/usr/bin/env python3

import numpy as np
import json
import os
import time
import sys
import threading
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from URRobotControl import URRobotControl
from pick_n_place.cylinder_pose_estimation import CylinderPoseEstimator

class PoseEstimatorWrapper:
    """
    Wrapper for pose estimation that uses an existing robot connection
    to avoid duplicate RTDE connections
    """
    def __init__(self, model_path, robot_instance):
        self.estimator = CylinderPoseEstimator(model_path)
        self.robot = robot_instance  # Use existing robot connection
        
        # Load hand-eye calibration matrix
        calib_path = os.path.join(os.path.dirname(__file__), 'hand_eye_calibration_best.json')
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        self.eye_hand_matrix = np.array(calib_data['eye_hand_matrix'])

    def get_pose_estimation(self):
        """
        Get cylinder pose in both camera and robot coordinates
        Returns same format as RealTimePoseRobotCoordinates.get_pose_estimation()
        """
        # Get 6D pose in camera frame
        success, pose_6dof = self.estimator.get_cylinder_6dof_pose(
            confidence_threshold=0.85,
            downsample_factor=2,
            max_iterations=1000,
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
        
        # Transform to robot frame using hand-eye calibration
        T_ee_obj = self.eye_hand_matrix @ T_cam_obj
        
        # Get current robot TCP pose
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


class ObjectFinder:
    def __init__(self, robot_instance=None, robot_ip="169.254.200.239", model_path='best.pt'):
        """
        Initialize the Object Finder for UR3 robot
        
        Args:
            robot_instance (URRobotControl): Existing robot connection to reuse (preferred)
            robot_ip (str): IP address of the UR robot (used if robot_instance is None)
            model_path (str): Path to YOLO model for cylinder detection
        """
        # Use existing robot instance if provided, otherwise create new one
        if robot_instance is not None:
            self.robot = robot_instance
            self.owns_robot = False  # Don't disconnect this robot when we're done
            print("🔄 Using existing robot connection")
        else:
            # Create single robot connection
            self.robot = URRobotControl(robot_ip)
            self.owns_robot = True  # We own this connection and should disconnect it
            print("🔌 Creating new robot connection")
        
        # Create pose estimator wrapper that uses our robot instance
        self.pose_estimator = PoseEstimatorWrapper(model_path, self.robot)
        
        # Initial scanning pose for pick-and-place [x, y, z, rx, ry, rz]
        self.initial_pose = [0.332, -0.0884, 0.270, 2.154, 1.485, -0.748]
        
        # Base joint scanning parameters
        self.base_joint_index = 0  # Joint 0 is the base rotation joint
        self.scan_range = np.radians(120)  # 120 degrees total scan range
        self.scan_step = np.radians(15)    # 15 degree steps
        self.scan_speed = 0.2              # Joint speed for scanning (rad/s) - SLOWER
        
        # Detection parameters
        self.confidence_threshold = 0.8  # YOLO confidence threshold
        self.detection_timeout = 30        # Maximum scanning time in seconds
        
        # OpenCV display settings
        self.show_camera = False
        self.camera_window_name = "Object Detection - Live Feed"
        self.display_thread = None
        self.display_running = False
        
    def connect(self):
        """Connect to robot and camera systems"""
        # Connect robot only if we own it
        if self.owns_robot:
            robot_ok = self.robot.connect()
            if not robot_ok:
                print("❌ Failed to connect to robot")
        else:
            robot_ok = True  # Assume existing connection is working
        
        # Connect camera only (using estimator)
        camera_ok = self.pose_estimator.estimator.connect()
        
        if not camera_ok:
            print("❌ Failed to connect to camera")
            
        return robot_ok and camera_ok
    
    def disconnect(self):
        """Disconnect from robot and camera systems"""
        # Disconnect camera
        self.pose_estimator.estimator.disconnect()
        # Disconnect robot only if we own it
        if self.owns_robot:
            self.robot.disconnect()
    
    def start_camera_display(self):
        """Start OpenCV camera display in a separate thread"""
        if self.display_running:
            return
        
        # Test camera access before starting display
        try:
            test_frame, _ = self.pose_estimator.estimator.camera.get_frames()
            if test_frame is None:
                print("⚠️ Camera not accessible - skipping display")
                return
        except Exception as e:
            print(f"⚠️ Camera test failed - skipping display: {e}")
            return
            
        self.show_camera = True
        self.display_running = True
        self.display_thread = threading.Thread(target=self._camera_display_loop, daemon=True)
        self.display_thread.start()
        print("📺 Camera display started")
    
    def stop_camera_display(self):
        """Stop OpenCV camera display"""
        self.show_camera = False
        self.display_running = False
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        
        # Safely destroy window with error handling
        try:
            # Check if window exists before trying to destroy it
            if cv2.getWindowProperty(self.camera_window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(self.camera_window_name)
        except cv2.error:
            # Window doesn't exist or already destroyed - this is fine
            pass
        except Exception as e:
            # Any other error - just print but don't crash
            print(f"⚠️ Window cleanup warning: {e}")
        
        print("📺 Camera display stopped")
    
    def _camera_display_loop(self):
        """Camera display loop running in separate thread"""
        window_created = False
        
        while self.display_running:
            try:
                # Get camera frame
                color_image, _ = self.pose_estimator.estimator.camera.get_frames()
                if color_image is None:
                    time.sleep(0.1)
                    continue
                
                # Get current detection result for display
                detection_result = self.pose_estimator.get_pose_estimation()
                
                # Annotate the image
                annotated_image = self._draw_detection_info(color_image.copy(), detection_result)
                
                # Create window on first frame
                if not window_created:
                    cv2.namedWindow(self.camera_window_name, cv2.WINDOW_AUTOSIZE)
                    window_created = True
                
                # Show the image
                cv2.imshow(self.camera_window_name, annotated_image)
                
                # Check for window close or 'q' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                # Check if window was closed by user
                try:
                    if cv2.getWindowProperty(self.camera_window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    # Window was destroyed externally
                    break
                    
                time.sleep(0.05)  # ~20 FPS
                
            except Exception as e:
                print(f"📺 Camera display error: {e}")
                break
        
        # Cleanup - only destroy if we created it
        if window_created:
            try:
                cv2.destroyWindow(self.camera_window_name)
            except cv2.error:
                # Window already destroyed - this is fine
                pass
    
    def _draw_detection_info(self, image, detection_result):
        """Draw detection information on the camera image"""
        # Add title
        cv2.putText(image, "Object Detection - Live Feed", 
                   (image.shape[1]//2 - 150, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Get current base angle for display
        current_angle = self.get_current_base_angle()
        if current_angle is not None:
            cv2.putText(image, f"Base Angle: {current_angle:.1f}°", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if not detection_result['success']:
            cv2.putText(image, "NO DETECTION", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "Scanning for objects...", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return image
        
        # Draw bounding box if available
        if detection_result['detection_box'] is not None:
            bbox = detection_result['detection_box']
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, "OBJECT", (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw detection confidence
        if detection_result['pose_6d_camera'] is not None:
            confidence = detection_result['pose_6d_camera']['confidence']
            color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 255, 255)
            cv2.putText(image, f"Confidence: {confidence:.3f}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw camera pose info
        if detection_result['pose_6d_camera'] is not None:
            pose = detection_result['pose_6d_camera']
            x, y, z = pose['position']
            y_offset = 130
            cv2.putText(image, "CAMERA POSE:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 25
            cv2.putText(image, f"X: {x:+.4f}m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(image, f"Y: {y:+.4f}m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(image, f"Z: {z:+.4f}m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw robot pose info if available
        if detection_result['pose_6d_robot'] is not None:
            pose_robot = detection_result['pose_6d_robot']
            x_r, y_r, z_r = pose_robot['position']
            y_offset += 35
            cv2.putText(image, "ROBOT POSE:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            cv2.putText(image, f"X: {x_r:+.4f}m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(image, f"Y: {y_r:+.4f}m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(image, f"Z: {z_r:+.4f}m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return image

    def move_to_initial_pose(self):
        """
        Move robot to initial scanning pose
        
        Returns:
            bool: True if movement successful, False otherwise
        """
        print(f" Moving to initial scanning pose: {self.initial_pose}")
        
        success = self.robot.move_l(
            pose=self.initial_pose,
            a=0.8,  # Tool acceleration [m/s^2] - SLOWER
            v=0.1,  # Tool speed [m/s] - SLOWER
            wait=True
        )
        
        if success:
            print("✅ Robot moved to initial pose successfully")
            # Wait for robot to stabilize
            time.sleep(2)
        else:
            print("❌ Failed to move robot to initial pose")
            
        return success
    
    def detect_object_at_current_position(self):
        """
        Try to detect an object at the current robot position without moving
        
        Returns:
            dict: {
                'success': bool,
                'object_position': [x, y, z] or None,  # Robot coordinates
                'confidence': float or None,           # Detection confidence
                'current_pose': [x, y, z, rx, ry, rz]  # Current robot pose
            }
        """
        print("🔍 Detecting object at current position...")
        
        # Attempt object detection and pose estimation
        detection_result = self.pose_estimator.get_pose_estimation()
        current_pose = self.robot.get_tcp_pose()
        
        if detection_result['success'] and detection_result['pose_6d_robot'] is not None:
            # Calculate combined confidence (YOLO + pose estimation)
            yolo_confidence = detection_result['pose_6d_camera']['confidence']
            combined_confidence = yolo_confidence * 0.8  # Weight YOLO confidence heavily
            
            print(f"🎯 Object detected! Confidence: {combined_confidence:.3f}")
            print(f"   Robot coordinates: {detection_result['pose_6d_robot']['position']}")
            
            if combined_confidence >= self.confidence_threshold:
                return {
                    'success': True,
                    'object_position': detection_result['pose_6d_robot']['position'],
                    'confidence': combined_confidence,
                    'current_pose': current_pose
                }
            else:
                print(f"⚠️ Object detected but confidence {combined_confidence:.3f} below threshold {self.confidence_threshold}")
                return {
                    'success': False,
                    'object_position': detection_result['pose_6d_robot']['position'],
                    'confidence': combined_confidence,
                    'current_pose': current_pose
                }
        else:
            print("❌ No object detected at current position")
            return {
                'success': False,
                'object_position': None,
                'confidence': None,
                'current_pose': current_pose
            }

    def detect_object_with_custom_confidence(self, confidence_threshold=0.7):
        """
        Try to detect an object at the current robot position with custom confidence threshold
        Used specifically for double-check detection with lower threshold
        
        Args:
            confidence_threshold (float): Custom confidence threshold (default: 0.7 for double-check)
        
        Returns:
            dict: {
                'success': bool,
                'object_position': [x, y, z] or None,  # Robot coordinates
                'confidence': float or None,           # Detection confidence
                'current_pose': [x, y, z, rx, ry, rz]  # Current robot pose
            }
        """
        print(f"🔍 Detecting object with {confidence_threshold*100:.0f}% confidence threshold...")
        
        # Attempt object detection and pose estimation
        detection_result = self.pose_estimator.get_pose_estimation()
        current_pose = self.robot.get_tcp_pose()
        
        if detection_result['success'] and detection_result['pose_6d_robot'] is not None:
            # Calculate combined confidence (YOLO + pose estimation)
            yolo_confidence = detection_result['pose_6d_camera']['confidence']
            combined_confidence = yolo_confidence * 0.8  # Weight YOLO confidence heavily
            
            print(f"🎯 Object detected! Confidence: {combined_confidence:.3f}")
            print(f"   Robot coordinates: {detection_result['pose_6d_robot']['position']}")
            
            if combined_confidence >= confidence_threshold:
                return {
                    'success': True,
                    'object_position': detection_result['pose_6d_robot']['position'],
                    'confidence': combined_confidence,
                    'current_pose': current_pose
                }
            else:
                print(f"⚠️ Object detected but confidence {combined_confidence:.3f} below threshold {confidence_threshold}")
                return {
                    'success': False,
                    'object_position': detection_result['pose_6d_robot']['position'],
                    'confidence': combined_confidence,
                    'current_pose': current_pose
                }
        else:
            print("❌ No object detected at current position")
            return {
                'success': False,
                'object_position': None,
                'confidence': None,
                'current_pose': current_pose
            }

    def rotate_base_to_angle(self, angle_degrees):
        """
        Rotate base joint to a specific angle in degrees
        
        Args:
            angle_degrees (float): Target angle in degrees
            
        Returns:
            bool: True if rotation successful, False otherwise
        """
        angle_radians = np.radians(angle_degrees)
        print(f"🔄 Rotating base to {angle_degrees:.1f}°")
        
        success = self.robot.move_joint(
            joint_index=self.base_joint_index,
            angle=angle_radians,
            a=0.8,          # Joint acceleration [rad/s^2] - SLOWER
            v=self.scan_speed,  # Joint speed [rad/s]
            wait=True
        )
        
        if success:
            print(f"✅ Base rotated to {angle_degrees:.1f}°")
            time.sleep(1)  # Stabilize
        else:
            print(f"❌ Failed to rotate base to {angle_degrees:.1f}°")
            
        return success

    def get_current_base_angle(self):
        """
        Get current base joint angle in degrees
        
        Returns:
            float: Current base angle in degrees, or None if failed
        """
        current_joints = self.robot.get_joint_positions()
        if current_joints is None:
            print("❌ Failed to get current joint positions")
            return None
        
        base_angle_degrees = np.degrees(current_joints[self.base_joint_index])
        return base_angle_degrees

    def scan_for_object(self):
        """
        DEPRECATED: Use continuous_object_search() instead.
        This function is kept for backward compatibility but redirects to the better method.
        
        Returns:
            tuple: (success, object_robot_coordinates, detection_confidence)
        """
        print("⚠️ DEPRECATED: scan_for_object() is deprecated. Use find_object() instead.")
        print("🔄 Redirecting to continuous object search...")
        
        result = self.find_object()
        
        if result['success']:
            return True, result['object_position'], result['confidence']
        else:
            return False, None, result.get('confidence')

    def move_to_object_xy(self, object_position):
        """
        Move robot to object's X,Y coordinates while maintaining current Z, RX, RY, RZ
        
        Args:
            object_position (list): [x, y, z] coordinates in robot frame
            
        Returns:
            bool: True if movement successful, False otherwise
        """
        target_x, target_y, detected_z = object_position
        
        # Get current robot pose to preserve Z, RX, RY, RZ
        current_pose = self.robot.get_tcp_pose()
        if current_pose is None:
            print("❌ Failed to get current robot pose")
            return False
        
        # Create target pose: only change X,Y, keep current Z, RX, RY, RZ
        target_pose = [
            target_x,           # Move to object X
            target_y,           # Move to object Y  
            current_pose[2],    # Keep current Z
            current_pose[3],    # Keep current RX
            current_pose[4],    # Keep current RY
            current_pose[5]     # Keep current RZ
        ]
        
        print(f"🎯 Moving to object X,Y coordinates: [{target_x:.4f}, {target_y:.4f}]")
        print(f"   Keeping current Z, RX, RY, RZ: [{current_pose[2]:.4f}, {current_pose[3]:.4f}, {current_pose[4]:.4f}, {current_pose[5]:.4f}]")
        
        success = self.robot.move_l(
            pose=target_pose,
            a=0.8,  # Tool acceleration [m/s^2] - SLOWER
            v=0.08, # Tool speed [m/s] - MUCH SLOWER for precision
            wait=True
        )
        
        if success:
            print("✅ Successfully moved to object X,Y position")
            time.sleep(1)  # Stabilize
        else:
            print("❌ Failed to move to object X,Y position")
            
        return success
    
    def find_object(self, scan_range_degrees=180, scan_speed=0.1, check_interval=0.5):
        """
        Find objects using continuous scanning approach.
        The robot rotates continuously and checks for objects at regular intervals,
        stopping when an object is found and moving to the object's X,Y position.
        
        Args:
            scan_range_degrees (float): Total range to scan in degrees (default: 180°)
            scan_speed (float): Continuous rotation speed in rad/s (default: 0.1)
            check_interval (float): Time interval between object checks in seconds (default: 0.5)
        
        Returns:
            dict: {
                'success': bool,
                'object_position': [x, y, z] or None,
                'confidence': float or None,
                'final_pose': [x, y, z, rx, ry, rz] or None,
                'scan_time': float,
                'found_at_angle': float or None,
                'error': str (if success=False)
            }
        """
        print("=" * 70)
        print("🚀 STARTING OBJECT SEARCH")
        print("=" * 70)
        print(f"📐 Scan range: ±{scan_range_degrees/2}°")
        print(f"🔄 Scan speed: {scan_speed} rad/s")
        print(f"⏱️ Check interval: {check_interval}s")
        
        # Start camera display (non-critical - don't fail if it doesn't work)
        try:
            self.start_camera_display()
        except Exception as e:
            print(f"⚠️ Camera display failed to start: {e}")
        
        try:
            result = self._execute_object_search(scan_range_degrees, scan_speed, check_interval)
            return result
        finally:
            # Always try to stop camera display (non-critical)
            try:
                self.stop_camera_display()
            except Exception as e:
                print(f"⚠️ Camera display cleanup warning: {e}")
    
    def _execute_object_search(self, scan_range_degrees, scan_speed, check_interval):
        """Execute the actual object search logic"""
        # Step 1: Move to initial scanning pose
        print("\n📍 STEP 1: Moving to initial scanning pose")
        if not self.move_to_initial_pose():
            return {
                'success': False,
                'object_position': None,
                'confidence': None,
                'final_pose': None,
                'scan_time': 0,
                'found_at_angle': None,
                'error': 'Failed to move to initial pose'
            }
        
        # Step 2: Check current position first
        print("\n🔍 STEP 2: Checking current position first")
        detection = self.detect_object_at_current_position()
        
        if detection['success']:
            print(f"✅ Object found at current position! Confidence: {detection['confidence']:.3f}")
            print("🎯 Moving to object position...")
            
            if self.move_to_object_xy(detection['object_position']):
                print("\n" + "=" * 70)
                print("🎉 OBJECT SEARCH SUCCESSFUL!")
                print("=" * 70)
                print(f"📐 Found at current angle: {self.get_current_base_angle():.1f}°")
                print(f"📍 Object position: {detection['object_position']}")
                print(f"🎯 Detection confidence: {detection['confidence']:.3f}")
                print("🎯 Robot positioned at object X,Y coordinates")
                print("=" * 70)
                
                return {
                    'success': True,
                    'object_position': detection['object_position'],
                    'confidence': detection['confidence'],
                    'final_pose': self.robot.get_tcp_pose(),
                    'scan_time': 0,
                    'found_at_angle': self.get_current_base_angle()
                }
        
        print("❌ No object at current position. Starting continuous scanning...")
        
        # Step 3: Setup continuous scanning
        print(f"\n🔄 STEP 3: Starting continuous base rotation")
        print("=" * 50)
        
        # Get current base angle and setup scan range
        current_angle = self.get_current_base_angle()
        if current_angle is None:
            return {
                'success': False,
                'object_position': None,
                'confidence': None,
                'final_pose': self.robot.get_tcp_pose(),
                'scan_time': 0,
                'found_at_angle': None,
                'error': 'Failed to get current base angle'
            }
        
        start_angle = current_angle - scan_range_degrees/2
        end_angle = current_angle + scan_range_degrees/2
        
        print(f"📐 Scan range: {start_angle:.1f}° → {end_angle:.1f}°")
        
        # Shared variables for threading
        object_found = threading.Event()
        best_detection = {'success': False, 'confidence': 0, 'position': None, 'angle': None}
        scan_start_time = time.time()
        
        def detection_loop():
            """Continuously check for objects while robot is moving"""
            check_count = 0
            while not object_found.is_set():
                check_count += 1
                current_scan_angle = self.get_current_base_angle()
                
                print(f"🔍 Check #{check_count}: Scanning at {current_scan_angle:.1f}°")
                
                # Quick object detection
                detection = self.detect_object_at_current_position()
                
                if detection['success']:
                    print(f"🎯 OBJECT FOUND at {current_scan_angle:.1f}°! Confidence: {detection['confidence']:.3f}")
                    
                    # Stop the robot immediately
                    self.robot.stop()
                    time.sleep(0.2)  # Brief pause for robot to stop
                    
                    # Update best detection and signal found
                    best_detection['success'] = True
                    best_detection['confidence'] = detection['confidence']
                    best_detection['position'] = detection['object_position']
                    best_detection['angle'] = current_scan_angle
                    
                    object_found.set()
                    return
                    
                elif detection['confidence'] is not None:
                    print(f"🔍 Low confidence at {current_scan_angle:.1f}°: {detection['confidence']:.3f}")
                    
                    # Keep track of best low-confidence detection
                    if detection['confidence'] > best_detection['confidence']:
                        best_detection['confidence'] = detection['confidence']
                        best_detection['position'] = detection['object_position']
                        best_detection['angle'] = current_scan_angle
                        print(f"📝 Best low-confidence detection so far")
                else:
                    print(f"❌ No object at {current_scan_angle:.1f}°")
                
                # Wait before next check
                time.sleep(check_interval)
        
        # Start the detection thread
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        try:
            # Move continuously through scan range
            print(f"🔄 Starting continuous rotation from {start_angle:.1f}° to {end_angle:.1f}°")
            
            # Move to start position
            if not self.rotate_base_to_angle(start_angle):
                object_found.set()  # Stop detection thread
                return {
                    'success': False,
                    'object_position': None,
                    'confidence': None,
                    'final_pose': self.robot.get_tcp_pose(),
                    'scan_time': time.time() - scan_start_time,
                    'found_at_angle': None,
                    'error': 'Failed to move to scan start position'
                }
            
            # Start continuous movement to end position
            print("🔄 Continuous scanning in progress... (will stop when object found)")
            
            # Use joint movement with the specified speed
            success = self.robot.move_joint(
                joint_index=self.base_joint_index,
                angle=np.radians(end_angle),
                a=0.5,  # Moderate acceleration
                v=scan_speed,  # Continuous speed
                wait=False  # Don't wait - move continuously
            )
            
            if not success:
                object_found.set()
                return {
                    'success': False,
                    'object_position': None,
                    'confidence': None,
                    'final_pose': self.robot.get_tcp_pose(),
                    'scan_time': time.time() - scan_start_time,
                    'found_at_angle': None,
                    'error': 'Failed to start continuous movement'
                }
            
            # Wait for object to be found or scan to complete
            max_scan_time = abs(end_angle - start_angle) / np.degrees(scan_speed) + 10  # Add 10s buffer
            
            if object_found.wait(timeout=max_scan_time):
                # Object found!
                scan_time = time.time() - scan_start_time
                
                if best_detection['success']:
                    print(f"\n🎯 Moving to detected object at angle {best_detection['angle']:.1f}°")
                    
                    if self.move_to_object_xy(best_detection['position']):
                        print("\n" + "=" * 70)
                        print("🎉 OBJECT SEARCH SUCCESSFUL!")
                        print("=" * 70)
                        print(f"📐 Found at angle: {best_detection['angle']:.1f}°")
                        print(f"📍 Object position: {best_detection['position']}")
                        print(f"🎯 Detection confidence: {best_detection['confidence']:.3f}")
                        print(f"⏱️ Total scan time: {scan_time:.1f}s")
                        print("🎯 Robot positioned at object X,Y coordinates")
                        print("=" * 70)
                        
                        return {
                            'success': True,
                            'object_position': best_detection['position'],
                            'confidence': best_detection['confidence'],
                            'final_pose': self.robot.get_tcp_pose(),
                            'scan_time': scan_time,
                            'found_at_angle': best_detection['angle']
                        }
                    else:
                        return {
                            'success': False,
                            'object_position': best_detection['position'],
                            'confidence': best_detection['confidence'],
                            'final_pose': self.robot.get_tcp_pose(),
                            'scan_time': scan_time,
                            'found_at_angle': best_detection['angle'],
                            'error': 'Found object but failed to move to it'
                        }
            else:
                # Scan completed without finding object
                self.robot.stop()
                scan_time = time.time() - scan_start_time
                
                print(f"\n📊 SCAN COMPLETE: {scan_time:.1f}s")
                
                if best_detection['confidence'] > 0:
                    print(f"⚠️ Best detection found:")
                    print(f"📐 Best angle: {best_detection['angle']:.1f}°")
                    print(f"🎯 Best confidence: {best_detection['confidence']:.3f} (below threshold {self.confidence_threshold})")
                    
                    return {
                        'success': False,
                        'object_position': best_detection['position'],
                        'confidence': best_detection['confidence'],
                        'final_pose': self.robot.get_tcp_pose(),
                        'scan_time': scan_time,
                        'found_at_angle': best_detection['angle'],
                        'error': f'Best detection confidence {best_detection["confidence"]:.3f} below threshold {self.confidence_threshold}'
                    }
                else:
                    print("😞 No objects detected in entire scan range")
                    
                    return {
                        'success': False,
                        'object_position': None,
                        'confidence': None,
                        'final_pose': self.robot.get_tcp_pose(),
                        'scan_time': scan_time,
                        'found_at_angle': None,
                        'error': 'No objects detected during scan'
                    }
                    
        except Exception as e:
            object_found.set()  # Stop detection thread
            self.robot.stop()  # Stop robot movement
            scan_time = time.time() - scan_start_time
            
            print(f"💥 Error during scanning: {e}")
            return {
                'success': False,
                'object_position': None,
                'confidence': None,
                'final_pose': self.robot.get_tcp_pose(),
                'scan_time': scan_time,
                'found_at_angle': None,
                'error': f'Scanning error: {str(e)}'
            }
        
        finally:
            # Always stop camera display when done
            self.stop_camera_display()

    def move_down_to_pick(self, initial_object_position, initial_confidence=None):
        """
        Move down to pick up an object after it has been found and positioned.
        Performs double-check verification by moving down 100mm, detecting again,
        and positioning precisely for picking.
        
        Args:
            initial_object_position (list): [x, y, z] position of detected object
            initial_confidence (float, optional): Initial detection confidence
            
        Returns:
            dict: {
                'success': bool,
                'final_position': [x, y, z] or None,
                'final_confidence': float or None,
                'pick_pose': [x, y, z, rx, ry, rz] or None,
                'error': str (if applicable)
            }
        """
        print("=" * 70)
        print("🔍 STARTING MOVE-DOWN-TO-PICK OPERATION")
        print("=" * 70)
        print(f"📍 Initial object position: {initial_object_position}")
        if initial_confidence:
            print(f"🎯 Initial confidence: {initial_confidence:.3f}")
        
        # Execute the move-down-to-pick operation (camera display already running from find_object)
        result = self._execute_move_down_to_pick(initial_object_position, initial_confidence)
        return result
    
    def _execute_move_down_to_pick(self, initial_object_position, initial_confidence):
        """Execute the actual move-down-to-pick logic"""
        try:
            # Get current pose
            current_pose = self.robot.get_tcp_pose()
            if current_pose is None:
                return {
                    'success': False,
                    'final_position': None,
                    'final_confidence': None,
                    'pick_pose': None,
                    'error': 'Failed to get current robot pose'
                }
            
            # Move down 100mm (0.1m) in Z axis
            double_check_pose = current_pose.copy()
            double_check_pose[2] -= 0.10  # Move down 100mm
            
            print(f"\n📉 Moving down 100mm for double-check verification")
            print(f"   From Z: {current_pose[2]:.4f}m → To Z: {double_check_pose[2]:.4f}m")
            
            success = self.robot.move_l(
                pose=double_check_pose,
                a=0.8,  # Tool acceleration [m/s^2]
                v=0.05, # Slow tool speed for precision [m/s]
                wait=True
            )
            
            if not success:
                return {
                    'success': False,
                    'final_position': None,
                    'final_confidence': None,
                    'pick_pose': None,
                    'error': 'Failed to move down for double-check'
                }
            
            # Stabilize and detect again
            time.sleep(1)
            print("\n🔍 Performing double-check detection...")
            
            double_check_detection = self.detect_object_with_custom_confidence(0.5)
            
            if double_check_detection['success']:
                print(f"✅ Double-check detection successful!")
                print(f"   Confidence: {double_check_detection['confidence']:.3f}")
                print(f"   Refined position: {double_check_detection['object_position']}")
                
                # Move to the new detected X,Y position only (keep current Z)
                print("\n🎯 Moving to refined object X,Y position (keeping current Z)...")
                
                # Get current pose to preserve the current Z (which is already 100mm lower)
                current_pose = self.robot.get_tcp_pose()
                if current_pose is None:
                    return {
                        'success': False,
                        'final_position': double_check_detection['object_position'],
                        'final_confidence': double_check_detection['confidence'],
                        'pick_pose': None,
                        'error': 'Failed to get current pose for refined positioning'
                    }
                
                # Create refined target pose: new X,Y from detection, current Z, orientations
                refined_pose = [
                    double_check_detection['object_position'][0],  # New X from detection
                    double_check_detection['object_position'][1],  # New Y from detection
                    current_pose[2],    # Keep current Z (already 100mm lower)
                    current_pose[3],    # Keep current RX
                    current_pose[4],    # Keep current RY
                    current_pose[5]     # Keep current RZ
                ]
                
                print(f"   Moving to refined X,Y: [{refined_pose[0]:.4f}, {refined_pose[1]:.4f}]")
                print(f"   Keeping current Z: {refined_pose[2]:.4f}m")
                
                success = self.robot.move_l(
                    pose=refined_pose,
                    a=0.8,  # Tool acceleration [m/s^2]
                    v=0.05, # Slow tool speed for precision [m/s]
                    wait=True
                )
                
                if success:
                    print("✅ Successfully moved to refined X,Y position")
                    
                    # Final positioning step: Move to pick position with offset
                    print("\n🎯 Moving to final pick position...")
                    
                    # Get current pose for final positioning
                    final_current_pose = self.robot.get_tcp_pose()
                    if final_current_pose is None:
                        return {
                            'success': False,
                            'final_position': double_check_detection['object_position'],
                            'final_confidence': double_check_detection['confidence'],
                            'pick_pose': None,
                            'error': 'Failed to get current pose for final positioning'
                        }
                    
                    # Final pick position with precise offsets
                    final_pick_pose = [
                        double_check_detection['object_position'][0] + 0.012,  # X offset
                        -0.35,  # Y offset: -350mm
                        double_check_detection['object_position'][2] - 0.034 , # Z offset
                        final_current_pose[3],  # Keep current RX
                        final_current_pose[4],  # Keep current RY
                        final_current_pose[5]   # Keep current RZ
                    ]
                    
                    object_z_mm = double_check_detection['object_position'][2] * 1000
                    final_z_mm = final_pick_pose[2] * 1000
                    print(f"   Final offsets: X+12mm, Y-350mm, Z-27mm")
                    print(f"   Object Z: {object_z_mm:.1f}mm → Pick Z: {final_z_mm:.1f}mm")
                    
                    final_success = self.robot.move_l(
                        pose=final_pick_pose,
                        a=0.8,  # Tool acceleration [m/s^2]
                        v=0.05, # Slow tool speed for precision [m/s]
                        wait=True
                    )
                    
                    if final_success:
                        print("✅ Successfully positioned at final pick location")
                        print("\n" + "=" * 70)
                        print("🎉 MOVE-DOWN-TO-PICK SUCCESSFUL!")
                        print("=" * 70)
                        print(f"📍 Final object position: {double_check_detection['object_position']}")
                        print(f"🎯 Final confidence: {double_check_detection['confidence']:.3f}")
                        print(f"🤖 Robot pick pose: {final_pick_pose}")
                        print("🎯 Ready for gripper operation!")
                        print("=" * 70)
                        
                        return {
                            'success': True,
                            'final_position': double_check_detection['object_position'],
                            'final_confidence': double_check_detection['confidence'],
                            'pick_pose': final_pick_pose,
                            'error': None
                        }
                    else:
                        return {
                            'success': False,
                            'final_position': double_check_detection['object_position'],
                            'final_confidence': double_check_detection['confidence'],
                            'pick_pose': None,
                            'error': 'Double-check successful but failed to move to final pick position'
                        }
                    
                else:
                    return {
                        'success': False,
                        'final_position': double_check_detection['object_position'],
                        'final_confidence': double_check_detection['confidence'],
                        'pick_pose': None,
                        'error': 'Double-check detected object but failed to move to refined X,Y position'
                    }
            else:
                # Object not found in double-check - this might be normal if the perspective changed
                print("⚠️ Object not detected in double-check position")
                print("   This might be due to changed perspective from lower height")
                
                # Move back to original detection position
                print("🔄 Moving back to original detection position...")
                if self.move_to_object_xy(initial_object_position):
                    return {
                        'success': False,
                        'final_position': initial_object_position,
                        'final_confidence': double_check_detection.get('confidence'),
                        'pick_pose': None,
                        'error': 'Object not visible from double-check position, returned to original'
                    }
                else:
                    return {
                        'success': False,
                        'final_position': None,
                        'final_confidence': None,
                        'pick_pose': None,
                        'error': 'Double-check failed and could not return to original position'
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'final_position': None,
                'final_confidence': None,
                'pick_pose': None,
                'error': f'Move-down-to-pick error: {str(e)}'
            }
        
        finally:
            print("=" * 70)
            print("🏁 MOVE-DOWN-TO-PICK COMPLETE")
            print("=" * 70)

    # DEPRECATED FUNCTIONS - Remove these in future versions
    def finding_object(self):
        """DEPRECATED: Use find_object() instead"""
        print("⚠️ DEPRECATED: finding_object() is deprecated. Use find_object() instead.")
        return self.find_object()
    
    def complete_object_search(self, scanning_angles=None):
        """DEPRECATED: Use find_object() instead"""
        print("⚠️ DEPRECATED: complete_object_search() is deprecated. Use find_object() instead.")
        return self.find_object()
    
    def continuous_object_search(self, scan_range_degrees=180, scan_speed=0.1, check_interval=0.5):
        """DEPRECATED: Use find_object() instead"""
        print("⚠️ DEPRECATED: continuous_object_search() is deprecated. Use find_object() instead.")
        return self.find_object(scan_range_degrees, scan_speed, check_interval)


def main():
    """
    Example usage of the simplified object finder
    """
    finder = ObjectFinder()
    
    if not finder.connect():
        print("❌ Failed to connect to robot/camera systems")
        return
    
    try:
        # Run the object finding workflow
        result = finder.find_object()
        
        if result['success']:
            print("\n✅ Object finding successful!")
            print(f"Object position (robot frame): {result['object_position']}")
            print(f"Detection confidence: {result['confidence']}")
            print(f"Robot is now positioned at: {result['final_pose']}")
            print("\n🔄 Ready for next step: pick operation")
        else:
            print(f"\n❌ Object finding failed: {result.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Object finding interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
    finally:
        finder.disconnect()
        print("🔌 Disconnected from systems")


if __name__ == "__main__":
    main() 