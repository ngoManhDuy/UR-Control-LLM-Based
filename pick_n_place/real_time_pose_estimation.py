import numpy as np
import cv2
from cylinder_pose_estimation import CylinderPoseEstimator

class RealTimePoseEstimator:
    def __init__(self, model_path='best.pt'):
        """
        Real-time pose estimator that outputs only YOLO detection box and 6D pose
        """
        self.estimator = CylinderPoseEstimator(model_path)
        
    def connect(self):
        """Connect to camera"""
        return self.estimator.connect()
    
    def disconnect(self):
        """Disconnect from camera"""
        self.estimator.disconnect()
    
    def get_pose_estimation(self):
        """
        Get cylinder pose estimation
        
        Returns:
            dict: {
                'success': bool,
                'detection_box': [x_min, y_min, x_max, y_max] or None,
                'pose_6d': {
                    'position': [x, y, z],           # meters in camera frame
                    'rotation_euler': [rx, ry, rz], # radians
                    'confidence': float              # 0-1
                } or None
            }
        """
        # Run the complete pipeline
        success, pose_6dof = self.estimator.get_cylinder_6dof_pose(
            confidence_threshold=0.3,
            downsample_factor=2,
            max_iterations=500,
            epsilon=0.0005,  # 0.5mm tolerance
            min_inliers=20,
            visualize=False
        )
        
        if success:
            # Get the detection box
            bbox_success, bbox, _, _, _ = self.estimator.detect_cylinder_bbox()
            
            return {
                'success': True,
                'detection_box': bbox if bbox_success else None,
                'pose_6d': {
                    'position': pose_6dof['position'].tolist(),
                    'rotation_euler': pose_6dof['rotation_euler'].tolist(),
                    'confidence': pose_6dof['pose_confidence']
                }
            }
        else:
            return {
                'success': False,
                'detection_box': None,
                'pose_6d': None
            }

    def draw_pose_info(self, image, result):
        """
        Draw bounding box and 6D pose information on the image
        
        Args:
            image: OpenCV image
            result: Result from get_pose_estimation()
            
        Returns:
            image: Annotated image
        """
        if not result['success']:
            # Draw "NO DETECTION" message
            cv2.putText(image, "NO DETECTION", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return image
        
        # Draw bounding box
        if result['detection_box'] is not None:
            bbox = result['detection_box']
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw bbox label
            cv2.putText(image, "CYLINDER", (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw 6D pose information
        pose = result['pose_6d']
        if pose is not None:
            x, y, z = pose['position']
            rx, ry, rz = pose['rotation_euler']
            confidence = pose['confidence']
            
            # Position information (left side)
            y_offset = 50
            cv2.putText(image, "6D POSE:", (10, y_offset), 
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
            
            # Rotation information
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
            
            # Draw coordinate frame at cylinder center if bounding box exists
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
                
                # Z-axis (blue) - pointing into screen, so just a circle
                cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)
                cv2.putText(image, "Z", (center_x - 20, center_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return image

def main():
    """Real-time pose estimation with live display"""
    estimator = RealTimePoseEstimator()
    
    if not estimator.connect():
        print("❌ Failed to connect to camera")
        return
    
    print("🎯 Real-Time Cylinder Pose Estimation")
    print("Press 'q' to quit")
    print("Live display: Bounding box + 6D pose on camera feed")
    
    try:
        while True:
            # Get camera frame
            color_image, _ = estimator.estimator.camera.get_frames()
            if color_image is None:
                continue
            
            # Get pose estimation
            result = estimator.get_pose_estimation()
            
            # Draw pose information on image
            annotated_image = estimator.draw_pose_info(color_image.copy(), result)
            
            # Add title
            cv2.putText(annotated_image, "Real-Time Pose Estimation", 
                       (annotated_image.shape[1]//2 - 150, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show annotated image
            cv2.imshow('Real-Time Pose Estimation', annotated_image)
            
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