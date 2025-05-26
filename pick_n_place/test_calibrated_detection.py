import numpy as np
import cv2
from pick_n_place.object_detection import ProductDetector

def test_calibrated_detection(robot_ip, calibration_file):
    """
    Test object detection with hand-eye calibration
    
    Args:
        robot_ip (str): Robot's IP address
        calibration_file (str): Path to saved hand-eye calibration file
    """
    # Initialize detector with calibration
    detector = ProductDetector(
        model_path='best.pt',
        robot_ip=robot_ip,
        calibration_file=calibration_file
    )
    
    if not detector.connect():
        print("Failed to connect to camera or robot")
        return
    
    print("\nStarting calibrated detection test...")
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        while True:
            # Get object pose in both camera and robot frames
            success_cam, cam_pose, _ = detector.get_product_pose_camera_frame()
            success_rob, rob_pose, image = detector.get_product_pose_robot_frame()
            
            if image is not None:
                if success_cam and success_rob:
                    # Display camera frame pose
                    cam_text = [
                        f"Camera Frame:",
                        f"Position (m): ({cam_pose[0]:.3f}, {cam_pose[1]:.3f}, {cam_pose[2]:.3f})",
                        f"Rotation (rad): ({cam_pose[3]:.3f}, {cam_pose[4]:.3f}, {cam_pose[5]:.3f})"
                    ]
                    
                    # Display robot base frame pose
                    rob_text = [
                        f"Robot Base Frame:",
                        f"Position (m): ({rob_pose[0]:.3f}, {rob_pose[1]:.3f}, {rob_pose[2]:.3f})",
                        f"Rotation (rad): ({rob_pose[3]:.3f}, {rob_pose[4]:.3f}, {rob_pose[5]:.3f})"
                    ]
                    
                    # Add text to image
                    for i, text in enumerate(cam_text):
                        cv2.putText(image, text, 
                                  (10, 30 + i*30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    for i, text in enumerate(rob_text):
                        cv2.putText(image, text, 
                                  (10, 120 + i*30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # Draw coordinate axes
                    def draw_axes(image, pose, color):
                        # Create rotation matrix
                        R = cv2.Rodrigues(pose[3:])[0]
                        # Get origin point (position)
                        origin = pose[:3]
                        # Project 3D axes to image
                        axis_length = 0.1  # 10cm axes
                        points_3d = np.float32([[0, 0, 0],
                                              [axis_length, 0, 0],
                                              [0, axis_length, 0],
                                              [0, 0, axis_length]])
                        # Transform points
                        points_3d_transformed = (R @ points_3d.T).T + origin
                        # Project to image
                        camera_matrix = detector.camera.camera_matrix
                        dist_coeffs = detector.camera.dist_coeffs
                        points_2d, _ = cv2.projectPoints(points_3d_transformed,
                                                       np.zeros(3), np.zeros(3),
                                                       camera_matrix, dist_coeffs)
                        points_2d = points_2d.reshape(-1, 2)
                        
                        # Draw axes
                        origin = tuple(map(int, points_2d[0]))
                        point_x = tuple(map(int, points_2d[1]))
                        point_y = tuple(map(int, points_2d[2]))
                        point_z = tuple(map(int, points_2d[3]))
                        
                        cv2.line(image, origin, point_x, (0, 0, 255), 2)  # X axis - Red
                        cv2.line(image, origin, point_y, (0, 255, 0), 2)  # Y axis - Green
                        cv2.line(image, origin, point_z, (255, 0, 0), 2)  # Z axis - Blue
                    
                    # Draw coordinate frames
                    draw_axes(image, cam_pose, (0, 255, 0))  # Camera frame in green
                    draw_axes(image, rob_pose, (255, 0, 0))  # Robot frame in blue
                
                # Show image
                cv2.imshow('Calibrated Detection', image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and success_rob:
                    cv2.imwrite('calibrated_detection.jpg', image)
                    print("\nDetection saved to calibrated_detection.jpg")
                    print("\nCamera Frame Pose:")
                    print(f"Position (m): {cam_pose[:3]}")
                    print(f"Rotation (rad): {cam_pose[3:]}")
                    print("\nRobot Base Frame Pose:")
                    print(f"Position (m): {rob_pose[:3]}")
                    print(f"Rotation (rad): {rob_pose[3:]}")
    
    except KeyboardInterrupt:
        print("\nStopping detection...")
    
    finally:
        detector.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your robot's IP and calibration file
    ROBOT_IP = "192.168.1.100"
    CALIBRATION_FILE = "hand_eye_calibration.json"
    
    test_calibrated_detection(ROBOT_IP, CALIBRATION_FILE) 