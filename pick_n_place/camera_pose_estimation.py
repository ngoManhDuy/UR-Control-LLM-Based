import numpy as np
import cv2
from pick_n_place.camera import RealSenseCamera
import time

class CameraPoseEstimator:
    def __init__(self, marker_size=0.1):  # marker_size in meters (100mm = 0.1m)
        self.camera = RealSenseCamera()
        self.marker_size = marker_size
        
        # ArUco dictionary for 6x6 markers
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Camera matrix and distortion coefficients will be populated after connection
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def connect(self):
        """Connect to the RealSense camera and get intrinsic parameters"""
        if not self.camera.connect():
            return False
            
        # Get camera intrinsics from RealSense
        if self.camera.intrinsics:
            # Convert RealSense intrinsics to OpenCV camera matrix format
            self.camera_matrix = np.array([
                [self.camera.intrinsics.fx, 0, self.camera.intrinsics.ppx],
                [0, self.camera.intrinsics.fy, self.camera.intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # For RealSense D435i, we can assume zero distortion as it provides rectified images
            self.dist_coeffs = np.zeros(5)
            return True
        return False
        
    def detect_marker_pose(self, marker_id=7):
        """
        Detect ArUco marker and estimate its pose relative to the camera
        
        Args:
            marker_id (int): ID of the marker to detect (default: 7)
            
        Returns:
            tuple: (success, rotation_vector, translation_vector, color_image)
                - success: Boolean indicating if marker was detected
                - rotation_vector: 3x1 rotation vector (Rodriguez rotation)
                - translation_vector: 3x1 translation vector (in meters)
                - color_image: The captured RGB image with marker detection visualization
        """
        if self.camera_matrix is None:
            print("Camera not connected or intrinsics not available")
            return False, None, None, None
            
        # Get frames from RealSense
        color_image, _ = self.camera.get_frames()
        if color_image is None:
            return False, None, None, None
            
        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        # Draw detected markers
        color_image_with_markers = color_image.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image_with_markers, corners, ids)
            
            # Find the marker we're looking for
            if marker_id in ids:
                marker_idx = np.where(ids == marker_id)[0][0]
                marker_corners = corners[marker_idx]
                
                # Estimate pose of the marker
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    marker_corners, self.marker_size, 
                    self.camera_matrix, self.dist_coeffs
                )
                
                # Draw axis for the marker
                cv2.drawFrameAxes(color_image_with_markers, self.camera_matrix, 
                                self.dist_coeffs, rvec[0], tvec[0], self.marker_size/2)
                
                return True, rvec[0], tvec[0], color_image_with_markers
                
        return False, None, None, color_image_with_markers
        
    def disconnect(self):
        """Disconnect from the camera"""
        self.camera.disconnect()

def test_marker_detection():
    """Test the marker detection and pose estimation"""
    estimator = CameraPoseEstimator(marker_size=0.1)  # 100mm marker
    
    if not estimator.connect():
        print("Failed to connect to camera")
        return
        
    print("\nStarting marker detection test...")
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        while True:
            success, rvec, tvec, image = estimator.detect_marker_pose()
            
            if image is not None:
                # Display detection status and pose if marker is found
                if success:
                    # Convert rotation vector to euler angles
                    rmat, _ = cv2.Rodrigues(rvec)
                    euler_angles = cv2.RQDecomp3x3(rmat)[0]
                    
                    # Display pose information
                    pose_text = [
                        f"Translation (x,y,z): {tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f} m",
                        f"Rotation (rx,ry,rz): {euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f} deg"
                    ]
                    
                    # Add text to image
                    for i, text in enumerate(pose_text):
                        cv2.putText(image, text, (10, 30 + 30*i), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show image
                cv2.imshow('ArUco Marker Detection', image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and success:
                    # Save current frame and pose data
                    timestamp = int(time.time())
                    cv2.imwrite(f'marker_detection_{timestamp}.jpg', image)
                    print(f"\nFrame saved as marker_detection_{timestamp}.jpg")
                    print("Marker pose:")
                    print(f"Translation: {tvec}")
                    print(f"Rotation (euler angles): {euler_angles}")
    
    except KeyboardInterrupt:
        print("\nStopping marker detection...")
    
    finally:
        estimator.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_marker_detection() 