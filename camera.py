import pyrealsense2 as rs
import numpy as np
import cv2
import time

class RealSenseCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure the pipeline to stream different resolutions of color and depth frames
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Create align object - align depth frames to color frames
        self.align = rs.align(rs.stream.color)
        
        # Camera intrinsic parameters (will be populated after connection)
        self.intrinsics = None
        self.depth_scale = None
        
    def connect(self):
        """Connect to the RealSense camera and start streaming"""
        try:
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Get the depth sensor's depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth Scale: {self.depth_scale}")
            
            # Get camera intrinsics
            color_profile = profile.get_stream(rs.stream.color)
            self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            print("Camera connected successfully!")
            print(f"Resolution: {self.intrinsics.width} x {self.intrinsics.height}")
            print(f"Focal Length: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}")
            print(f"Principal Point: cx={self.intrinsics.ppx:.2f}, cy={self.intrinsics.ppy:.2f}")
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to camera: {e}")
            return False
    
    def get_frames(self):
        """Get aligned RGB and depth frames"""
        try:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)
            
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                return None, None
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None
    
    def get_3d_point(self, x, y, depth_value):
        """Convert 2D pixel coordinates to 3D world coordinates"""
        if self.intrinsics is None:
            return None
        
        # Convert depth value to meters
        depth_in_meters = depth_value * self.depth_scale
        
        # Use RealSense intrinsics to convert to 3D
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [x, y], depth_in_meters
        )
        
        return point_3d  # Returns [x, y, z] in meters
    
    def disconnect(self):
        """Stop streaming and disconnect"""
        try:
            self.pipeline.stop()
            print("Camera disconnected successfully!")
        except Exception as e:
            print(f"Error disconnecting camera: {e}")

def test_camera():
    """Test the camera connection and display live feed"""
    camera = RealSenseCamera()
    
    if not camera.connect():
        print("Failed to connect to camera. Please check connection.")
        return
    
    print("\nStarting live feed test...")
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        while True:
            # Get frames
            color_image, depth_image = camera.get_frames()
            
            if color_image is None or depth_image is None:
                continue
            
            # Apply colormap on depth image (for visualization)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Stack images horizontally
            images = np.hstack((color_image, depth_colormap))
            
            # Show images
            cv2.imshow('RealSense RGB-D Feed (RGB | Depth)', images)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                cv2.imwrite(f'rgb_frame_{timestamp}.jpg', color_image)
                cv2.imwrite(f'depth_frame_{timestamp}.png', depth_image)
                print(f"Frames saved with timestamp: {timestamp}")
            
            # Test 3D coordinate conversion (center pixel)
            height, width = depth_image.shape
            center_x, center_y = width // 2, height // 2
            center_depth = depth_image[center_y, center_x]
            
            if center_depth > 0:
                point_3d = camera.get_3d_point(center_x, center_y, center_depth)
                if point_3d:
                    # Display 3D coordinates on image
                    cv2.circle(color_image, (center_x, center_y), 5, (0, 255, 0), -1)
                    text = f"3D: ({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})m"
                    cv2.putText(color_image, text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    except KeyboardInterrupt:
        print("\nStopping camera test...")
    
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("RealSense D435i Camera Setup - Step 1")
    print("=" * 40)
    
    # Test camera connection and functionality
    test_camera()