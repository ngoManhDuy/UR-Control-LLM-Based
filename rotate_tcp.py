import sys
import os
from URRobotControl import URRobotControl

def rotate_tcp_to_default(robot_ip="169.254.200.239"):

    # Initialize robot control
    robot = URRobotControl(host=robot_ip)
    
    # Connect to robot
    if not robot.connect():
        print("❌ Failed to connect to robot")
        return False
        
    try:
        print("✅ Connected to robot successfully")
        
        # Get current TCP pose
        current_pose = robot.get_tcp_pose()
        if current_pose is None:
            print("❌ Failed to get current TCP pose")
            return False
            
        # Create new pose with current position but default orientation
        target_pose = [
            -0.054,  # Keep current x
            -0.340,  # Keep current y
            0.199,  # Keep current z
            2.9777,          # Default rx
            -0.3917,         # Default ry
            -1.0838          # Default rz
        ]
        
        # Move to target pose
        print("🔄 Rotating TCP to default orientation...")
        success = robot.move_l(pose=target_pose, a=1.2, v=0.25, wait=True)
        
        if success:
            print("✅ TCP rotation completed successfully")
            return True
        else:
            print("❌ Failed to rotate TCP")
            return False
            
    except Exception as e:
        print(f"❌ Error during rotation: {e}")
        return False
        
    finally:
        # Disconnect from robot
        robot.disconnect()
        print("Disconnected from robot")

if __name__ == "__main__":
    print("TCP Rotation to Default Orientation")
    print("=" * 40)
    rotate_tcp_to_default() 