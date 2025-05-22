#!/usr/bin/env python3
import time
import sys
import math
from URRobotControl import URRobotControl

def main():
    # Replace with your robot's IP address
    robot_ip = "192.168.1.100"  # Example IP, replace with your robot's actual IP
    
    print(f"Connecting to UR robot at {robot_ip}...")
    robot = URRobotControl(robot_ip)
    
    # Connect to the robot
    if not robot.connect():
        print("Failed to connect to the robot")
        return
    
    print("Connected to the robot")
    print(f"Robot status: {robot}")
    
    # Get and display current TCP pose
    tcp_pose = robot.get_tcp_pose()
    if tcp_pose:
        print(f"Current TCP Pose: [{', '.join([f'{x:.3f}' for x in tcp_pose])}]")
    
    # Get and display current joint positions
    joint_positions = robot.get_joint_positions()
    if joint_positions:
        print(f"Current Joint Positions: [{', '.join([f'{j:.3f}' for j in joint_positions])}]")
    
    # Example of moving the robot
    try:
        # Move example 1: Move relative in z-axis (up) by 0.05 meters
        print("\nMoving up 5cm...")
        if robot.move_relative(dz=0.05, v=0.1):
            print("Move completed")
        else:
            print("Move failed")
        time.sleep(1)
        
        # Move example 2: Move relative in z-axis (down) by 0.05 meters
        print("\nMoving down 5cm...")
        if robot.move_relative(dz=-0.05, v=0.1):
            print("Move completed")
        else:
            print("Move failed")
        time.sleep(1)
        
        # Move example 3: Rotate joint 5 (wrist) by 45 degrees
        print("\nRotating wrist joint by 45 degrees...")
        current_joint = joint_positions[5]
        target_angle = current_joint + math.radians(45)
        if robot.move_joint(5, target_angle, v=0.5):
            print("Move completed")
        else:
            print("Move failed")
        time.sleep(1)
        
        # Move example 4: Return joint 5 to original position
        print("\nReturning wrist joint to original position...")
        if robot.move_joint(5, current_joint, v=0.5):
            print("Move completed")
        else:
            print("Move failed")
        time.sleep(1)
        
        # Freedrive mode example
        print("\nEntering freedrive mode for 5 seconds...")
        robot.freedrive_mode(True)
        time.sleep(5)
        print("Exiting freedrive mode...")
        robot.freedrive_mode(False)
        
        # Get and display updated TCP pose after manual movement
        tcp_pose = robot.get_tcp_pose()
        if tcp_pose:
            print(f"\nUpdated TCP Pose: [{', '.join([f'{x:.3f}' for x in tcp_pose])}]")
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"\nError during robot operation: {e}")
    
    # Disconnect from the robot
    print("\nDisconnecting from robot...")
    robot.disconnect()
    print("Disconnected")

if __name__ == "__main__":
    main() 