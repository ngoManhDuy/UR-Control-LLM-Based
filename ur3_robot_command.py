#!/usr/bin/env python3
import time
import sys
import math
import argparse
import numpy as np
from URRobotControl import URRobotControl

def display_help():
    """Display help information for the program."""
    print("\n=== UR Robot Control Program ===")
    print("Available commands:")
    print("  pose                - Show current robot pose (6D coordinates)")
    print("  joints              - Show current joint positions")
    print("  move x y z rx ry rz - Move robot to specified 6D pose")
    print("  movej j1 j2 j3 j4 j5 j6 - Move robot to specified joint positions (in degrees)")
    print("  rotate joint angle  - Rotate specific joint to angle (in degrees)")
    print("  open                - Open gripper")
    print("  close               - Close gripper")
    print("  relative dx dy dz [drx dry  drz] - Move relative to current position")
    print("  freedrive on/off    - Enable/disable freedrive mode")
    print("  help                - Show this help message")
    print("  exit                - Exit the program")
    print("\nExamples:")
    print("  move 0.5 0.2 0.3 0 3.14 0")
    print("  rotate 5 45")
    print("  relative 0 0 0.05")

def parse_command(command, robot):
    """Parse and execute a command."""
    parts = command.lower().strip().split()
    
    if not parts:
        return
    
    cmd = parts[0]
    
    try:
        if cmd == "pose":
            # Show current robot pose
            pose = robot.get_tcp_pose()
            if pose is not None:
                print(f"Current TCP Pose [x, y, z, rx, ry, rz]:")
                print(f"  Position: [{pose[0]:.5f}, {pose[1]:.5f}, {pose[2]:.5f}] meters")
                print(f"  Rotation: [{pose[3]:.5f}, {pose[4]:.5f}, {pose[5]:.5f}] radians")
                print(f"  Rotation: [{math.degrees(pose[3]):.2f}, {math.degrees(pose[4]):.2f}, {math.degrees(pose[5]):.2f}] degrees")
            else:
                print("Failed to get current pose")
                
        elif cmd == "joints":
            # Show current joint positions
            joints = robot.get_joint_positions()
            if joints is not None:
                print("Current Joint Positions [j0, j1, j2, j3, j4, j5]:")
                print(f"  Radians: [{', '.join([f'{j:.5f}' for j in joints])}]")
                print(f"  Degrees: [{', '.join([f'{math.degrees(j):.2f}' for j in joints])}]")
            else:
                print("Failed to get current joint positions")
                
        elif cmd == "move" and len(parts) >= 7:
            # Move to absolute position
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                rx = float(parts[4])
                ry = float(parts[5])
                rz = float(parts[6])
                
                # Get speed and acceleration if provided
                a = float(parts[7]) if len(parts) > 7 else 1.2
                v = float(parts[8]) if len(parts) > 8 else 0.25
                
                print(f"Moving to pose: [{x}, {y}, {z}, {rx}, {ry}, {rz}]...")
                result = robot.move_l(pose=[x, y, z, rx, ry, rz], a=a, v=v)
                if isinstance(result, bool):
                    if result:
                        print("Move completed successfully")
                    else:
                        print("Move failed")
                else:
                    print("Move command sent")
            except ValueError:
                print("Invalid coordinates. Please provide 6 numeric values for x, y, z, rx, ry, rz")
                
        elif cmd == "movej" and len(parts) >= 7:
            # Move to joint positions (input in degrees, convert to radians)
            try:
                j1 = math.radians(float(parts[1]))
                j2 = math.radians(float(parts[2]))
                j3 = math.radians(float(parts[3]))
                j4 = math.radians(float(parts[4]))
                j5 = math.radians(float(parts[5]))
                j6 = math.radians(float(parts[6]))
                
                # Get speed and acceleration if provided
                a = float(parts[7]) if len(parts) > 7 else 1.4
                v = float(parts[8]) if len(parts) > 8 else 1.05
                
                joint_positions = [j1, j2, j3, j4, j5, j6]
                print(f"Moving to joint positions (degrees): [{parts[1]}, {parts[2]}, {parts[3]}, {parts[4]}, {parts[5]}, {parts[6]}]...")
                result = robot.move_j(joints=joint_positions, a=a, v=v)
                if isinstance(result, bool):
                    if result:
                        print("Move completed successfully")
                    else:
                        print("Move failed")
                else:
                    print("Move command sent")
            except ValueError:
                print("Invalid joint angles. Please provide 6 numeric values in degrees")
                
        elif cmd == "rotate" and len(parts) >= 3:
            # Rotate specific joint
            try:
                joint_index = int(parts[1])
                if joint_index < 0 or joint_index > 5:
                    print(f"Invalid joint index: {joint_index}. Must be between 0 and 5.")
                    return
                
                angle_deg = float(parts[2])
                angle_rad = math.radians(angle_deg)
                
                # Get speed and acceleration if provided
                a = float(parts[3]) if len(parts) > 3 else 1.4
                v = float(parts[4]) if len(parts) > 4 else 1.05
                
                print(f"Rotating joint {joint_index} to {angle_deg} degrees...")
                result = robot.move_joint(joint_index, angle_rad, a=a, v=v)
                if isinstance(result, bool):
                    if result:
                        print("Rotation completed successfully")
                    else:
                        print("Rotation failed")
                else:
                    print("Rotation command sent")
            except ValueError:
                print("Invalid input. Please provide joint index (0-5) and angle in degrees")
                
        elif cmd == "open":
            # Open gripper
            print("Opening gripper...")
            result = robot.open_gripper(duration=0.4)
            if isinstance(result, bool):
                if result:
                    print("Gripper opened successfully")
                else:
                    print("Failed to open gripper")
            else:
                print("Open gripper command sent")
                
        elif cmd == "close":
            # Close gripper
            print("Closing gripper...")
            result = robot.close_gripper(duration=0.4)
            if isinstance(result, bool):
                if result:
                    print("Gripper closed successfully")
                else:
                    print("Failed to close gripper")
            else:
                print("Close gripper command sent")
                
        elif cmd == "relative" and len(parts) >= 2:
            # Move relative to current position
            try:
                dx = float(parts[1]) if len(parts) > 1 else 0
                dy = float(parts[2]) if len(parts) > 2 else 0
                dz = float(parts[3]) if len(parts) > 3 else 0
                drx = float(parts[4]) if len(parts) > 4 else 0
                dry = float(parts[5]) if len(parts) > 5 else 0
                drz = float(parts[6]) if len(parts) > 6 else 0
                
                # Get speed and acceleration if provided
                a = float(parts[7]) if len(parts) > 7 else 1.2
                v = float(parts[8]) if len(parts) > 8 else 0.25
                
                print(f"Moving relative: dx={dx}, dy={dy}, dz={dz}, drx={drx}, dry={dry}, drz={drz}...")
                result = robot.move_relative(dx=dx, dy=dy, dz=dz, drx=drx, dry=dry, drz=drz, a=a, v=v)
                if isinstance(result, bool):
                    if result:
                        print("Relative move completed successfully")
                    else:
                        print("Relative move failed")
                else:
                    print("Relative move command sent")
            except ValueError:
                print("Invalid coordinates. Please provide numeric values")
                
        elif cmd == "freedrive":
            # Enable/disable freedrive mode
            if len(parts) > 1 and parts[1] in ["on", "true", "1", "yes"]:
                print("Enabling freedrive mode...")
                result = robot.freedrive_mode(True)
                if isinstance(result, bool):
                    if result:
                        print("Freedrive mode enabled")
                    else:
                        print("Failed to enable freedrive mode")
                else:
                    print("Freedrive mode command sent")
            else:
                print("Disabling freedrive mode...")
                result = robot.freedrive_mode(False)
                if isinstance(result, bool):
                    if result:
                        print("Freedrive mode disabled")
                    else:
                        print("Failed to disable freedrive mode")
                else:
                    print("Freedrive mode command sent")
                    
        elif cmd == "help":
            display_help()
            
        elif cmd == "exit":
            return False
            
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for available commands")
            
    except Exception as e:
        print(f"Error executing command: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='UR Robot Control Program')
    parser.add_argument('--ip', type=str, default="169.254.200.239",
                        help='IP address of the robot')
    args = parser.parse_args()
    
    robot_ip = args.ip
    
    print(f"Connecting to UR robot at {robot_ip}...")
    robot = URRobotControl(robot_ip)
    
    # Connect to the robot
    try:
        connect_result = robot.connect()
        if isinstance(connect_result, bool) and connect_result:
            print("Connected to the robot")
        else:
            print("Connection status unclear, attempting to continue")
    except Exception as e:
        print(f"Failed to connect to the robot: {e}")
        return
    
    print(f"Robot status: {robot}")
    
    display_help()
    
    # Main command loop
    running = True
    while running:
        try:
            command = input("\nEnter command > ")
            running = parse_command(command, robot)
        except KeyboardInterrupt:
            print("\nOperation interrupted by user")
            running = False
        except Exception as e:
            print(f"\nError: {e}")
    
    # Disconnect from the robot
    print("\nDisconnecting from robot...")
    robot.disconnect()
    print("Disconnected")

if __name__ == "__main__":
    main()