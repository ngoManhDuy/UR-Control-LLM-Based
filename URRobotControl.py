import numpy as np
import time
from src.UR.robotModel import RobotModel
from src.UR.urScript import UrScript
from src.UR.connectionState import ConnectionState


class URRobotControl:
    """
    Universal Robot Control Class
    
    This class provides a high-level interface to control Universal Robots through
    their TCP/IP interface. It allows for:
    - Connection to the robot using an IP address
    - Getting the current TCP pose (position and orientation)
    - Controlling robot movements (MoveJ, MoveL, MoveP)
    - Individual joint control
    """
    
    def __init__(self, host, log_path="ur_log/", log_config_path=None, has_force_torque=False):
        """
        Initialize the URRobotControl class.
        
        Args:
            host (str): IP address of the robot
            log_path (str): Path for logging
            log_config_path (str): Path for log configuration
            has_force_torque (bool): Whether the robot has a force-torque sensor
        """
        self.host = host
        self.robot_model = RobotModel(log_path=log_path, log_config_path=log_config_path)
        self.ur_script = UrScript(host, self.robot_model, hasForceTorque=has_force_torque)
        self.connection_state = ConnectionState.DISCONNECTED
        
    def connect(self):
        """
        Connect to the robot.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Check if robot is already connected by verifying we can get the TCP pose
            if self.get_tcp_pose() is not None:
                self.connection_state = ConnectionState.CONNECTED
                return True
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            self.connection_state = ConnectionState.ERROR
            return False
    
    def disconnect(self):
        """
        Disconnect from the robot.
        
        Returns:
            bool: True if disconnection is successful
        """
        try:
            self.ur_script.robotConnector.close()
            self.connection_state = ConnectionState.DISCONNECTED
            return True
        except Exception as e:
            print(f"Disconnection error: {e}")
            self.connection_state = ConnectionState.ERROR
            return False
    
    def get_tcp_pose(self):
        """
        Get the current TCP pose (position and orientation).
        
        Returns:
            list: [x, y, z, rx, ry, rz] where x, y, z are position in meters
                 and rx, ry, rz are orientation in radians (axis-angle representation)
        """
        return self.robot_model.ActualTCPPose()
    
    def get_joint_positions(self):
        """
        Get the current joint positions.
        
        Returns:
            list: Six joint angles in radians [j0, j1, j2, j3, j4, j5]
        """
        return self.robot_model.ActualQ()
    
    def move_j(self, joints=None, pose=None, a=1.4, v=1.05, t=0, r=0, wait=True):
        """
        Move to position (linear in joint-space).
        
        Args:
            joints (list): Target joint positions [j0, j1, j2, j3, j4, j5] in radians
            pose (list): Target pose [x, y, z, rx, ry, rz]
            a (float): Joint acceleration of leading axis [rad/s^2]
            v (float): Joint speed of leading axis [rad/s]
            t (float): Time [s]
            r (float): Blend radius [m]
            wait (bool): If True, function returns when movement is finished
            
        Returns:
            bool: True if movement is successful
        """
        try:
            self.ur_script.movej(q=joints, pose=pose, a=a, v=v, t=t, r=r, wait=wait)
            return True
        except Exception as e:
            print(f"MoveJ error: {e}")
            return False
    
    def move_l(self, pose=None, joints=None, a=1.2, v=0.25, t=0, r=0, wait=True):
        """
        Move to position (linear in tool-space).
        
        Args:
            pose (list): Target pose [x, y, z, rx, ry, rz]
            joints (list): Target joint positions [j0, j1, j2, j3, j4, j5] in radians
            a (float): Tool acceleration [m/s^2]
            v (float): Tool speed [m/s]
            t (float): Time [s]
            r (float): Blend radius [m]
            wait (bool): If True, function returns when movement is finished
            
        Returns:
            bool: True if movement is successful
        """
        try:
            self.ur_script.movel(pose=pose, q=joints, a=a, v=v, t=t, r=r, wait=wait)
            return True
        except Exception as e:
            print(f"MoveL error: {e}")
            return False
    
    def move_p(self, pose=None, joints=None, a=1.2, v=0.25, r=0, wait=True):
        """
        Move process - blend circular (in tool-space) and move linear (in tool-space).
        
        Args:
            pose (list): Target pose [x, y, z, rx, ry, rz]
            joints (list): Target joint positions [j0, j1, j2, j3, j4, j5] in radians
            a (float): Tool acceleration [m/s^2]
            v (float): Tool speed [m/s]
            r (float): Blend radius [m]
            wait (bool): If True, function returns when movement is finished
            
        Returns:
            bool: True if movement is successful
        """
        try:
            self.ur_script.movep(pose=pose, q=joints, a=a, v=v, r=r, wait=wait)
            return True
        except Exception as e:
            print(f"MoveP error: {e}")
            return False
    
    def move_relative(self, dx=0, dy=0, dz=0, drx=0, dry=0, drz=0, a=1.2, v=0.25, wait=True):
        """
        Move relative to current position in tool space.
        
        Args:
            dx (float): Change in x position [m]
            dy (float): Change in y position [m]
            dz (float): Change in z position [m]
            drx (float): Change in rx orientation [rad]
            dry (float): Change in ry orientation [rad]
            drz (float): Change in rz orientation [rad]
            a (float): Tool acceleration [m/s^2]
            v (float): Tool speed [m/s]
            wait (bool): If True, function returns when movement is finished
            
        Returns:
            bool: True if movement is successful
        """
        current_pose = self.get_tcp_pose()
        if current_pose is None:
            print("Failed to get current TCP pose")
            return False
        
        target_pose = [
            current_pose[0] + dx,
            current_pose[1] + dy,
            current_pose[2] + dz,
            current_pose[3] + drx,
            current_pose[4] + dry,
            current_pose[5] + drz
        ]
        
        return self.move_l(pose=target_pose, a=a, v=v, wait=wait)
    
    def move_joint(self, joint_index, angle, a=1.4, v=1.05, wait=True):
        """
        Move a specific joint to a target angle.
        
        Args:
            joint_index (int): Index of the joint to move (0-5)
            angle (float): Target angle in radians
            a (float): Joint acceleration [rad/s^2]
            v (float): Joint speed [rad/s]
            wait (bool): If True, function returns when movement is finished
            
        Returns:
            bool: True if movement is successful
        """
        if joint_index < 0 or joint_index > 5:
            print(f"Invalid joint index: {joint_index}. Must be between 0 and 5.")
            return False
        
        current_joints = self.get_joint_positions()
        if current_joints is None:
            print("Failed to get current joint positions")
            return False
        
        target_joints = current_joints.copy()
        target_joints[joint_index] = angle
        
        return self.move_j(joints=target_joints, a=a, v=v, wait=wait)
    
    def set_all_joint_positions(self, joint_angles, a=1.4, v=1.05, wait=True):
        """
        Set all joint positions at once.
        
        Args:
            joint_angles (list): Six joint angles in radians [j0, j1, j2, j3, j4, j5]
            a (float): Joint acceleration [rad/s^2]
            v (float): Joint speed [rad/s]
            wait (bool): If True, function returns when movement is finished
            
        Returns:
            bool: True if movement is successful
        """
        if len(joint_angles) != 6:
            print(f"Invalid joint angles: {joint_angles}. Must provide 6 joint angles.")
            return False
        
        return self.move_j(joints=joint_angles, a=a, v=v, wait=wait)
    
    def is_program_running(self):
        """
        Check if a program is currently running on the robot.
        
        Returns:
            bool: True if a program is running
        """
        return self.robot_model.RuntimeState()
    
    def stop(self):
        """
        Stop any running program.
        
        Returns:
            bool: True if stop command sent successfully
        """
        try:
            self.ur_script.stopj(a=1.5)
            return True
        except Exception as e:
            print(f"Stop error: {e}")
            return False
    
    def freedrive_mode(self, enable=True):
        """
        Enable or disable freedrive mode.
        
        Args:
            enable (bool): True to enable freedrive mode, False to disable
            
        Returns:
            bool: True if mode change is successful
        """
        try:
            if enable:
                self.ur_script.freedrive_mode()
            else:
                self.ur_script.end_freedrive_mode()
            return True
        except Exception as e:
            print(f"Freedrive mode error: {e}")
            return False
    
    def teach_mode(self, enable=True):
        """
        Enable or disable teach mode.
        
        Args:
            enable (bool): True to enable teach mode, False to disable
            
        Returns:
            bool: True if mode change is successful
        """
        try:
            if enable:
                self.ur_script.teach_mode()
            else:
                self.ur_script.end_teach_mode()
            return True
        except Exception as e:
            print(f"Teach mode error: {e}")
            return False
    
    def get_tcp_force(self):
        """
        Get the current TCP force.
        
        Returns:
            list: [fx, fy, fz, tx, ty, tz] where fx, fy, fz are forces in N
                 and tx, ty, tz are torques in Nm
        """
        return self.robot_model.ActualTCPForce()
    
    def get_status(self):
        """
        Get the current robot status.
        
        Returns:
            dict: Dictionary containing robot status information
        """
        return {
            "connected": self.connection_state == ConnectionState.CONNECTED,
            "tcp_pose": self.get_tcp_pose(),
            "joint_positions": self.get_joint_positions(),
            "program_running": self.is_program_running()
        }
    
    def __str__(self):
        """
        String representation of the robot control instance.
        
        Returns:
            str: String representation
        """
        status = self.get_status()
        if status["tcp_pose"] is not None:
            pose_str = f"TCP: [{', '.join([f'{x:.3f}' for x in status['tcp_pose']])}]"
        else:
            pose_str = "TCP: Unknown"
            
        if status["joint_positions"] is not None:
            joints_str = f"Joints: [{', '.join([f'{j:.3f}' for j in status['joint_positions']])}]"
        else:
            joints_str = "Joints: Unknown"
            
        return f"URRobotControl @ {self.host} - Connected: {status['connected']}, {pose_str}, {joints_str}" 