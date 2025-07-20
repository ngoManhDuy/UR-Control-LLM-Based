#!/usr/bin/env python3

import math
from typing import List, Dict, Any

class RobotSafetyValidator:
    """Safety validation system for UR3 robot movements"""
    
    def __init__(self, ur_controller):
        self.ur_controller = ur_controller
        
        # UR3 Joint limits (degrees)
        self.joint_limits = {
            0: (-360, 360),    # Base joint
            1: (-360, 360),    # Shoulder joint  
            2: (-360, 360),    # Elbow joint
            3: (-360, 360),    # Wrist 1
            4: (-360, 360),    # Wrist 2
            5: (-360, 360),    # Wrist 3
        }
        
        # UR3 Workspace limits (meters) - approximate safe working area
        self.workspace_limits = {
            'x_min': -0.85, 'x_max': 0.85,
            'y_min': -0.85, 'y_max': 0.85, 
            'z_min': 0.0,   'z_max': 1.0
        }
        
        # Safety margins
        self.safety_margin = 0.05  # 5cm safety margin from workspace limits
        
    def validate_joint_movement(self, joint_index: int, target_angle: float) -> Dict[str, Any]:
        """Validate if a joint can safely move to target angle"""
        try:
            # Check joint index validity
            if joint_index < 0 or joint_index > 5:
                return {
                    'safe': False,
                    'reason': f"Invalid joint index {joint_index}. Must be 0-5",
                    'suggestion': "Use joint index 0-5 (0=base, 1=shoulder, 2=elbow, 3-5=wrists)"
                }
            
            # Check joint limits
            min_angle, max_angle = self.joint_limits[joint_index]
            if target_angle < min_angle or target_angle > max_angle:
                return {
                    'safe': False,
                    'reason': f"Joint {joint_index} angle {target_angle}° exceeds limits ({min_angle}° to {max_angle}°)",
                    'suggestion': f"Use angle between {min_angle}° and {max_angle}°"
                }
            
            # Get current joint angles to check for large movements
            try:
                current_joints = self.ur_controller.robot.get_joint_angles()
                if current_joints:
                    current_angle = math.degrees(current_joints[joint_index])
                    angle_change = abs(target_angle - current_angle)
                    
                    # Warn about large movements (>90 degrees)
                    if angle_change > 90:
                        return {
                            'safe': True,
                            'reason': f"Large movement detected: {angle_change:.1f}° change for joint {joint_index}",
                            'suggestion': "Consider smaller incremental movements for safety",
                            'warning': True
                        }
            except:
                pass  # Continue if can't get current position
            
            return {'safe': True, 'reason': 'Movement is within safe limits'}
            
        except Exception as e:
            return {
                'safe': False,
                'reason': f"Error validating joint movement: {e}",
                'suggestion': "Check robot connection and try again"
            }
    
    def validate_pose_movement(self, target_pose: List[float]) -> Dict[str, Any]:
        """Validate if TCP can safely move to target pose"""
        try:
            # Handle both list and numpy array inputs
            if hasattr(target_pose, 'tolist'):
                target_pose = target_pose.tolist()
            
            x, y, z = float(target_pose[0]), float(target_pose[1]), float(target_pose[2])
            
            # Check workspace boundaries with safety margin
            if (x < self.workspace_limits['x_min'] + self.safety_margin or 
                x > self.workspace_limits['x_max'] - self.safety_margin):
                return {
                    'safe': False,
                    'reason': f"X position {x:.3f}m outside safe workspace",
                    'suggestion': f"Keep X between {self.workspace_limits['x_min'] + self.safety_margin:.2f}m and {self.workspace_limits['x_max'] - self.safety_margin:.2f}m"
                }
            
            if (y < self.workspace_limits['y_min'] + self.safety_margin or 
                y > self.workspace_limits['y_max'] - self.safety_margin):
                return {
                    'safe': False,
                    'reason': f"Y position {y:.3f}m outside safe workspace", 
                    'suggestion': f"Keep Y between {self.workspace_limits['y_min'] + self.safety_margin:.2f}m and {self.workspace_limits['y_max'] - self.safety_margin:.2f}m"
                }
            
            if (z < self.workspace_limits['z_min'] + self.safety_margin or 
                z > self.workspace_limits['z_max'] - self.safety_margin):
                return {
                    'safe': False,
                    'reason': f"Z position {z:.3f}m outside safe workspace",
                    'suggestion': f"Keep Z between {self.workspace_limits['z_min'] + self.safety_margin:.2f}m and {self.workspace_limits['z_max'] - self.safety_margin:.2f}m"
                }
            
            # Check for ground collision (Z too low)
            if z < 0.1:
                return {
                    'safe': False,
                    'reason': f"Z position {z:.3f}m too close to ground/table",
                    'suggestion': "Keep Z above 0.1m to avoid collisions"
                }
            
            # Calculate distance from current position
            try:
                current_pose = self.ur_controller.robot.get_tcp_pose()
                if current_pose is not None:
                    # Handle numpy array or list
                    if hasattr(current_pose, 'tolist'):
                        current_pose = current_pose.tolist()
                    
                    current_x, current_y, current_z = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
                    distance = math.sqrt((x - current_x)**2 + (y - current_y)**2 + (z - current_z)**2)
                    
                    # Warn about large movements (>50cm)
                    if distance > 0.5:
                        return {
                            'safe': True,
                            'reason': f"Large movement detected: {distance:.2f}m distance",
                            'suggestion': "Consider smaller incremental movements for safety",
                            'warning': True
                        }
            except Exception as e:
                # Don't fail the safety check if we can't get current position for distance check
                print(f"⚠️ Could not calculate movement distance: {e}")
                pass
            
            return {'safe': True, 'reason': 'Movement is within safe workspace'}
            
        except Exception as e:
            return {
                'safe': False,
                'reason': f"Error validating pose movement: {e}",
                'suggestion': "Check robot connection and try again"
            }
    
    def validate_relative_movement(self, dx: float = 0, dy: float = 0, dz: float = 0) -> Dict[str, Any]:
        """Validate if relative movement is safe"""
        try:
            # Get current position
            current_pose = self.ur_controller.robot.get_tcp_pose()
            if current_pose is None:
                return {
                    'safe': False,
                    'reason': "Cannot get current robot position",
                    'suggestion': "Check robot connection"
                }
            
            # Handle numpy array or list
            if hasattr(current_pose, 'tolist'):
                current_pose = current_pose.tolist()
            
            # Calculate target position
            target_x = float(current_pose[0]) + float(dx)
            target_y = float(current_pose[1]) + float(dy)  
            target_z = float(current_pose[2]) + float(dz)
            
            # Create target pose with current orientation
            target_pose = [target_x, target_y, target_z, 
                          float(current_pose[3]), float(current_pose[4]), float(current_pose[5])]
            
            # Validate the target position
            return self.validate_pose_movement(target_pose)
            
        except Exception as e:
            return {
                'safe': False,
                'reason': f"Error validating relative movement: {e}",
                'suggestion': "Check robot connection and try again"
            } 

    def check_reachability(self, target_pose: List[float]) -> Dict[str, Any]:
        """Simple check if robot can physically reach the target 6D coordinates"""
        try:
            # Handle both list and numpy array inputs
            if hasattr(target_pose, 'tolist'):
                target_pose = target_pose.tolist()
            
            x, y, z = float(target_pose[0]), float(target_pose[1]), float(target_pose[2])
            rx, ry, rz = float(target_pose[3]), float(target_pose[4]), float(target_pose[5])
            
            # Simple workspace boundary check (basic reachability)
            max_reach = 0.85  # UR3 approximate max reach in meters
            distance_from_base = math.sqrt(x**2 + y**2)
            
            # Check if position is within robot's physical reach
            if distance_from_base > max_reach:
                return {
                    'reachable': False,
                    'reason': f"Position ({x:.3f}, {y:.3f}, {z:.3f}) is {distance_from_base:.3f}m from base, exceeds max reach {max_reach}m",
                    'suggestion': f"Move closer to robot base (within {max_reach}m radius)"
                }
            
            # Check Z limits (robot can't reach below base or too high)
            if z < -0.2 or z > 1.2:
                return {
                    'reachable': False,
                    'reason': f"Z position {z:.3f}m is outside robot's vertical reach (-0.2m to 1.2m)",
                    'suggestion': "Adjust Z position to be within robot's vertical range"
                }
            
            # Try inverse kinematics check if robot is connected
            try:
                # This is the real test - can the robot calculate joint angles for this pose?
                joint_angles = self.ur_controller.robot.get_inverse_kinematics(target_pose)
                
                if joint_angles is None or len(joint_angles) != 6:
                    return {
                        'reachable': False,
                        'reason': f"Robot cannot calculate valid joint angles for pose ({x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f})",
                        'suggestion': "Try a different position/orientation that is within robot's kinematic limits"
                    }
                
                # Check if any joint would exceed limits
                for i, angle in enumerate(joint_angles):
                    angle_deg = math.degrees(angle)
                    min_limit, max_limit = self.joint_limits[i]
                    
                    if angle_deg < min_limit or angle_deg > max_limit:
                        return {
                            'reachable': False,
                            'reason': f"Joint {i} would need {angle_deg:.1f}° (exceeds limit {min_limit}° to {max_limit}°)",
                            'suggestion': f"Adjust pose to keep joint {i} within limits"
                        }
                
                return {
                    'reachable': True,
                    'reason': f"Robot can reach pose ({x:.3f}, {y:.3f}, {z:.3f}) with valid joint configuration",
                    'joint_angles': [math.degrees(a) for a in joint_angles]
                }
                
            except Exception as ik_error:
                # Fallback to basic geometric check if IK fails
                return {
                    'reachable': True,
                    'reason': f"Position appears reachable (geometric check), but couldn't verify with inverse kinematics: {ik_error}",
                    'warning': True
                }
            
        except Exception as e:
            return {
                'reachable': False,
                'reason': f"Error checking reachability: {e}",
                'suggestion': "Check robot connection and pose format"
            } 