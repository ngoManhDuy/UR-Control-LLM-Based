#!/usr/bin/env python3

# Function definitions for LLM to control UR Robot
FUNCTION_DEFINITIONS = [
    # Basic Movement Functions
    {
        "name": "move_to_pose",
        "description": "Move the robot's end-effector (TCP) to a specific pose in space using linear movement in tool-space.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "Target X position in meters"
                },
                "y": {
                    "type": "number",
                    "description": "Target Y position in meters"
                },
                "z": {
                    "type": "number",
                    "description": "Target Z position in meters"
                },
                "rx": {
                    "type": "number",
                    "description": "Target RX rotation in radians"
                },
                "ry": {
                    "type": "number",
                    "description": "Target RY rotation in radians"
                },
                "rz": {
                    "type": "number",
                    "description": "Target RZ rotation in radians"
                },
                "speed": {
                    "type": "number",
                    "description": "Tool speed in m/s (default: 0.25)",
                    "default": 0.25
                },
                "acceleration": {
                    "type": "number",
                    "description": "Tool acceleration in m/s^2 (default: 1.2)",
                    "default": 1.2
                }
            },
            "required": ["x", "y", "z", "rx", "ry", "rz"]
        }
    },
    {
        "name": "move_joints",
        "description": "Move all robot joints to specific angles using joint-space movement.",
        "parameters": {
            "type": "object",
            "properties": {
                "j0": {
                    "type": "number",
                    "description": "Target angle for joint 0 (base) in degrees"
                },
                "j1": {
                    "type": "number",
                    "description": "Target angle for joint 1 (shoulder) in degrees"
                },
                "j2": {
                    "type": "number",
                    "description": "Target angle for joint 2 (elbow) in degrees"
                },
                "j3": {
                    "type": "number",
                    "description": "Target angle for joint 3 (wrist 1) in degrees"
                },
                "j4": {
                    "type": "number",
                    "description": "Target angle for joint 4 (wrist 2) in degrees"
                },
                "j5": {
                    "type": "number",
                    "description": "Target angle for joint 5 (wrist 3) in degrees"
                },
                "speed": {
                    "type": "number",
                    "description": "Joint speed in rad/s (default: 1.05)",
                    "default": 1.05
                },
                "acceleration": {
                    "type": "number",
                    "description": "Joint acceleration in rad/s^2 (default: 1.4)",
                    "default": 1.4
                }
            },
            "required": ["j0", "j1", "j2", "j3", "j4", "j5"]
        }
    },
    {
        "name": "move_relative",
        "description": "Move the robot relative to its current position in tool space.",
        "parameters": {
            "type": "object",
            "properties": {
                "dx": {
                    "type": "number",
                    "description": "Change in X position in meters",
                    "default": 0
                },
                "dy": {
                    "type": "number",
                    "description": "Change in Y position in meters",
                    "default": 0
                },
                "dz": {
                    "type": "number",
                    "description": "Change in Z position in meters",
                    "default": 0
                },
                "drx": {
                    "type": "number",
                    "description": "Change in RX rotation in radians",
                    "default": 0
                },
                "dry": {
                    "type": "number",
                    "description": "Change in RY rotation in radians",
                    "default": 0
                },
                "drz": {
                    "type": "number",
                    "description": "Change in RZ rotation in radians",
                    "default": 0
                },
                "speed": {
                    "type": "number",
                    "description": "Tool speed in m/s (default: 0.25)",
                    "default": 0.25
                },
                "acceleration": {
                    "type": "number",
                    "description": "Tool acceleration in m/s^2 (default: 1.2)",
                    "default": 1.2
                }
            }
        }
    },
    {
        "name": "rotate_joint",
        "description": "Rotate a specific joint to a target angle.",
        "parameters": {
            "type": "object",
            "properties": {
                "joint_index": {
                    "type": "integer",
                    "description": "Index of the joint to rotate (0-5)",
                    "minimum": 0,
                    "maximum": 5
                },
                "angle": {
                    "type": "number",
                    "description": "Target angle in degrees"
                },
                "speed": {
                    "type": "number",
                    "description": "Joint speed in rad/s (default: 1.05)",
                    "default": 1.05
                },
                "acceleration": {
                    "type": "number",
                    "description": "Joint acceleration in rad/s^2 (default: 1.4)",
                    "default": 1.4
                }
            },
            "required": ["joint_index", "angle"]
        }
    },
    
    # Gripper Control Functions
    {
        "name": "control_gripper",
        "description": "Control the robot's gripper (open or close).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform with the gripper",
                    "enum": ["open", "close"]
                },
                "duration": {
                    "type": "number",
                    "description": "Duration of the gripper action in seconds (default: 0.5)",
                    "default": 0.5
                }
            },
            "required": ["action"]
        }
    },
    
    # Robot Mode Control Functions
    {
        "name": "set_robot_mode",
        "description": "Change the robot's operating mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Mode to set the robot to",
                    "enum": ["freedrive", "normal"]
                },
                "enable": {
                    "type": "boolean",
                    "description": "True to enable the mode, False to disable",
                    "default": True
                }
            },
            "required": ["mode"]
        }
    },
    
    # Information Functions
    {
        "name": "get_robot_status",
        "description": "Get the current status of the robot including position, joint angles, and connection state.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    
    # Safety Functions
    {
        "name": "stop_robot",
        "description": "Immediately stop any robot movement.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
] 