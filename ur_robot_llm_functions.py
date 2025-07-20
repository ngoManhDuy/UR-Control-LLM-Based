#!/usr/bin/env python3

# ESSENTIAL FUNCTION DEFINITIONS (Cost-Optimized)
# Reduced from 17 to 8 essential functions to cut API costs by ~70%
FUNCTION_DEFINITIONS = [
    # Basic Movement Functions
    {
        "name": "move_to_pose",
        "description": "Move robot TCP to specific pose (linear movement).",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X position in meters"},
                "y": {"type": "number", "description": "Y position in meters"},
                "z": {"type": "number", "description": "Z position in meters"},
                "rx": {"type": "number", "description": "RX rotation in radians"},
                "ry": {"type": "number", "description": "RY rotation in radians"},
                "rz": {"type": "number", "description": "RZ rotation in radians"},
                "speed": {"type": "number", "description": "Speed m/s (default: 0.25)", "default": 0.25},
                "acceleration": {"type": "number", "description": "Acceleration m/s^2 (default: 1.2)", "default": 1.2}
            },
            "required": ["x", "y", "z", "rx", "ry", "rz"]
        }
    },
    {
        "name": "move_joints",
        "description": "Move all robot joints to specific angles.",
        "parameters": {
            "type": "object",
            "properties": {
                "j0": {"type": "number", "description": "Joint 0 angle in degrees"},
                "j1": {"type": "number", "description": "Joint 1 angle in degrees"},
                "j2": {"type": "number", "description": "Joint 2 angle in degrees"},
                "j3": {"type": "number", "description": "Joint 3 angle in degrees"},
                "j4": {"type": "number", "description": "Joint 4 angle in degrees"},
                "j5": {"type": "number", "description": "Joint 5 angle in degrees"}
            },
            "required": ["j0", "j1", "j2", "j3", "j4", "j5"]
        }
    },
    {
        "name": "move_relative",
        "description": "Move robot relative to current position.",
        "parameters": {
            "type": "object",
            "properties": {
                "dx": {"type": "number", "description": "Change in X (m)", "default": 0},
                "dy": {"type": "number", "description": "Change in Y (m)", "default": 0},
                "dz": {"type": "number", "description": "Change in Z (m)", "default": 0},
                "drx": {"type": "number", "description": "Change in RX (rad)", "default": 0},
                "dry": {"type": "number", "description": "Change in RY (rad)", "default": 0},
                "drz": {"type": "number", "description": "Change in RZ (rad)", "default": 0}
            }
        }
    },
    {
        "name": "control_gripper",
        "description": "Control gripper (open/close).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Action", "enum": ["open", "close"]},
                "duration": {"type": "number", "description": "Duration in seconds", "default": 0.5}
                },
            "required": ["action"]
        }
    },
    {
        "name": "get_robot_status",
        "description": "Get current robot status and position.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "stop_robot",
        "description": "Emergency stop robot.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "rotate_joint",
        "description": "Rotate specific joint to angle.",
        "parameters": {
            "type": "object",
            "properties": {
                "joint_index": {"type": "integer", "description": "Joint index (0-5)", "minimum": 0, "maximum": 5},
                "angle": {"type": "number", "description": "Target angle in degrees"}
            },
            "required": ["joint_index", "angle"]
        }
    },
    {
        "name": "find_object",
        "description": "Find objects using continuous scanning. Positions robot at object's X,Y coordinates. Use move_down_to_pick() afterward for precise picking.",
        "parameters": {
            "type": "object",
            "properties": {
                "scan_range_degrees": {"type": "number", "description": "Total scan range degrees", "default": 180},
                "scan_speed": {"type": "number", "description": "Scan speed rad/s", "default": 0.1},
                "check_interval": {"type": "number", "description": "Check interval seconds", "default": 0.5}
            }
        }
    },
    {
        "name": "move_down_to_pick",
        "description": "Move down, re-detect object, and position precisely for picking. Use after find_object() to prepare for gripper operation. Automatically uses the last found object coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Object X position (optional - uses last found object if not provided)"},
                "y": {"type": "number", "description": "Object Y position (optional - uses last found object if not provided)"},
                "z": {"type": "number", "description": "Object Z position (optional - uses last found object if not provided)"},
                "confidence": {"type": "number", "description": "Initial detection confidence (optional)"}
            }
        }
    },
    {
        "name": "rotate_joint_relative",
        "description": "Rotate a specific joint by a relative amount (forward/backward from current position).",
        "parameters": {
            "type": "object",
            "properties": {
                "joint_index": {"type": "integer", "description": "Joint index (0-5)", "minimum": 0, "maximum": 5},
                "angle_change": {"type": "number", "description": "Angle change in degrees (positive for forward, negative for backward)"}
            },
            "required": ["joint_index", "angle_change"]
        }
    },
    # Safety Validation Functions
    {
        "name": "check_movement_safety",
        "description": "Check if a planned movement is safe before execution. Use this for any risky movements.",
        "parameters": {
            "type": "object",
            "properties": {
                "movement_type": {"type": "string", "description": "Type of movement", "enum": ["pose", "relative", "joint"]},
                "x": {"type": "number", "description": "X position in meters (for pose movements)"},
                "y": {"type": "number", "description": "Y position in meters (for pose movements)"},
                "z": {"type": "number", "description": "Z position in meters (for pose movements)"},
                "rx": {"type": "number", "description": "RX rotation in radians (for pose movements)"},
                "ry": {"type": "number", "description": "RY rotation in radians (for pose movements)"},
                "rz": {"type": "number", "description": "RZ rotation in radians (for pose movements)"},
                "dx": {"type": "number", "description": "Change in X (m) for relative movements"},
                "dy": {"type": "number", "description": "Change in Y (m) for relative movements"},
                "dz": {"type": "number", "description": "Change in Z (m) for relative movements"},
                "joint_index": {"type": "integer", "description": "Joint index (0-5) for joint movements"},
                "angle": {"type": "number", "description": "Target angle in degrees for joint movements"}
            },
            "required": ["movement_type"]
        }
    },
    {
        "name": "get_workspace_limits",
        "description": "Get the safe workspace boundaries and joint limits for the robot.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "validate_position",
        "description": "Check if a specific position is within the robot's safe workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X position in meters"},
                "y": {"type": "number", "description": "Y position in meters"},
                "z": {"type": "number", "description": "Z position in meters"}
            },
            "required": ["x", "y", "z"]
        }
    },
    {
        "name": "check_reachability",
        "description": "Check if robot can physically reach specific 6D coordinates (position + orientation). Use this before attempting to move to new coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X position in meters"},
                "y": {"type": "number", "description": "Y position in meters"},
                "z": {"type": "number", "description": "Z position in meters"},
                "rx": {"type": "number", "description": "RX rotation in radians"},
                "ry": {"type": "number", "description": "RY rotation in radians"},
                "rz": {"type": "number", "description": "RZ rotation in radians"}
            },
            "required": ["x", "y", "z", "rx", "ry", "rz"]
        }
    }
]
    
# FULL FUNCTION DEFINITIONS (Commented out to reduce API costs)
# Uncomment specific functions as needed for advanced features
"""
EXTENDED_FUNCTION_DEFINITIONS = [
    {
        "name": "complete_object_search",
        "description": "Discrete angle scanning for objects (fallback method).",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "YOLO model path", "default": "best.pt"}
            }
        }
    },
    {
        "name": "rotate_tcp_for_alignment",
        "description": "Fine TCP rotation for object alignment.",
        "parameters": {
            "type": "object",
            "properties": {
                "rotation_axis": {"type": "string", "description": "Rotation axis", "enum": ["rx", "ry", "rz"]},
                "angle_degrees": {"type": "number", "description": "Rotation angle in degrees"},
                "speed": {"type": "number", "description": "Movement speed", "default": 0.05}
            },
            "required": ["rotation_axis", "angle_degrees"]
        }
    },
    {
        "name": "move_to_scanning_pose",
        "description": "Move robot to initial scanning pose.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "YOLO model path", "default": "best.pt"}
            }
        }
    },
    {
        "name": "detect_object_at_position",
        "description": "Detect object at current robot position.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "rotate_base_and_scan",
        "description": "Rotate base to angle and scan for objects.",
        "parameters": {
            "type": "object",
            "properties": {
                "angle_degrees": {"type": "number", "description": "Target base angle in degrees"}
            },
            "required": ["angle_degrees"]
        }
    },
    {
        "name": "get_base_angle",
        "description": "Get current base joint angle.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "move_to_object_xy",
        "description": "Move to object's X,Y coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X coordinate"},
                "y": {"type": "number", "description": "Y coordinate"},
                "z": {"type": "number", "description": "Z coordinate"}
            },
            "required": ["x", "y", "z"]
        }
    },
    {
        "name": "find_object_and_position",
        "description": "Find object and position robot automatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "YOLO model path", "default": "best.pt"}
            }
        }
    },
    {
        "name": "set_robot_mode",
        "description": "Change robot operating mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "description": "Robot mode", "enum": ["freedrive", "teach"]},
                "enable": {"type": "boolean", "description": "Enable mode", "default": true}
            },
            "required": ["mode"]
        }
    }
]
"""

# ... existing code ... 