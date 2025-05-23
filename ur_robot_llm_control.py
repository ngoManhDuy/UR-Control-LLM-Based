#!/usr/bin/env python3
import time
import os
import json
import math
import asyncio
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI
from URRobotControl import URRobotControl
from ur_robot_llm_functions import FUNCTION_DEFINITIONS
import logging

# Set up logging
logging.basicConfig(
    filename='ur_robot.log',
    format="%(asctime)s-%(levelname)s-%(message)s",
    level=logging.INFO
)

# Load environment variables
load_dotenv()

# Create OpenAI client
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# System message for the chat
SYSTEM_MESSAGE = {
    "role": "system", 
    "content": """You are an advanced control assistant for a UR3 robot. Your role is to:
1) Answer user questions about the robot's capabilities
2) Translate natural language commands into precise robot control functions
3) Help users understand the robot's current state and movements

You have access to various control functions:
- Moving to specific poses in space
- Controlling individual joints
- Relative movements from current position
- Gripper control (open/close)
- Mode control (freedrive/normal)
- Status monitoring and emergency stop

Always prioritize safety and provide clear feedback about what actions you're taking.
If a command is ambiguous, ask for clarification rather than making assumptions."""
}

class URRobotLLMController:
    def __init__(self):
        """Initialize connection to the UR robot"""
        self.ROBOT_IP = '169.254.200.239'
        
        logging.info("Initializing UR Robot")
        print("Initializing UR Robot connection...")
        
        try:
            self.robot = URRobotControl(self.ROBOT_IP)
            connect_result = self.robot.connect()
            
            if connect_result:
                print("Robot initialized and connected!")
                logging.info("Robot initialized and connected")
            else:
                print("Warning: Robot connection status unclear")
                logging.warning("Robot connection status unclear")
                
        except Exception as e:
            print(f"Error initializing robot: {e}")
            logging.error(f"Error initializing robot: {e}")
            sys.exit(1)

    def move_to_pose(self, x, y, z, rx, ry, rz, speed=0.25, acceleration=1.2):
        """Move robot TCP to a specific pose."""
        try:
            success = self.robot.move_l(pose=[x, y, z, rx, ry, rz], v=speed, a=acceleration)
            return success
        except Exception as e:
            logging.error(f"Move to pose error: {e}")
            return False

    def move_joints(self, j0, j1, j2, j3, j4, j5, speed=1.05, acceleration=1.4):
        """Move robot joints to specific angles (input in degrees, converted to radians)."""
        try:
            # Convert degrees to radians
            joints = [math.radians(j) for j in [j0, j1, j2, j3, j4, j5]]
            success = self.robot.move_j(joints=joints, v=speed, a=acceleration)
            return success
        except Exception as e:
            logging.error(f"Move joints error: {e}")
            return False

    def move_relative(self, dx=0, dy=0, dz=0, drx=0, dry=0, drz=0, speed=0.25, acceleration=1.2):
        """Move robot relative to current position."""
        try:
            success = self.robot.move_relative(dx=dx, dy=dy, dz=dz, 
                                            drx=drx, dry=dry, drz=drz,
                                            v=speed, a=acceleration)
            return success
        except Exception as e:
            logging.error(f"Relative move error: {e}")
            return False

    def rotate_joint(self, joint_index, angle, speed=1.05, acceleration=1.4):
        """Rotate a specific joint to an angle (input in degrees, converted to radians)."""
        try:
            angle_rad = math.radians(angle)
            success = self.robot.move_joint(joint_index, angle_rad, v=speed, a=acceleration)
            return success
        except Exception as e:
            logging.error(f"Rotate joint error: {e}")
            return False

    def control_gripper(self, action, duration=0.5):
        """Control the gripper (open/close)."""
        try:
            if action.lower() == "open":
                success = self.robot.open_gripper(duration=duration)
            elif action.lower() == "close":
                success = self.robot.close_gripper(duration=duration)
            else:
                logging.error(f"Invalid gripper action: {action}")
                return False
            return success
        except Exception as e:
            logging.error(f"Gripper control error: {e}")
            return False

    def set_robot_mode(self, mode, enable=True):
        """Set robot mode (freedrive/normal)."""
        try:
            if mode.lower() == "freedrive":
                success = self.robot.freedrive_mode(enable)
            elif mode.lower() == "normal":
                success = self.robot.freedrive_mode(not enable)  # Disable freedrive for normal mode
            else:
                logging.error(f"Invalid robot mode: {mode}")
                return False
            return success
        except Exception as e:
            logging.error(f"Set mode error: {e}")
            return False

    def get_robot_status(self):
        """Get current robot status."""
        try:
            status = self.robot.get_status()
            return status
        except Exception as e:
            logging.error(f"Get status error: {e}")
            return None

    def stop_robot(self):
        """Emergency stop the robot."""
        try:
            success = self.robot.stop()
            return success
        except Exception as e:
            logging.error(f"Stop robot error: {e}")
            return False

    def close(self):
        """Cleanup and close the robot connection."""
        print("Closing robot connection")
        self.robot.disconnect()
        logging.info("Robot connection closed")

async def process_command(command, controller):
    """Process natural language command using LLM and execute appropriate robot functions."""
    messages = [
        SYSTEM_MESSAGE,
        {"role": "user", "content": command}
    ]

    print("Processing command...")
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=FUNCTION_DEFINITIONS,
            function_call="auto"
        )
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return

    message = response.choices[0].message

    # Handle function calls
    if message.function_call:
        function_name = message.function_call.name
        try:
            arguments = json.loads(message.function_call.arguments)
            
            # Map function names to controller methods
            function_map = {
                "move_to_pose": controller.move_to_pose,
                "move_joints": controller.move_joints,
                "move_relative": controller.move_relative,
                "rotate_joint": controller.rotate_joint,
                "control_gripper": controller.control_gripper,
                "set_robot_mode": controller.set_robot_mode,
                "get_robot_status": controller.get_robot_status,
                "stop_robot": controller.stop_robot
            }
            
            if function_name in function_map:
                print(f"Executing {function_name} with arguments: {arguments}")
                result = function_map[function_name](**arguments)
                
                if result is True:
                    print(f"Successfully executed {function_name}")
                elif result is False:
                    print(f"Failed to execute {function_name}")
                elif result is not None:  # For get_robot_status
                    print(f"Robot status: {result}")
            else:
                print(f"Unknown function: {function_name}")
                
        except Exception as e:
            print(f"Error executing function: {e}")
            logging.error(f"Function execution error: {e}")
    
    elif message.content:
        print(f"Robot Assistant: {message.content}")
    else:
        print("Received an empty response.")

async def main():
    # Create instance of the robot controller
    controller = URRobotLLMController()

    print("\nUR Robot Advanced Control with LLM Interface")
    print("-------------------------------------------")
    print("Type natural language commands to control the robot.")
    print("Examples:")
    print("  - 'Move the robot to position x=0.5, y=0.2, z=0.3'")
    print("  - 'Rotate joint 0 to 45 degrees'")
    print("  - 'Open the gripper'")
    print("  - 'Enable freedrive mode'")
    print("  - 'What is the current robot status?'")
    print("  - Type 'exit' to quit")
    print("-------------------------------------------")
    
    try:
        while True:
            command = input("\nEnter command: ").strip()
            
            if command.lower() in ['exit', 'quit']:
                break
                
            await process_command(command, controller)
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"Main loop error: {e}")
    finally:
        controller.close()
        print("Program terminated.")

if __name__ == "__main__":
    asyncio.run(main()) 