import os
import json
import asyncio
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI
from URRobotControl import URRobotControl
from ur_robot_llm_functions import FUNCTION_DEFINITIONS
import math

# Load environment variables
load_dotenv()

# Create OpenAI client
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# System message for the chat
SYSTEM_MESSAGE = {
    "role": "system", 
    "content": """You are a control assistant for a UR3 robot. You can:
    1. Move the robot to specific poses
    2. Control joint movements
    3. Move the robot relative to its current position
    4. Control the gripper
    5. Change robot modes (freedrive, normal)
    6. Get robot status
    7. Stop robot movement
    
    Always prioritize safety and verify movements are within safe limits."""
}

class URController:
    def __init__(self):
        """Initialize connection to the UR robot"""
        self.ROBOT_IP = '169.254.200.239'
        
        # Initialize robot connection
        print("Initializing UR Robot connection...")
        self.robot = URRobotControl(self.ROBOT_IP)
        
        # Connect to the robot
        if not self.robot.connect():
            print("Failed to connect to robot")
            sys.exit(1)
            
        print("Robot initialized!")

    def close(self):
        """Cleanup and close the robot connection."""
        print("Closing robot connection")
        self.robot.disconnect()

async def process_command(command, ur_controller):
    """
    Process a natural language command using OpenAI's function calling.
    """
    # Build the conversation messages
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

    # Check if the response contains a function call
    if message.function_call:
        function_call = message.function_call
        function_name = function_call.name
        arguments = function_call.arguments
        print(f"Function call received: {function_name} with arguments {arguments}")

        try:
            args = json.loads(arguments)
            
            # Execute the appropriate function based on the function name
            if function_name == "move_to_pose":
                success = ur_controller.robot.move_l(
                    pose=[args["x"], args["y"], args["z"], args["rx"], args["ry"], args["rz"]],
                    a=args.get("acceleration", 1.2),
                    v=args.get("speed", 0.25)
                )
                print("Move to pose completed" if success else "Move to pose failed")
                
            elif function_name == "move_joints":
                # Convert degrees to radians
                joints = [
                    math.radians(args["j0"]),
                    math.radians(args["j1"]),
                    math.radians(args["j2"]),
                    math.radians(args["j3"]),
                    math.radians(args["j4"]),
                    math.radians(args["j5"])
                ]
                success = ur_controller.robot.move_j(
                    joints=joints,
                    a=args.get("acceleration", 1.4),
                    v=args.get("speed", 1.05)
                )
                print("Move joints completed" if success else "Move joints failed")
                
            elif function_name == "move_relative":
                success = ur_controller.robot.move_relative(
                    dx=args.get("dx", 0),
                    dy=args.get("dy", 0),
                    dz=args.get("dz", 0),
                    drx=args.get("drx", 0),
                    dry=args.get("dry", 0),
                    drz=args.get("drz", 0),
                    a=args.get("acceleration", 1.2),
                    v=args.get("speed", 0.25)
                )
                print("Relative move completed" if success else "Relative move failed")
                
            elif function_name == "rotate_joint":
                success = ur_controller.robot.move_joint(
                    joint_index=args["joint_index"],
                    angle=math.radians(args["angle"]),
                    a=args.get("acceleration", 1.4),
                    v=args.get("speed", 1.05)
                )
                print("Joint rotation completed" if success else "Joint rotation failed")
                
            elif function_name == "control_gripper":
                if args["action"] == "open":
                    success = ur_controller.robot.open_gripper(duration=args.get("duration", 0.5))
                else:
                    success = ur_controller.robot.close_gripper(duration=args.get("duration", 0.5))
                print(f"Gripper {args['action']} completed" if success else f"Gripper {args['action']} failed")
                
            elif function_name == "set_robot_mode":
                if args["mode"] == "freedrive":
                    success = ur_controller.robot.freedrive_mode(enable=args.get("enable", True))
                else:
                    success = ur_controller.robot.teach_mode(enable=args.get("enable", True))
                print(f"Robot mode set to {args['mode']}" if success else f"Failed to set robot mode to {args['mode']}")
                
            elif function_name == "get_robot_status":
                status = ur_controller.robot.get_status()
                print("Robot Status:", status)
                
            elif function_name == "stop_robot":
                success = ur_controller.robot.stop()
                print("Robot stopped" if success else "Failed to stop robot")
                
            else:
                print(f"Unknown function: {function_name}")
                
        except Exception as e:
            print(f"Error executing function: {e}")
            
    elif message.content:  # If there's a normal response, print it
        print(f"Robot Assistant: {message.content}")
    else:
        print("Received an empty response.")

async def main():
    # Create instance of the robot controller
    ur = URController()

    print("\nUR Robot Control with LLM Interface")
    print("----------------------------------")
    print("Type natural language commands to control the robot.")
    print("Examples:")
    print("  - 'Move the robot to position x=0.5, y=0.2, z=0.3'")
    print("  - 'Rotate joint 0 to 45 degrees'")
    print("  - 'Open the gripper'")
    print("  - 'What is the current robot status?'")
    print("  - Type 'exit' to quit")
    print("----------------------------------")
    
    try:
        while True:
            # Get command from keyboard
            command = input("\nEnter command: ").strip()
            
            if command.lower() in ['exit', 'quit', 'goodbye']:
                print("Exiting program.")
                break
                
            # Process the command
            await process_command(command, ur)
            
    finally:
        # Cleanup
        ur.close()

if __name__ == "__main__":
    asyncio.run(main()) 