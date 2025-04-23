import time
import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from src import UR
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

# OpenAI function definitions
FUNCTION_DEFINITIONS = [
    {
        "name": "move_direction",
        "description": "Moves the robot in the specified direction. Right/Left moves the robot horizontally. Up/Down moves the end effector vertically.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "description": "The direction to move the robot. 'Right'/'Left' moves horizontally. 'Up'/'Down' moves vertically."
                }
            },
            "required": ["direction"]
        }
    },
    {
        "name": "move_sequence",
        "description": "Moves the robot in a sequence of directions, one after another.",
        "parameters": {
            "type": "object",
            "properties": {
                "directions": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of directions to move the robot in sequence. Each direction can be 'Right', 'Left', 'Up', or 'Down'."
                }
            },
            "required": ["directions"]
        }
    }
]

# System message for the chat
SYSTEM_MESSAGE = {
    "role": "system", 
    "content": "You are a control assistant for a UR3 robot. You have two main jobs: 1) answer user questions, and 2) translate natural language commands into robot movements. For movement, 'left' and 'right' rotate the robot's base joint, while 'up' and 'down' move the end effector vertically. You can process sequences of movements like 'go left then move down' by extracting all directions from the command."
}

class URController:
    def __init__(self):
        """Initialize connection to the UR robot"""
        self.ROBOT_IP = '169.254.200.239'
        
        # Initialize robot connection
        logging.info("Initializing UR Robot")
        print("Initializing UR Robot connection...")
        self.robotModel = UR.robotModel.RobotModel()
        self.robot = UR.urScriptExt.UrScriptExt(
            host=self.ROBOT_IP, robotModel=self.robotModel
        )

        # Reset any errors and initialize the robot
        self.robot.reset_error()
        print("Robot initialized!")
        logging.info("Robot initialized")
        
        # Define movement parameters
        self.acceleration = 0.5  # Robot acceleration value
        self.velocity = 0.2      # Robot speed value
        
        # Start realtime control
        self.robot.init_realtime_control()
        time.sleep(2)  # Wait for initialization

        # Store current position
        self.current_pose = self.robot.get_actual_tcp_pose()

        # Define movement step size (in meters)
        self.step_size = 0.05  # 5cm movement per command
        
        # Pre-define direction mappings for faster lookup
        self.direction_map = {
            "left": (0, -1, 0, "Moving left"),
            "right": (0, 1, 0, "Moving right"),
            "up": (0, 0, 1, "Moving up"),
            "down": (0, 0, -1, "Moving down")
        }

    def move_direction(self, direction):
        """Move robot in the specified direction."""
        direction = direction.lower()
        
        # Use the direction map for faster lookup
        if direction not in self.direction_map:
            print(f"Unknown direction: {direction}")
            logging.warning(f"Invalid direction: {direction}")
            return False
            
        # Get the current TCP pose only once
        current_pose = self.robot.get_actual_tcp_pose()
        new_pose = current_pose.copy()
        
        # Get direction values from the map
        x_change, y_change, z_change, message = self.direction_map[direction]
        
        try:
            # Apply the changes
            new_pose[0] += x_change * self.step_size
            new_pose[1] += y_change * self.step_size
            new_pose[2] += z_change * self.step_size
            
            print(message)
            logging.info(message)
            
            # Move to the new position using realtime control
            self.robot.set_realtime_pose(new_pose)
            time.sleep(1)  # Wait for the movement to complete
            
            # Update the current position
            self.current_pose = self.robot.get_actual_tcp_pose()
            logging.info(f"Moved {direction}: New pose = {self.current_pose}")
            return True
            
        except Exception as e:
            logging.error(f"Movement failed: {str(e)}")
            print(f"Movement failed: {str(e)}")
            return False

    def close(self):
        """Cleanup and close the robot connection."""
        print("Closing robot connection")
        self.robot.close()
        logging.info("Robot connection closed")

async def process_command(command, ur_controller):
    """
    Sends the natural language command to the OpenAI API using function calling.
    If a function call is returned, it parses the arguments and executes the move_direction.
    """
    # Build the conversation messages
    messages = [
        SYSTEM_MESSAGE,
        {"role": "user", "content": command}
    ]

    print("Processing command...")
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
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

        if function_name == "move_direction":
            try:
                args = json.loads(arguments)
                direction = args.get("direction")
                if direction.lower() in ["right", "left", "up", "down"]:
                    success = ur_controller.move_direction(direction)
                    if success:
                        print(f"Robot moved {direction.lower()} successfully.")
                    else:
                        print(f"Robot failed to move {direction.lower()}.")
                else:
                    print(f"Invalid direction received: {direction}")
            except Exception as e:
                print(f"Error executing movement: {e}")
        elif function_name == "move_sequence":
            try:
                args = json.loads(arguments)
                directions = args.get("directions", [])
                
                if not directions:
                    print("No directions received in the sequence.")
                    return
                
                for direction in directions:
                    if direction.lower() in ["right", "left", "up", "down"]:
                        print(f"Executing movement: {direction}")
                        success = ur_controller.move_direction(direction)
                        if success:
                            print(f"Robot moved {direction.lower()} successfully.")
                        else:
                            print(f"Robot failed to move {direction.lower()}.")
                            break  # Stop sequence if any movement fails
                    else:
                        print(f"Invalid direction in sequence: {direction}")
                
                print("Movement sequence completed.")
            except Exception as e:
                print(f"Error executing movement sequence: {e}")
        else:
            print("Received an unknown function call.")
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
    print("  - 'Move the robot to the left'")
    print("  - 'Go up by 5 centimeters'")
    print("  - 'What can you do?'")
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