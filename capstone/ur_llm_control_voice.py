import time
import os
from dotenv import load_dotenv
import json
import asyncio
from openai import AsyncOpenAI
from src import UR
import numpy as np
import logging
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from voice_handler import VoiceHandler
from functools import lru_cache
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QEventLoop
from chat_ui import ChatWindow

# Set up logging once at the module level
logging.basicConfig(
    filename='ur_robot.log',  # Save logs to file instead of console
    format="%(asctime)s-%(levelname)s-%(message)s",
    level=logging.INFO
)

# Load environment variables once
load_dotenv()

# Create a single OpenAI client instance
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Cache for OpenAI function definitions to avoid recreating them on each call
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

# System message for the chat - defined once
SYSTEM_MESSAGE = {
    "role": "system", 
    "content": "You are a voice control assistant for a UR3 robot. You are at Phenikaa Universtiy in Hanoi/Vietnam. You have two main jobs: 1) answer user questions, and 2) translate natural language commands into robot movements. For movement, 'left' and 'right' rotate the robot's base joint, while 'up' and 'down' move the end effector vertically. You can process sequences of movements like 'go left then move down' by extracting all directions from the command."
}

# Exit command set for faster lookup
EXIT_COMMANDS = {'exit', 'quit', 'goodbye'}

class URController:
    def __init__(self, chat_window=None):
        """Initialize connection to the UR robot"""
        self.ROBOT_IP = '169.254.200.239'
        self.chat_window = chat_window
        
        # Initialize robot connection
        logging.info("Initializing UR Robot")
        self.log_status("Initializing UR Robot connection...")
        
        try:
            self.robotModel = UR.robotModel.RobotModel()
            self.robot = UR.urScriptExt.UrScriptExt(
                host=self.ROBOT_IP, robotModel=self.robotModel
            )

            # Reset any errors and initialize the robot
            self.robot.reset_error()
            self.log_status("Robot initialized!")
            logging.info("Robot initialized")
            
            # Define movement parameters
            self.acceleration = 0.5  # Robot acceleration value
            self.velocity = 0.2      # Robot speed value
            
            # Start realtime control
            self.robot.init_realtime_control()
            time.sleep(1)  # Reduced wait time for initialization

            # Store current position
            self.current_pose = self.robot.get_actual_tcp_pose()

            # Define movement step size (in meters)
            self.step_size = 0.06  # 5cm movement per command
            
            # Pre-define direction mappings for faster lookup
            self.direction_map = {
                "left": (0, -1, 0, "Moving left"),
                "right": (0, 1, 0, "Moving right"),
                "up": (0, 0, 1, "Moving up"),
                "down": (0, 0, -1, "Moving down")
            }
            
            # Flag for robot connection status
            self.is_connected = True
            
        except Exception as e:
            self.log_status(f"Failed to connect to robot: {str(e)}")
            logging.error(f"Failed to connect to robot: {str(e)}")
            self.is_connected = False
            # Create dummy attributes for graceful failure
            self.robot = None
            self.robotModel = None
            self.current_pose = [0, 0, 0, 0, 0, 0]  # Dummy pose
            self.direction_map = {
                "left": (0, -1, 0, "Moving left (simulated)"),
                "right": (0, 1, 0, "Moving right (simulated)"),
                "up": (0, 0, 1, "Moving up (simulated)"),
                "down": (0, 0, -1, "Moving down (simulated)")
            }
    
    def log_status(self, message):
        """Log a status message to both console and chat UI if available"""
        print(message)
        if self.chat_window:
            self.chat_window.signals.add_status_message.emit(message)

    def move_direction(self, direction):
        """Move robot in the specified direction."""
        direction = direction.lower()
        
        # Use the direction map for faster lookup
        if direction not in self.direction_map:
            self.log_status(f"Unknown direction: {direction}")
            logging.warning(f"Invalid direction: {direction}")
            return False
        
        # Simulation mode if robot is not connected
        if not self.is_connected:
            _, _, _, message = self.direction_map[direction]
            self.log_status(f"{message} (SIMULATED)")
            time.sleep(0.5)  # Short delay for simulation
            return True
            
        # Get the current TCP pose only once
        try:
            current_pose = self.robot.get_actual_tcp_pose()
            new_pose = current_pose.copy()
            
            # Get direction values from the map
            x_change, y_change, z_change, message = self.direction_map[direction]
            
            # Apply the changes
            new_pose[0] += x_change * self.step_size
            new_pose[1] += y_change * self.step_size
            new_pose[2] += z_change * self.step_size
            
            self.log_status(message)
            logging.info(message)
            
            # Move to the new position using realtime control
            self.robot.set_realtime_pose(new_pose)
            
            # Use QTimer to keep UI responsive during movement
            def process_events():
                QApplication.processEvents()
            
            timer = QTimer()
            timer.timeout.connect(process_events)
            timer.start(50)  # Process events every 50ms
            
            time.sleep(0.75)  # Reduced wait time for movement
            timer.stop()
            
            # Update the current position
            self.current_pose = self.robot.get_actual_tcp_pose()
            logging.info(f"Moved {direction}: New pose = {self.current_pose}")
            return True
            
        except Exception as e:
            error_msg = f"Movement failed: {str(e)}"
            logging.error(error_msg)
            self.log_status(error_msg)
            return False

    def close(self):
        """Cleanup and close the robot connection."""
        self.log_status("Closing robot connection")
        if self.is_connected and self.robot:
            try:
                self.robot.close()
                logging.info("Robot connection closed")
            except:
                logging.warning("Error while closing robot connection")

class VoiceHandlerUI(VoiceHandler):
    """Extended VoiceHandler to integrate with the chat UI"""
    def __init__(self, chat_window=None):
        super().__init__()
        self.chat_window = chat_window
    
    def speak(self, text):
        """Override to add text to chat window"""
        # First update UI, then generate speech to improve responsiveness
        if self.chat_window:
            self.chat_window.signals.add_robot_message.emit(text)
            QApplication.processEvents()  # Process UI events right away
            
        # Call the original speak method
        super().speak(text)

async def process_command(command, ur_controller, voice_handler, chat_window=None):
    """
    Sends the natural language command to the OpenAI API using function calling.
    If a function call is returned, it parses the arguments and executes the move_direction.
    """
    # Add the user's command to the chat window
    if chat_window:
        chat_window.signals.add_user_message.emit(command)
    
    # Process UI events immediately
    await asyncio.sleep(0)
    QApplication.processEvents()
    
    # Build the conversation messages - reuse the system message
    messages = [
        SYSTEM_MESSAGE,
        {"role": "user", "content": command}
    ]

    # Show a thinking indicator
    if chat_window:
        chat_window.signals.add_status_message.emit("Processing your request...")
    
    # Use asyncio.sleep(0) periodically to yield control and keep UI responsive
    await asyncio.sleep(0)
    QApplication.processEvents()
    
    try:
        # Use a timeout for the API call
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                functions=FUNCTION_DEFINITIONS,
                function_call="auto"  
            ),
            timeout=10.0  # 10 second timeout
        )
    except asyncio.TimeoutError:
        error_msg = "Request timed out. The server took too long to respond."
        print(error_msg)
        voice_handler.speak("Sorry, the request timed out. Please try again.")
        if chat_window:
            chat_window.signals.add_status_message.emit(error_msg)
        return
    except Exception as e:
        error_msg = f"Error calling OpenAI API: {e}"
        print(error_msg)
        voice_handler.speak("I encountered an error while processing your command.")
        if chat_window:
            chat_window.signals.add_status_message.emit(error_msg)
        return

    message = response.choices[0].message
    
    # Process UI events again
    await asyncio.sleep(0)
    QApplication.processEvents()

    # Check if the response contains a function call
    if message.function_call:
        function_call = message.function_call
        function_name = function_call.name
        arguments = function_call.arguments
        print(f"Function call received: {function_name} with arguments {arguments}")
        if chat_window:
            chat_window.signals.add_status_message.emit(f"Processing function: {function_name}")

        if function_name == "move_direction":
            try:
                args = json.loads(arguments)
                direction = args.get("direction")
                if direction.lower() in ["right", "left", "up", "down"]:
                    success = ur_controller.move_direction(direction)
                    if success:
                        voice_handler.speak(f"I moved {direction.lower()} successfully.")
                    else:
                        voice_handler.speak(f"I failed to move {direction.lower()}.")
                else:
                    voice_handler.speak(f"I received an invalid direction: {direction}")
            except Exception as e:
                voice_handler.speak("I encountered an error while executing the movement.")
                print("Error parsing function call arguments:", e)
        elif function_name == "move_sequence":
            try:
                args = json.loads(arguments)
                directions = args.get("directions", [])
                
                if not directions:
                    voice_handler.speak("I didn't understand any directions in your command.")
                    print("No directions received in the sequence.")
                    return
                
                # Tell the user what sequence will be executed
                direction_list = ", then ".join(directions)
                voice_handler.speak(f"I'll move {direction_list}.")
                
                # Process UI events
                await asyncio.sleep(0)
                QApplication.processEvents()
                
                for direction in directions:
                    if direction.lower() in ["right", "left", "up", "down"]:
                        print(f"Executing movement: {direction}")
                        success = ur_controller.move_direction(direction)
                        # Process UI events between movements
                        QApplication.processEvents()
                        
                        if success:
                            print(f"Robot moved {direction.lower()} successfully.")
                        else:
                            voice_handler.speak(f"I failed to move {direction.lower()}.")
                            print(f"Robot failed to move {direction.lower()}.")
                            break  # Stop sequence if any movement fails
                    else:
                        voice_handler.speak(f"Invalid direction: {direction}")
                        print(f"Invalid direction in sequence: {direction}")
                
                voice_handler.speak("I completed the movement sequence.")
                print("Movement sequence completed.")
            except Exception as e:
                voice_handler.speak("I encountered an error while executing the movement sequence.")
                print(f"Error executing movement sequence: {e}")
        else:
            voice_handler.speak("I received an unknown command.")
            print("Received an unknown function call.")
    elif message.content:  # If there's a normal response, speak it
        voice_handler.speak(message.content)
        print("API Response:", message.content)
    else:
        voice_handler.speak("I didn't receive any response.")
        print("Received an empty response.")

async def main():
    # Initialize the PyQt application
    app = QApplication(sys.argv)
    chat_window = ChatWindow()
    chat_window.show()
    
    # Create instances of the robot controller and voice handler with UI integration
    chat_window.signals.add_status_message.emit("Connecting to UR3 robot...")
    ur = URController(chat_window)
    voice_handler = VoiceHandlerUI(chat_window)

    # Setup an event loop for handling events during sleep
    def process_events_during_sleep(seconds):
        loop = QEventLoop()
        QTimer.singleShot(int(seconds * 1000), loop.quit)
        loop.exec()
    
    try:
        # Initial greeting
        chat_window.signals.add_status_message.emit("Starting voice recognition system...")
        process_events_during_sleep(0.5)  # Short pause to update UI
        
        voice_handler.speak("Hello! I am your UR3 robot assistant. How can I help you today?")
        chat_window.signals.add_status_message.emit("Listening... Say something or type 'exit' to quit")
        
        # Set up event processing
        while True:
            # Process Qt events before listening - crucial for UI responsiveness
            app.processEvents()
            
            # Listen for voice command with a status update
            chat_window.signals.add_status_message.emit("Listening... (speak when ready)")
            command = voice_handler.listen_for_command()
            
            if command is None:
                voice_handler.speak("I didn't catch that. Could you please repeat?")
                continue
                
            if command.lower() in EXIT_COMMANDS:
                voice_handler.speak("Goodbye! Have a great day!")
                process_events_during_sleep(1.0)  # Let the UI update before exiting
                break
                
            await process_command(command, ur, voice_handler, chat_window)
            
            # Give UI time to update after command processing
            process_events_during_sleep(0.2)
            
    finally:
        # Cleanup
        chat_window.signals.add_status_message.emit("Closing connections...")
        process_events_during_sleep(0.5)  # Let the user see the final status
        
        ur.close()
        voice_handler.close()
        
        # Exit application
        app.quit()

if __name__ == "__main__":
    asyncio.run(main())