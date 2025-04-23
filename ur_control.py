import time
from src import UR
import numpy as np
import logging

# Configure logging to file only, no console output
logging.basicConfig(
    #filename='ur_robot.log',  # Save logs to file instead of console
    format="%(asctime)s-%(levelname)s-%(message)s",
    level=logging.INFO
)

class URController:
    def __init__(self, robot_ip='169.254.200.239'):
        """Initialize connection to the UR robot"""
        self.ROBOT_IP = robot_ip
        
        print("Initializing UR Robot connection...")
        self.robotModel = UR.robotModel.RobotModel()
        self.robot = UR.urScriptExt.UrScriptExt(
            host=self.ROBOT_IP,
            robotModel=self.robotModel
        )
        
        # Reset any errors and initialize the robot
        self.robot.reset_error()
        print("Robot initialized!")
        
        # Define movement parameters
        self.acceleration = 0.5  # Robot acceleration value (lower for smoother movement)
        self.velocity = 0.2      # Robot speed value (lower for safer movement)
        
        # Start realtime control
        self.robot.init_realtime_control()
        time.sleep(2)  # Short wait to ensure initialization
        
        # Get current position as reference
        self.current_pose = self.robot.get_actual_tcp_pose()
        
        # Define the step size for each movement (in meters)
        self.step_size = 0.05
        
    def move_robot(self, direction):
        """Move the robot in the specified direction"""
        # Get the current TCP pose
        current_pose = self.robot.get_actual_tcp_pose()
        
        # Create a copy of the current pose to modify
        new_pose = current_pose.copy()
        
        # Modify the pose based on the specified direction
        if direction.lower() == "left":
            new_pose[0] -= self.step_size  # Move in negative X direction
            print("Moving left")
        elif direction.lower() == "right":
            new_pose[0] += self.step_size  # Move in positive X direction
            print("Moving right")
        elif direction.lower() == "up":
            new_pose[2] += self.step_size  # Move in positive Z direction
            print("Moving up")
        elif direction.lower() == "down":
            new_pose[2] -= self.step_size  # Move in negative Z direction
            print("Moving down")
        else:
            print(f"Unknown direction: {direction}")
            return False
        
        # Move to the new position using realtime control
        self.robot.set_realtime_pose(new_pose)
        time.sleep(1)  # Wait a moment for the movement to complete
        
        # Update the current position
        self.current_pose = self.robot.get_actual_tcp_pose()
        return True
        
    def command_interface(self):
        """Simple command interface to control the robot"""
        print("\nUR Robot Control Interface")
        print("-------------------------")
        print("Available commands:")
        print("  left  - Move robot left")
        print("  right - Move robot right")
        print("  up    - Move robot up")
        print("  down  - Move robot down")
        print("  exit  - Exit the program")
        
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == "exit":
                break
            elif command in ["left", "right", "up", "down"]:
                self.move_robot(command)
            else:
                print("Unknown command. Please use: left, right, up, down, or exit")
                
        self.close()
    
    def close(self):
        """Close the connection to the robot"""
        print("Closing robot connection")
        self.robot.close()

if __name__ == "__main__":
    try:
        # Create the robot controller with the specified IP
        controller = URController(robot_ip='192.168.56.102')
        
        # Start the command interface
        controller.command_interface()
    except Exception as e:
        print(f"Error: {e}")
        # Attempt to close the connection if an error occurs
        try:
            controller.close()
        except:
            pass 