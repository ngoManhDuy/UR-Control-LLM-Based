#!/usr/bin/env python3
import time
import logging
import sys
import tty
import termios
from src import UR

# Set up logging
logging.basicConfig(
    filename='gripper.log',
    format="%(asctime)s-%(levelname)s-%(message)s",
    level=logging.INFO
)

class GripperController:
    def __init__(self):
        """Initialize connection to the UR robot for gripper control"""
        self.ROBOT_IP = '169.254.200.239'
        
        # Initialize robot connection
        logging.info("Initializing UR Robot for gripper control")
        print("Initializing UR Robot connection...")
        self.robotModel = UR.robotModel.RobotModel()
        self.robot = UR.urScriptExt.UrScriptExt(
            host=self.ROBOT_IP, robotModel=self.robotModel
        )

        # Reset any errors and initialize the robot
        self.robot.reset_error()
        print("Robot initialized!")
        logging.info("Robot initialized")
        
        # Start realtime control
        self.robot.init_realtime_control()
        time.sleep(2)  # Wait for initialization

    def open_gripper(self):
        """Open the gripper"""
        try:
            # Set digital output to open gripper
            self.robot.set_digital_out(0, True)  # Digital output 0 controls the gripper
            print("Opening gripper")
            logging.info("Gripper opened")
            return True
        except Exception as e:
            logging.error(f"Failed to open gripper: {str(e)}")
            print(f"Failed to open gripper: {str(e)}")
            return False

    def close_gripper(self):
        """Close the gripper"""
        try:
            # Set digital output to close gripper
            self.robot.set_digital_out(0, False)  # Digital output 0 controls the gripper
            print("Closing gripper")
            logging.info("Gripper closed")
            return True
        except Exception as e:
            logging.error(f"Failed to close gripper: {str(e)}")
            print(f"Failed to close gripper: {str(e)}")
            return False

    def close(self):
        """Cleanup and close the robot connection"""
        print("Closing robot connection")
        self.robot.close()
        logging.info("Robot connection closed")

def get_key():
    """Get a single keypress from the terminal"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    # Create instance of the gripper controller
    gripper = GripperController()

    print("\nUR Gripper Control")
    print("----------------------------------")
    print("Press 'o' to open the gripper")
    print("Press 'c' to close the gripper")
    print("Press 'q' to quit")
    print("----------------------------------")
    
    try:
        while True:
            key = get_key()
            
            if key == 'o':
                gripper.open_gripper()
            elif key == 'c':
                gripper.close_gripper()
            elif key == 'q':
                print("Exiting program.")
                break
                
    finally:
        # Cleanup
        gripper.close()

if __name__ == "__main__":
    main() 