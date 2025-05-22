import urx
import time

# Robot IP
ROBOT_IP = "169.254.200.239"

# Map your gripper names to output pins
GRIPPER_DO = {
    "XL_kep": 1,  # close solenoid
    "XL_nha": 2,  # open solenoid
}

def pulse_output(robot, name, duration=0.5):
    """
    Pulse the named gripper output True for `duration` seconds,
    then switch it off again.
    """
    pin = GRIPPER_DO[name]
    robot.set_digital_out(pin, True)
    time.sleep(duration)
    robot.set_digital_out(pin, False)

def close_gripper(robot, duration=0.5):
    # ensure the opposite is off first
    robot.set_digital_out(GRIPPER_DO["XL_nha"], False)
    pulse_output(robot, "XL_kep", duration)

def open_gripper(robot, duration=0.5):
    robot.set_digital_out(GRIPPER_DO["XL_kep"], False)
    pulse_output(robot, "XL_nha", duration)

if __name__ == "__main__":
    robot = urx.Robot(ROBOT_IP)
    try:
        # close then open
        close_gripper(robot, duration=0.4)
        time.sleep(1.0)
        open_gripper(robot, duration=0.4)
    finally:
        robot.close()
