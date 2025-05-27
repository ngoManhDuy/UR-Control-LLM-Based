#!/usr/bin/env python3
import time
import os
import json
import math
import asyncio
import sys
import wave
import pyaudio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from URRobotControl import URRobotControl
from ur_robot_llm_functions import FUNCTION_DEFINITIONS
from capstone.audio_noise_suppression import IndustrialNoiseSuppressor
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

class VoiceInput:
    def __init__(self):
        self.chunk_size = 1024
        self.sample_rate = 16000
        self.record_seconds = 5
        self.noise_suppressor = IndustrialNoiseSuppressor(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size
        )
        self.audio = pyaudio.PyAudio()

    def record_voice(self):
        """Record voice command and apply noise suppression"""
        print("\nListening for voice command...")
        
        # Open audio stream
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        frames = []
        
        try:
            # Record audio
            for _ in range(0, int(self.sample_rate / self.chunk_size * self.record_seconds)):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                # Apply noise suppression in real-time
                filtered_data = self.noise_suppressor.process_chunk(data)
                frames.append(filtered_data)
                
        finally:
            stream.stop_stream()
            stream.close()

        # Save temporary WAV file
        temp_wav = "temp_voice_command.wav"
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))

        return temp_wav

    def transcribe_audio(self, audio_file):
        """Transcribe the recorded audio using OpenAI Whisper API"""
        try:
            with open(audio_file, "rb") as audio:
                client = AsyncOpenAI()
                # Create a new event loop for async operation
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio,
                        response_format="text"
                    )
                )
                loop.close()
                return response
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return None
        finally:
            # Clean up temporary file
            if os.path.exists(audio_file):
                os.remove(audio_file)

    def close(self):
        """Clean up resources"""
        self.audio.terminate()
        self.noise_suppressor.close()

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
    voice_input = VoiceInput()

    print("\nUR Robot Advanced Control with LLM Interface")
    print("-------------------------------------------")
    print("Choose input mode:")
    print("1. Text input (type commands)")
    print("2. Voice input (speak commands)")
    print("Type 'switch' to change modes")
    print("Type 'exit' to quit")
    print("-------------------------------------------")
    
    # Default to text input mode
    voice_mode = False
    
    try:
        while True:
            if voice_mode:
                print("\nVOICE MODE: Press Enter to start recording (5 seconds)...")
                input()  # Wait for Enter key
                
                # Record and process voice
                audio_file = voice_input.record_voice()
                command = voice_input.transcribe_audio(audio_file)
                
                if command:
                    print(f"\nTranscribed command: {command}")
                    await process_command(command, controller)
                else:
                    print("Failed to transcribe voice command. Please try again.")
            else:
                command = input("\nEnter command: ").strip()
            
            if command.lower() == 'exit':
                break
            elif command.lower() == 'switch':
                voice_mode = not voice_mode
                print(f"\nSwitched to {'voice' if voice_mode else 'text'} input mode")
                continue
                
            if not voice_mode:
                await process_command(command, controller)
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"Main loop error: {e}")
    finally:
        controller.close()
        voice_input.close()
        print("Program terminated.")

if __name__ == "__main__":
    asyncio.run(main()) 