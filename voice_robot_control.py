#!/usr/bin/env python3

import os
import json
import asyncio
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI
from URRobotControl import URRobotControl
from ur_robot_llm_functions import FUNCTION_DEFINITIONS
from pick_n_place.finding_object import ObjectFinder
from robot_safety_validator import RobotSafetyValidator
from robot_prompts import SYSTEM_MESSAGE, PLANNER_MESSAGE
import math
from typing import List, Dict, Any

# Import the STT module
from stt_whisper import STT_module

# Load environment variables
load_dotenv()

# Create OpenAI client
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class RobotAgent:
    """Agentic robot controller that can plan and execute sequential tasks"""
    
    def __init__(self, ur_controller):
        self.ur_controller = ur_controller
        self.current_task_plan = []
        self.task_index = 0
        
        # Initialize safety validator
        self.safety_validator = RobotSafetyValidator(ur_controller)
        
        # Store last found object for automatic use in move_down_to_pick
        self.last_found_object = None
        
    async def plan_tasks(self, user_command: str) -> List[Dict[str, Any]]:
        """Plan a sequence of tasks for the given command"""
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    PLANNER_MESSAGE,
                    {"role": "user", "content": user_command}
                ],
                temperature=0.1
            )
            
            # Parse the JSON response
            plan_text = response.choices[0].message.content.strip()
            task_plan = json.loads(plan_text)
            
            print(f"🎯 Generated task plan with {len(task_plan)} steps:")
            for i, task in enumerate(task_plan, 1):
                print(f"   {i}. {task['description']}")
            
            return task_plan
            
        except Exception as e:
            print(f"❌ Error planning tasks: {e}")
            # Fallback to single task execution
            return []
    
    async def execute_task_plan(self, task_plan: List[Dict[str, Any]]) -> bool:
        """Execute a planned sequence of tasks with safety validation"""
        if not task_plan:
            print("⚠️ No task plan provided, falling back to single function execution")
            return False
            
        print(f"🚀 Executing {len(task_plan)} tasks sequentially...")
        
        for i, task in enumerate(task_plan, 1):
            print(f"\n📋 Step {i}/{len(task_plan)}: {task['description']}")
            
            # Safety check before execution
            safety_result = await self.validate_task_safety(task['function'], task['parameters'])
            
            if not safety_result['safe']:
                print(f"🛑 SAFETY CHECK FAILED: {safety_result['reason']}")
                print(f"💡 Suggestion: {safety_result['suggestion']}")
                print(f"❌ Task {i} blocked for safety, stopping execution")
                return False
            elif safety_result.get('warning'):
                print(f"⚠️ SAFETY WARNING: {safety_result['reason']}")
                print(f"💡 Suggestion: {safety_result['suggestion']}")
                print("🤔 Proceeding with caution...")
            else:
                print(f"✅ Safety check passed: {safety_result['reason']}")
            
            success = await self.execute_single_task(
                task['function'], 
                task['parameters']
            )
            
            if not success:
                print(f"❌ Task {i} failed, stopping execution")
                return False
                
            # Small delay between tasks for safety
            await asyncio.sleep(0.5)
        
        print("✅ All tasks completed successfully!")
        return True
    
    async def validate_task_safety(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate safety of a specific task before execution"""
        try:
            if function_name == "move_to_pose":
                pose = [parameters["x"], parameters["y"], parameters["z"], 
                       parameters["rx"], parameters["ry"], parameters["rz"]]
                return self.safety_validator.validate_pose_movement(pose)
                
            elif function_name == "move_relative":
                return self.safety_validator.validate_relative_movement(
                    dx=parameters.get("dx", 0),
                    dy=parameters.get("dy", 0), 
                    dz=parameters.get("dz", 0)
                )
                
            elif function_name == "rotate_joint":
                return self.safety_validator.validate_joint_movement(
                    joint_index=parameters["joint_index"],
                    target_angle=parameters["angle"]
                )
                
            elif function_name == "rotate_joint_relative":
                # For relative movement, we need to get current position first
                current_joints = self.ur_controller.robot.get_joint_positions()
                if current_joints is None:
                    return {
                        'safe': False,
                        'reason': 'Cannot get current joint positions for safety validation',
                        'suggestion': 'Check robot connection and try again'
                    }
                
                joint_index = parameters["joint_index"]
                angle_change = parameters["angle_change"]
                current_angle_deg = math.degrees(current_joints[joint_index])
                target_angle_deg = current_angle_deg + angle_change
                
                return self.safety_validator.validate_joint_movement(
                    joint_index=joint_index,
                    target_angle=target_angle_deg
                )
                
            elif function_name == "move_joints":
                # Validate all joint angles
                for i, angle_key in enumerate(['j0', 'j1', 'j2', 'j3', 'j4', 'j5']):
                    if angle_key in parameters:
                        joint_result = self.safety_validator.validate_joint_movement(i, parameters[angle_key])
                        if not joint_result['safe']:
                            return joint_result
                return {'safe': True, 'reason': 'All joint movements are within safe limits'}
                
            elif function_name == "find_object":
                # Object finding is inherently safe - no safety check needed
                return {'safe': True, 'reason': 'Object finding is a safe operation that handles its own movements'}
                
            else:
                # Non-movement commands are generally safe
                return {'safe': True, 'reason': f'{function_name} is a non-movement command'}
                
        except Exception as e:
            return {
                'safe': False,
                'reason': f"Error during safety validation: {e}",
                'suggestion': "Check parameters and try again"
            }
    
    async def execute_single_task(self, function_name: str, parameters: Dict[str, Any]) -> bool:
        """Execute a single robot task"""
        try:
            print(f"🔧 Executing: {function_name} with {parameters}")
            
            if function_name == "move_to_pose":
                return self.ur_controller.robot.move_l(
                    pose=[parameters["x"], parameters["y"], parameters["z"], 
                          parameters["rx"], parameters["ry"], parameters["rz"]],
                    a=parameters.get("acceleration", 1.2),
                    v=parameters.get("speed", 0.25)
                )
                
            elif function_name == "move_joints":
                joints = [
                    math.radians(parameters["j0"]),
                    math.radians(parameters["j1"]),
                    math.radians(parameters["j2"]),
                    math.radians(parameters["j3"]),
                    math.radians(parameters["j4"]),
                    math.radians(parameters["j5"])
                ]
                return self.ur_controller.robot.move_j(
                    joints=joints,
                    a=parameters.get("acceleration", 1.4),
                    v=parameters.get("speed", 1.05)
                )
                
            elif function_name == "move_relative":
                return self.ur_controller.robot.move_relative(
                    dx=parameters.get("dx", 0),
                    dy=parameters.get("dy", 0),
                    dz=parameters.get("dz", 0),
                    drx=parameters.get("drx", 0),
                    dry=parameters.get("dry", 0),
                    drz=parameters.get("drz", 0),
                    a=parameters.get("acceleration", 1.2),
                    v=parameters.get("speed", 0.25)
                )
                
            elif function_name == "rotate_joint":
                return self.ur_controller.robot.move_joint(
                    joint_index=parameters["joint_index"],
                    angle=math.radians(parameters["angle"]),
                    a=parameters.get("acceleration", 1.4),
                    v=parameters.get("speed", 1.05)
                )
                
            elif function_name == "rotate_joint_relative":
                # Get current joint positions
                current_joints = self.ur_controller.robot.get_joint_positions()
                if current_joints is None:
                    print(f"❌ Failed to get current joint positions")
                    return False
                
                joint_index = parameters["joint_index"]
                angle_change = parameters["angle_change"]
                
                # Calculate new absolute angle
                current_angle_deg = math.degrees(current_joints[joint_index])
                new_angle_deg = current_angle_deg + angle_change
                
                print(f"🔄 Relative joint rotation: Joint {joint_index} from {current_angle_deg:.1f}° to {new_angle_deg:.1f}° (change: {angle_change:+.1f}°)")
                
                # Execute the movement to the new absolute position
                return self.ur_controller.robot.move_joint(
                    joint_index=joint_index,
                    angle=math.radians(new_angle_deg),
                    a=parameters.get("acceleration", 1.4),
                    v=parameters.get("speed", 1.05)
                )
                
            elif function_name == "control_gripper":
                if parameters["action"] == "open":
                    return self.ur_controller.robot.open_gripper(
                        duration=parameters.get("duration", 0.5)
                    )
                else:
                    return self.ur_controller.robot.close_gripper(
                        duration=parameters.get("duration", 0.5)
                    )
                    
            elif function_name == "get_robot_status":
                status = self.ur_controller.robot.get_status()
                print("🤖 Robot Status:", status)
                return True
                
            elif function_name == "stop_robot":
                return self.ur_controller.robot.stop()
                
            elif function_name == "find_object":
                if self.ur_controller.object_finder is None:
                    if not self.ur_controller.initialize_object_finder():
                        print("❌ Failed to initialize computer vision system")
                        return False
                
                scan_range = parameters.get('scan_range_degrees', 180)
                scan_speed = parameters.get('scan_speed', 0.1)
                check_interval = parameters.get('check_interval', 0.5)
                
                result = self.ur_controller.object_finder.find_object(
                    scan_range_degrees=scan_range,
                    scan_speed=scan_speed,
                    check_interval=check_interval
                )
                
                if result['success']:
                    print(f"🎉 Object found at position {result['object_position']}")
                    self.last_found_object = result # Store the result globally
                    return True
                else:
                    print(f"❌ Object search failed")
                    return False
                    
            elif function_name == "move_down_to_pick":
                if self.ur_controller.object_finder is None:
                    if not self.ur_controller.initialize_object_finder():
                        print("❌ Failed to initialize computer vision system")
                        return False
                
                # Extract object position from parameters or use last found object
                if "x" in parameters and "y" in parameters and "z" in parameters:
                    object_position = [parameters["x"], parameters["y"], parameters["z"]]
                    initial_confidence = parameters.get("confidence")
                elif self.last_found_object and self.last_found_object['success']:
                    # Use the last found object automatically
                    object_position = self.last_found_object['object_position']
                    initial_confidence = self.last_found_object['confidence']
                    print(f"🔄 Using last found object at {object_position}")
                else:
                    print("❌ No object coordinates provided and no previously found object")
                    return False
                
                result = self.ur_controller.object_finder.move_down_to_pick(
                    initial_object_position=object_position,
                    initial_confidence=initial_confidence
                )
                
                if result['success']:
                    print(f"🎯 Ready to pick at pose {result['pick_pose']}")
                    print(f"🎉 Final object position: {result['final_position']}")
                    print(f"🎯 Final confidence: {result['final_confidence']:.3f}")
                    return True
                else:
                    print(f"❌ Move-down-to-pick failed: {result['error']}")
                    return False
                    
            # Safety Function Implementations
            elif function_name == "check_movement_safety":
                movement_type = parameters["movement_type"]
                
                if movement_type == "pose":
                    pose = [parameters["x"], parameters["y"], parameters["z"], 
                           parameters["rx"], parameters["ry"], parameters["rz"]]
                    result = self.safety_validator.validate_pose_movement(pose)
                elif movement_type == "relative":
                    result = self.safety_validator.validate_relative_movement(
                        dx=parameters.get("dx", 0),
                        dy=parameters.get("dy", 0),
                        dz=parameters.get("dz", 0)
                    )
                elif movement_type == "joint":
                    result = self.safety_validator.validate_joint_movement(
                        joint_index=parameters["joint_index"],
                        target_angle=parameters["angle"]
                    )
                else:
                    result = {'safe': False, 'reason': f'Unknown movement type: {movement_type}'}
                
                # Print safety check result
                if result['safe']:
                    if result.get('warning'):
                        print(f"⚠️ SAFETY WARNING: {result['reason']}")
                        print(f"💡 {result['suggestion']}")
                    else:
                        print(f"✅ SAFETY CHECK PASSED: {result['reason']}")
                else:
                    print(f"🛑 SAFETY CHECK FAILED: {result['reason']}")
                    print(f"💡 {result['suggestion']}")
                
                return True  # Safety check function always "succeeds" (it just reports)
                
            elif function_name == "get_workspace_limits":
                limits = {
                    'workspace': self.safety_validator.workspace_limits,
                    'joint_limits': self.safety_validator.joint_limits,
                    'safety_margin': self.safety_validator.safety_margin
                }
                print("🔍 Robot Workspace and Joint Limits:")
                print(f"   📏 Workspace: X={limits['workspace']['x_min']:.2f} to {limits['workspace']['x_max']:.2f}m")
                print(f"               Y={limits['workspace']['y_min']:.2f} to {limits['workspace']['y_max']:.2f}m")
                print(f"               Z={limits['workspace']['z_min']:.2f} to {limits['workspace']['z_max']:.2f}m")
                print(f"   🔧 All joints: {limits['joint_limits'][0][0]}° to {limits['joint_limits'][0][1]}°")
                print(f"   🛡️ Safety margin: {limits['safety_margin']*100}cm")
                return True
                
            elif function_name == "validate_position":
                x, y, z = parameters["x"], parameters["y"], parameters["z"]
                pose = [x, y, z, 0, 0, 0]  # Dummy orientation for position check
                result = self.safety_validator.validate_pose_movement(pose)
                
                if result['safe']:
                    if result.get('warning'):
                        print(f"⚠️ Position ({x:.2f}, {y:.2f}, {z:.2f}) is reachable but: {result['reason']}")
                        print(f"💡 {result['suggestion']}")
                    else:
                        print(f"✅ Position ({x:.2f}, {y:.2f}, {z:.2f}) is within safe workspace")
                else:
                    print(f"❌ Position ({x:.2f}, {y:.2f}, {z:.2f}) is unsafe: {result['reason']}")
                    print(f"💡 {result['suggestion']}")
                
                return True  # Position validation always "succeeds" (it just reports)
                    
            elif function_name == "check_reachability":
                x, y, z = parameters["x"], parameters["y"], parameters["z"]
                rx, ry, rz = parameters["rx"], parameters["ry"], parameters["rz"]
                pose = [x, y, z, rx, ry, rz]
                
                result = self.safety_validator.check_reachability(pose)
                
                if result['reachable']:
                    if result.get('warning'):
                        print(f"⚠️ Position ({x:.2f}, {y:.2f}, {z:.2f}) is reachable but: {result['reason']}")
                    else:
                        print(f"✅ Robot CAN reach pose ({x:.2f}, {y:.2f}, {z:.2f}, {rx:.2f}, {ry:.2f}, {rz:.2f})")
                        if 'joint_angles' in result:
                            angles = result['joint_angles']
                            print(f"   🔧 Required joint angles: {[f'{a:.1f}°' for a in angles]}")
                else:
                    print(f"❌ Robot CANNOT reach pose ({x:.2f}, {y:.2f}, {z:.2f}, {rx:.2f}, {ry:.2f}, {rz:.2f})")
                    print(f"   Reason: {result['reason']}")
                    print(f"   💡 {result['suggestion']}")
                
                return True  # Reachability check always "succeeds" (it just reports)
                    
            else:
                print(f"❌ Unknown function: {function_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error executing {function_name}: {e}")
            return False

class URController:
    def __init__(self):
        """Initialize connection to the UR robot"""
        self.ROBOT_IP = '169.254.200.239'
        
        # Initialize robot connection
        print("Initializing UR Robot connection...")
        self.robot = URRobotControl(self.ROBOT_IP)
        
        # Initialize object finder (but don't connect yet)
        self.object_finder = None
        
        # Connect to the robot
        if not self.robot.connect():
            print("Failed to connect to robot")
            sys.exit(1)
            
        print("Robot initialized!")

    def initialize_object_finder(self, model_path='best.pt'):
        """Initialize and connect the object finder system"""
        if self.object_finder is None:
            print("Initializing computer vision system...")
            # Pass the existing robot connection to avoid duplicates
            self.object_finder = ObjectFinder(robot_instance=self.robot, model_path=model_path)
            
            # Connect camera only (robot already connected)
            camera_ok = self.object_finder.pose_estimator.estimator.connect()
            if camera_ok:
                print("✅ Computer vision system initialized successfully")
                return True
            else:
                print("❌ Failed to initialize camera system")
                return False
        return True

    def close(self):
        """Cleanup and close the robot connection."""
        print("Closing connections...")
        if self.object_finder:
            try:
                self.object_finder.pose_estimator.estimator.disconnect()
                print("Camera disconnected")
            except:
                pass
        
        self.robot.disconnect()
        print("Robot disconnected")

class VoiceRobotController:
    def __init__(self):
        """Initialize voice control system"""
        print("🎤 Initializing Voice-Controlled Robot System...")
        print("=" * 60)
        
        # Initialize robot controller
        self.ur_controller = URController()
        
        # Initialize robot agent for task planning and execution
        self.robot_agent = RobotAgent(self.ur_controller)
        
        # Initialize conversation memory
        self.conversation_history = [SYSTEM_MESSAGE]
        
        # Initialize STT module with denoising enabled
        def status_update(message):
            print(f"🔊 STT Status: {message}")
        
        print("🔊 Initializing Speech-to-Text system...")
        self.stt = STT_module(
            model_name="openai/whisper-medium",
            enable_denoising=True,  # Enable denoising
            status_callback=status_update
        )
        
        print("✅ Voice-controlled robot system ready!")
        
    def add_to_conversation(self, role, content):
        """Add a message to the conversation history"""
        self.conversation_history.append({"role": role, "content": content})
        
        # Keep conversation history manageable (last 10 messages + system message)
        if len(self.conversation_history) > 11:
            # Keep system message and last 10 messages
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-10:]
    
    def clear_conversation(self):
        """Clear conversation history except system message"""
        self.conversation_history = [SYSTEM_MESSAGE]
        print("🧠 Conversation memory cleared")
        
    def close(self):
        """Cleanup and close all connections"""
        print("🔌 Shutting down voice control system...")
        self.ur_controller.close()
        print("✅ Voice control system shutdown complete")

async def process_voice_command(command_text, voice_controller):
    """
    Process a voice command using the agentic approach with task planning.
    """
    print(f"🤖 Processing command: '{command_text}'")
    
    # Add user command to conversation history
    voice_controller.add_to_conversation("user", command_text)
    
    try:
        # First, try to plan the command as a sequence of tasks
        task_plan = await voice_controller.robot_agent.plan_tasks(command_text)
        
        if task_plan and len(task_plan) > 0:
            # Execute the planned tasks sequentially
            success = await voice_controller.robot_agent.execute_task_plan(task_plan)
            
            if success:
                voice_controller.add_to_conversation("system", f"Successfully executed {len(task_plan)} tasks")
            else:
                voice_controller.add_to_conversation("system", "Task execution failed")
        else:
            # Fallback to original single function approach for simple commands
            print("🔄 Falling back to single function execution...")
            await process_single_function_command(command_text, voice_controller)
            
    except Exception as e:
        print(f"❌ Error in command processing: {e}")
        voice_controller.add_to_conversation("system", f"Error processing command: {e}")

async def process_single_function_command(command_text, voice_controller):
    """
    Fallback to original single function calling approach
    """
    # Use conversation history
    messages = voice_controller.conversation_history.copy()

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            functions=FUNCTION_DEFINITIONS,
            function_call="auto"  
        )
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        return

    message = response.choices[0].message

    # Check if the response contains a function call
    if message.function_call:
        function_call = message.function_call
        function_name = function_call.name
        arguments = function_call.arguments
        print(f"🔧 Executing: {function_name} with arguments {arguments}")

        # Add function call to conversation history
        voice_controller.add_to_conversation("assistant", f"Executing {function_name} with parameters: {arguments}")

        try:
            args = json.loads(arguments)
            
            # Use the robot agent to execute the single task
            success = await voice_controller.robot_agent.execute_single_task(function_name, args)
            
            result_msg = f"✅ {function_name} completed" if success else f"❌ {function_name} failed"
            print(result_msg)
            voice_controller.add_to_conversation("system", f"Function execution result: {result_msg}")
                
        except Exception as e:
            error_msg = f"❌ Error executing function: {e}"
            print(error_msg)
            voice_controller.add_to_conversation("system", error_msg)
            
    elif message.content:  # If there's a normal response, print it
        print(f"🤖 Robot Assistant: {message.content}")
        # Add assistant response to conversation history
        voice_controller.add_to_conversation("assistant", message.content)
    else:
        print("❌ Received an empty response.")
        voice_controller.add_to_conversation("system", "Received empty response from LLM")

async def main():
    """Main voice control loop"""
    print("\n🎤 VOICE-CONTROLLED UR ROBOT SYSTEM")
    print("=" * 60)
    print("🗣️  Say voice commands to control the robot")
    print("📋 Examples:")
    print("   • 'Move the robot to position x zero point five, y zero point two, z zero point three'")
    print("   • 'Rotate joint zero to forty five degrees'")
    print("   • 'Open the gripper'")
    print("   • 'Find an object for me'")
    print("   • 'Search for objects until you find one'")
    print("   • 'What is the current robot status?'")
    print("   • 'Stop the robot'")
    print("🔊 Say 'exit', 'quit', or 'goodbye' to stop")
    print("=" * 60)
    print("💡 Features enabled:")
    print("   ✅ Voice Activity Detection (VAD)")
    print("   ✅ Audio Denoising")
    print("   ✅ Continuous Object Scanning")
    print("   ✅ Natural Language Processing")
    print("=" * 60)
    
    # Initialize voice control system
    try:
        voice_controller = VoiceRobotController()
    except Exception as e:
        print(f"❌ Failed to initialize voice control system: {e}")
        return
    
    try:
        print("\n🎤 Voice control active! Start speaking...")
        
        while True:
            print("\n" + "─" * 50)
            print("🎤 Ready for voice command...")
            
            # Record and transcribe voice command
            try:
                transcribed_text = voice_controller.stt.process_single_recording(max_duration=30)
                
                if transcribed_text is None:
                    print("⚠️  No speech detected, try again...")
                    continue
                
                # Clean up transcribed text
                transcribed_text = transcribed_text.strip()
                print(f"📝 Transcribed: '{transcribed_text}'")
                
                # Check for exit commands
                if transcribed_text.lower() in ['exit', 'quit', 'goodbye', 'stop listening', 'end program']:
                    print("👋 Voice command to exit detected. Goodbye!")
                    break
                
                # Skip empty or very short transcriptions
                if len(transcribed_text) < 3:
                    print("⚠️  Command too short, try again...")
                    continue
                
                # Process the voice command
                await process_voice_command(transcribed_text, voice_controller)
                
            except KeyboardInterrupt:
                print("\n🛑 Keyboard interrupt detected")
                break
            except Exception as e:
                print(f"❌ Error processing voice command: {e}")
                print("🔄 Continuing to listen for commands...")
                continue
            
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
    finally:
        # Cleanup
        print("\n🔌 Cleaning up...")
        voice_controller.close()
        print("✅ Voice control system shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Program terminated by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1) 