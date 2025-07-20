#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import asyncio
import sys
from datetime import datetime
import queue
import io
import contextlib
from voice_robot_control import VoiceRobotController, process_voice_command

class RobotControlUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice-Controlled UR3 Robot System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Configure dark theme
        self.setup_dark_theme()
        
        # Initialize voice controller (will be created when starting)
        self.voice_controller = None
        self.is_listening = False
        self.listen_thread = None
        
        # Message queue for thread-safe UI updates
        self.message_queue = queue.Queue()
        
        # Create UI components
        self.create_widgets()
        
        # Start message queue processor
        self.process_message_queue()
        
    def setup_dark_theme(self):
        """Configure dark theme for the application"""
        # Configure ttk styles for dark theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='#ffffff')
        style.configure('TButton', background='#404040', foreground='#ffffff')
        style.map('TButton', 
                 background=[('active', '#505050')],
                 foreground=[('active', '#ffffff')])
        style.configure('Header.TLabel', 
                       background='#2b2b2b', 
                       foreground='#00ff88', 
                       font=('Arial', 12, 'bold'))
        
    def create_widgets(self):
        """Create and arrange all UI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Conversation panel
        self.create_conversation_panel(main_frame)
        
        # Right side - Control panel
        self.create_control_panel(main_frame)
        
    def create_conversation_panel(self, parent):
        """Create the conversation display panel on the left"""
        # Conversation frame (left side)
        conv_frame = ttk.Frame(parent, style='TFrame')
        conv_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Conversation header
        conv_header = ttk.Label(conv_frame, 
                               text="Chat", 
                               style='Header.TLabel')
        conv_header.pack(anchor=tk.W, pady=(0, 10))
        
        # Chat area with scrollbar
        self.conversation_text = scrolledtext.ScrolledText(
            conv_frame,
            wrap=tk.WORD,
            width=60,
            height=35,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Segoe UI', 11),
            insertbackground='#ffffff',
            selectbackground='#404040',
            selectforeground='#ffffff',
            state=tk.DISABLED,  # Make read-only
            cursor="arrow"
        )
        self.conversation_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for chat bubble styling
        # User messages (left side - green bubbles)
        self.conversation_text.tag_config('user_bubble', 
                                         background='#00ff88',
                                         foreground='#000000',
                                         font=('Segoe UI', 10),
                                         relief=tk.RAISED,
                                         borderwidth=1,
                                         lmargin1=20,
                                         lmargin2=20,
                                         rmargin=300,
                                         spacing1=8,
                                         spacing3=8,
                                         wrap=tk.WORD)
        
        # LLM messages (right side - blue bubbles)  
        self.conversation_text.tag_config('llm_bubble',
                                         background='#0078ff',
                                         foreground='#ffffff', 
                                         font=('Segoe UI', 10),
                                         relief=tk.RAISED,
                                         borderwidth=1,
                                         lmargin1=250,
                                         lmargin2=250,
                                         rmargin=20,
                                         spacing1=8,
                                         spacing3=8,
                                         wrap=tk.WORD)
        
        # Timestamp styling
        self.conversation_text.tag_config('timestamp', 
                                         foreground='#888888', 
                                         font=('Segoe UI', 8),
                                         justify=tk.CENTER,
                                         spacing1=5,
                                         spacing3=5)
        
        # Clear conversation button
        clear_btn = ttk.Button(conv_frame, 
                              text="Clear Chat", 
                              command=self.clear_conversation)
        clear_btn.pack(pady=(10, 0), anchor=tk.W)
        
    def create_control_panel(self, parent):
        """Create the control panel on the right"""
        # Control frame (right side)
        control_frame = ttk.Frame(parent, style='TFrame')
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Control header
        control_header = ttk.Label(control_frame, 
                                  text="Voice Control", 
                                  style='Header.TLabel')
        control_header.pack(anchor=tk.W, pady=(0, 20))
        
        # Voice control buttons
        self.create_voice_controls(control_frame)
        
        # System status
        self.create_status_panel(control_frame)
        
    def create_voice_controls(self, parent):
        """Create voice control buttons"""
        voice_frame = ttk.LabelFrame(parent, text="Voice Control", padding=10)
        voice_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Start/Stop voice control button
        self.voice_btn = ttk.Button(voice_frame, 
                                   text="Start Voice Control", 
                                   command=self.toggle_voice_control,
                                   width=25)
        self.voice_btn.pack(pady=5)
        
        # Voice status label
        self.voice_status = ttk.Label(voice_frame, 
                                     text="Status: Stopped", 
                                     style='TLabel')
        self.voice_status.pack(pady=5)
        
        # Manual voice input
        ttk.Label(voice_frame, text="Manual Command:", style='TLabel').pack(anchor=tk.W, pady=(10, 0))
        
        self.manual_input = tk.Entry(voice_frame, 
                                    width=30, 
                                    bg='#404040', 
                                    fg='#ffffff',
                                    insertbackground='#ffffff')
        self.manual_input.pack(fill=tk.X, pady=5)
        self.manual_input.bind('<Return>', self.process_manual_command)
        
        manual_btn = ttk.Button(voice_frame, 
                               text="Send Command", 
                               command=self.process_manual_command)
        manual_btn.pack(pady=5)
        
    def create_status_panel(self, parent):
        """Create system status panel"""
        status_frame = ttk.LabelFrame(parent, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Robot connection status
        self.robot_status = ttk.Label(status_frame, 
                                     text="Robot: Disconnected", 
                                     style='TLabel')
        self.robot_status.pack(anchor=tk.W, pady=2)
        
        # STT status
        self.stt_status = ttk.Label(status_frame, 
                                   text="STT: Not initialized", 
                                   style='TLabel')
        self.stt_status.pack(anchor=tk.W, pady=2)
        
        # Safety status
        self.safety_status = ttk.Label(status_frame, 
                                      text="Safety: Ready", 
                                      style='TLabel')
        self.safety_status.pack(anchor=tk.W, pady=2)
        
    def add_message(self, sender, message, msg_type='normal'):
        """Add a message to the chat display with bubble styling"""
        timestamp = datetime.now().strftime("%H:%M")
        
        # Only show user and LLM messages (filter out most system messages)
        if msg_type in ['system', 'success'] and 'error' not in message.lower() and 'failed' not in message.lower():
            return  # Skip system/success messages
        
        # Enable text widget for editing
        self.conversation_text.config(state=tk.NORMAL)
        
        # Add some spacing between messages
        if self.conversation_text.get("end-2c", "end-1c") != "\n":
            self.conversation_text.insert(tk.END, "\n")
        
        if msg_type == 'user':
            # User message (left side - green bubble)
            self.conversation_text.insert(tk.END, f"{message}\n", 'user_bubble')
            # Add timestamp below user message (small, left-aligned)
            self.conversation_text.insert(tk.END, f"{timestamp}\n\n", 'timestamp')
        else:
            # LLM/Robot message (right side - blue bubble)
            # Handle different message types but all go to right side
            if msg_type == 'error':
                display_message = f"Error: {message}"
            elif 'error' in message.lower() or 'failed' in message.lower():
                display_message = f"Error: {message}"
            else:
                display_message = message
                
            self.conversation_text.insert(tk.END, f"{display_message}\n", 'llm_bubble')
            # Add timestamp below LLM message (small, right-aligned)
            self.conversation_text.insert(tk.END, f"{timestamp}\n\n", 'timestamp')
        
        # Auto-scroll to bottom
        self.conversation_text.see(tk.END)
        
        # Make text widget read-only again
        self.conversation_text.config(state=tk.DISABLED)
        
    def clear_conversation(self):
        """Clear the chat history"""
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear the chat history?"):
            self.conversation_text.config(state=tk.NORMAL)
            self.conversation_text.delete(1.0, tk.END)
            self.conversation_text.config(state=tk.DISABLED)
            
    def toggle_voice_control(self):
        """Toggle voice control on/off"""
        if not self.is_listening:
            self.start_voice_control()
        else:
            self.stop_voice_control()
            
    def start_voice_control(self):
        """Start voice control system"""
        try:
            self.add_message("", "Initializing voice control system...", 'system')
            
            # Initialize voice controller in separate thread
            self.listen_thread = threading.Thread(target=self.initialize_and_listen, daemon=True)
            self.listen_thread.start()
            
        except Exception as e:
            self.add_message("", f"Failed to start voice control: {e}", 'error')
            
    def initialize_and_listen(self):
        """Initialize voice controller and start listening (runs in separate thread)"""
        try:
            # Initialize voice controller
            self.voice_controller = VoiceRobotController()
            
            # Update UI
            self.message_queue.put(('ui_update', 'voice_started'))
            self.message_queue.put(('message', ('', 'Voice control ready! Start speaking to me...', 'robot')))
            
            self.is_listening = True
            
            # Voice listening loop
            while self.is_listening:
                try:
                    # Record and transcribe voice
                    transcribed_text = self.voice_controller.stt.process_single_recording(max_duration=30)
                    
                    if transcribed_text is None:
                        continue
                    
                    transcribed_text = transcribed_text.strip()
                    if len(transcribed_text) < 3:
                        continue
                    
                    # Check for exit commands
                    if transcribed_text.lower() in ['exit', 'quit', 'goodbye', 'stop listening']:
                        self.message_queue.put(('message', ('', transcribed_text, 'user')))
                        self.message_queue.put(('message', ('', 'Goodbye! Voice control stopped.', 'robot')))
                        break
                    
                    # Add user message to conversation
                    self.message_queue.put(('message', ('', transcribed_text, 'user')))
                    
                    # Process command
                    asyncio.run(self.process_voice_command_async(transcribed_text))
                    
                except Exception as e:
                    self.message_queue.put(('message', ('', f"Sorry, I had trouble processing that: {e}", 'error')))
                    
        except Exception as e:
            self.message_queue.put(('message', ('', f"Failed to initialize voice control: {e}", 'error')))
        finally:
            self.is_listening = False
            self.message_queue.put(('ui_update', 'voice_stopped'))
            
    async def process_voice_command_for_ui(self, command_text, voice_controller):
        """
        Custom voice command processor that returns responses for UI display.
        Note: Does NOT add to conversation history - that's handled by the calling function.
        """
        from robot_prompts import SYSTEM_MESSAGE, PLANNER_MESSAGE
        from openai import AsyncOpenAI
        from ur_robot_llm_functions import FUNCTION_DEFINITIONS
        import os
        import json
        
        # Create OpenAI client
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        try:
            # First, try to plan the command as a sequence of tasks
            task_plan = await voice_controller.robot_agent.plan_tasks(command_text)
            
            if task_plan and len(task_plan) > 0:
                # First, show just the plan
                plan_message = f"Sure! I've got a plan with {len(task_plan)} steps:\n"
                for i, task in enumerate(task_plan, 1):
                    plan_message += f"   {i}. {task['description']}\n"
                plan_message += f"\nAlright, let me get started..."
                
                # Send the plan first
                self.message_queue.put(('message', ('', plan_message, 'robot')))
                # Add plan to conversation history
                self.voice_controller.add_to_conversation("assistant", plan_message)
                
                # Then execute the tasks
                success, execution_details = await self.execute_task_plan_consolidated(voice_controller, task_plan)
                
                # Finally, send the execution results as a separate message
                if success:
                    result_message = f"Perfect! I've completed all {len(task_plan)} steps successfully!"
                else:
                    result_message = f"Hmm, ran into an issue:\n{execution_details}"
                
                # Send the results
                self.message_queue.put(('message', ('', result_message, 'robot')))
                # Add result to conversation history
                self.voice_controller.add_to_conversation("assistant", result_message)
                
                # Return None since we already sent messages via queue
                return None
            else:
                # Fallback to original single function approach for simple commands
                print(f"🔄 No task plan generated, falling back to single function execution...")
                
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
                    return f"Sorry, I had trouble understanding: {e}"

                message = response.choices[0].message

                # Check if the response contains a function call
                if message.function_call:
                    function_call = message.function_call
                    function_name = function_call.name
                    arguments = function_call.arguments

                    try:
                        args = json.loads(arguments)
                        
                        # Check if this is a movement function that needs safety validation
                        movement_functions = ['move_to_pose', 'move_joints', 'move_relative', 'rotate_joint']
                        safety_msg = ""
                        if function_name in movement_functions:
                            # Perform safety check first
                            safety_result = await voice_controller.robot_agent.validate_task_safety(function_name, args)
                            
                            if not safety_result['safe']:
                                return f"Hold on - I can't do that safely right now.\n{safety_result['reason']}\n{safety_result['suggestion']}"
                            elif safety_result.get('warning'):
                                safety_msg = f"Just so you know: {safety_result['reason']}\n"
                        
                        # Use the robot agent to execute the single task
                        success = await voice_controller.robot_agent.execute_single_task(function_name, args)
                        
                        if success:
                            return f"{safety_msg}All done! I successfully executed {function_name}."
                        else:
                            return f"{safety_msg}Sorry, {function_name} didn't work out as planned."
                            
                    except Exception as e:
                        return f"Oops, ran into an issue: {e}"
                        
                elif message.content:  # If there's a normal response, return it
                    # Normal response from LLM (conversation history handled in calling function)
                    return message.content
                else:
                    return "I'm not quite sure how to help with that one."
                
        except Exception as e:
            return f"Sorry, something went wrong: {e}"
    
    async def execute_task_plan_consolidated(self, voice_controller, task_plan):
        """
        Execute task plan and return success status
        
        Returns:
            tuple: (success: bool, error_message: str if failed)
        """
        try:
            for i, task in enumerate(task_plan, 1):
                function_name = task['function']
                parameters = task['parameters']
                description = task['description']
                
                print(f"🔧 Executing step {i}/{len(task_plan)}: {function_name}")
                
                # Execute the task
                if function_name == "check_movement_safety":
                    # This is a safety check task
                    movement_type = parameters["movement_type"]
                    
                    if movement_type == "pose":
                        pose = [parameters["x"], parameters["y"], parameters["z"], 
                               parameters["rx"], parameters["ry"], parameters["rz"]]
                        result = voice_controller.robot_agent.safety_validator.validate_pose_movement(pose)
                    elif movement_type == "relative":
                        result = voice_controller.robot_agent.safety_validator.validate_relative_movement(
                            dx=parameters.get("dx", 0),
                            dy=parameters.get("dy", 0),
                            dz=parameters.get("dz", 0)
                        )
                    elif movement_type == "joint":
                        result = voice_controller.robot_agent.safety_validator.validate_joint_movement(
                            joint_index=parameters["joint_index"],
                            target_angle=parameters["angle"]
                        )
                    else:
                        result = {'safe': False, 'reason': f'Unknown movement type: {movement_type}'}
                    
                    # Handle safety check result
                    if not result['safe']:
                        return False, f"Safety check failed: {result['reason']}"
                    elif result.get('warning'):
                        print(f"⚠️ Safety warning: {result['reason']}")
                    
                    # Safety check passed, continue to next task
                    continue
                
                else:
                    # This is a regular task - execute it
                    # Validate parameters before execution
                    try:
                        # Check for invalid string values in numeric parameters
                        for param_name, param_value in parameters.items():
                            if param_name in ['speed', 'acceleration', 'duration', 'angle'] and isinstance(param_value, str):
                                return False, f"Invalid parameter: {param_name} must be numeric, got '{param_value}'"
                        
                        success = await voice_controller.robot_agent.execute_single_task(function_name, parameters)
                        
                        if not success:
                            return False, f"Task {i} ({function_name}) failed"
                    except Exception as e:
                        return False, f"Execution error in step {i}: {str(e)}"
            
            # All tasks completed successfully
            return True, ""
            
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    async def execute_task_plan_with_safety(self, voice_controller, task_plan):
        """
        Execute task plan with proper safety checking integration
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            self.message_queue.put(('message', ('', f"🚀 Executing {len(task_plan)} tasks sequentially...", 'robot')))
            
            safety_failed = False
            safety_message = ""
            
            for i, task in enumerate(task_plan, 1):
                function_name = task['function']
                parameters = task['parameters']
                description = task['description']
                
                # Show current step
                self.message_queue.put(('message', ('', f"📋 Step {i}/{len(task_plan)}: {description}", 'robot')))
                
                # Execute the task
                if function_name == "check_movement_safety":
                    # This is a safety check task
                    movement_type = parameters["movement_type"]
                    
                    if movement_type == "pose":
                        pose = [parameters["x"], parameters["y"], parameters["z"], 
                               parameters["rx"], parameters["ry"], parameters["rz"]]
                        result = voice_controller.robot_agent.safety_validator.validate_pose_movement(pose)
                    elif movement_type == "relative":
                        result = voice_controller.robot_agent.safety_validator.validate_relative_movement(
                            dx=parameters.get("dx", 0),
                            dy=parameters.get("dy", 0),
                            dz=parameters.get("dz", 0)
                        )
                    elif movement_type == "joint":
                        result = voice_controller.robot_agent.safety_validator.validate_joint_movement(
                            joint_index=parameters["joint_index"],
                            target_angle=parameters["angle"]
                        )
                    else:
                        result = {'safe': False, 'reason': f'Unknown movement type: {movement_type}'}
                    
                    # Handle safety check result
                    if not result['safe']:
                        safety_failed = True
                        safety_message = f"🛑 SAFETY CHECK FAILED: {result['reason']}. {result['suggestion']}"
                        self.message_queue.put(('message', ('', safety_message, 'robot')))
                        return False, safety_message
                    elif result.get('warning'):
                        # Send warning to chat but continue
                        self.message_queue.put(('message', ('', f"⚠️ SAFETY WARNING: {result['reason']}", 'robot')))
                    else:
                        # Safety check passed
                        self.message_queue.put(('message', ('', f"✅ Safety check passed: {result['reason']}", 'robot')))
                    
                    # Safety check passed, continue to next task
                    continue
                
                else:
                    # This is a regular task - execute it
                    self.message_queue.put(('message', ('', f"🔧 Executing: {function_name} with {parameters}", 'robot')))
                    
                    success = await voice_controller.robot_agent.execute_single_task(function_name, parameters)
                    
                    if not success:
                        error_msg = f"❌ Task {i} ({function_name}) failed"
                        self.message_queue.put(('message', ('', error_msg, 'robot')))
                        return False, error_msg
                    else:
                        self.message_queue.put(('message', ('', f"✅ Task {i} completed successfully", 'robot')))
            
            success_msg = f"✅ All {len(task_plan)} tasks completed successfully!"
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Error during task execution: {e}"
            self.message_queue.put(('message', ('', f"❌ {error_msg}", 'robot')))
            return False, error_msg
    
    async def process_voice_command_async(self, command_text):
        """Process voice command asynchronously"""
        try:
            # Add user command to conversation history
            self.voice_controller.add_to_conversation("user", command_text)
            
            # Debug: Show conversation history length
            history_length = len(self.voice_controller.conversation_history)
            print(f"🧠 Conversation history has {history_length} messages")
            
            # Use our custom UI-friendly command processor
            response = await self.process_voice_command_for_ui(command_text, self.voice_controller)
            
            # Display the response in the chat (if there is one - might be None if already sent via queue)
            if response:
                self.message_queue.put(('message', ('', response, 'robot')))
                # IMPORTANT: Add assistant response to conversation history too
                self.voice_controller.add_to_conversation("assistant", response)
                
                # Debug: Show updated conversation history length
                updated_length = len(self.voice_controller.conversation_history)
                print(f"🧠 After response, conversation history has {updated_length} messages")
            else:
                # Messages were already sent via queue and added to conversation history
                updated_length = len(self.voice_controller.conversation_history)
                print(f"🧠 After processing, conversation history has {updated_length} messages")
            
        except Exception as e:
            error_msg = f"Sorry, I couldn't complete that command: {e}"
            self.message_queue.put(('message', ('', error_msg, 'error')))
            # Add error to conversation history as well
            self.voice_controller.add_to_conversation("assistant", error_msg)
            
    def stop_voice_control(self):
        """Stop voice control system"""
        self.is_listening = False
        if self.voice_controller:
            self.voice_controller.close()
            self.voice_controller = None
        self.add_message("", "Voice control stopped. Use manual commands or restart voice control.", 'robot')
        self.update_voice_button(False)
        
    def process_manual_command(self, event=None):
        """Process manually typed command"""
        command = self.manual_input.get().strip()
        if command:
            self.manual_input.delete(0, tk.END)
            self.execute_voice_command(command)
            
    def execute_voice_command(self, command):
        """Execute a voice command"""
        if not self.voice_controller:
            self.add_message("", "Voice control not initialized. Start voice control first.", 'error')
            return
            
        self.add_message("", command, 'user')
        
        # Execute in separate thread
        thread = threading.Thread(target=lambda: asyncio.run(self.process_voice_command_async(command)), daemon=True)
        thread.start()
        
    def update_voice_button(self, is_active):
        """Update voice control button state"""
        if is_active:
            self.voice_btn.config(text="Stop Voice Control")
            self.voice_status.config(text="Status: Listening...")
        else:
            self.voice_btn.config(text="Start Voice Control")
            self.voice_status.config(text="Status: Stopped")
            
    def process_message_queue(self):
        """Process messages from worker threads (runs on main thread)"""
        try:
            while True:
                message_type, data = self.message_queue.get_nowait()
                
                if message_type == 'message':
                    sender, text, msg_type = data
                    self.add_message(sender, text, msg_type)
                elif message_type == 'ui_update':
                    if data == 'voice_started':
                        self.update_voice_button(True)
                        self.robot_status.config(text="Robot: Connected")
                        self.stt_status.config(text="STT: Active")
                    elif data == 'voice_stopped':
                        self.update_voice_button(False)
                        self.robot_status.config(text="Robot: Disconnected")
                        self.stt_status.config(text="STT: Stopped")
                        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_message_queue)
        
    def on_closing(self):
        """Handle application closing"""
        if self.is_listening:
            self.stop_voice_control()
        self.root.destroy()

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = RobotControlUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main() 