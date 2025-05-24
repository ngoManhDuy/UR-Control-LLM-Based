#!/usr/bin/env python

"""
UR3 Robot Voice Control with Noise Suppression - Industrial Environment Edition

This script launches the voice-controlled robot application with enhanced noise suppression
capabilities specifically designed for industrial environments with pneumatic machines
and air compressors.
"""

import asyncio
import sys
import traceback
import os
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer, QEventLoop

# Import the enhanced voice handler instead of standard voice handler
from enhanced_voice_handler import EnhancedVoiceHandler
from ur_llm_control_voice import URController, process_command
from chat_ui import ChatWindow


def show_error_dialog(error_msg):
    """Display a Qt error dialog with the specified message."""
    app = QApplication.instance() or QApplication(sys.argv)
    error_box = QMessageBox()
    error_box.setIcon(QMessageBox.Icon.Critical)
    error_box.setWindowTitle("Application Error")
    error_box.setText("An error occurred while starting the application:")
    error_box.setDetailedText(error_msg)
    error_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    error_box.exec()
    return app


async def calibrate_and_run(chat_window, ur_controller, voice_handler):
    """Calibrate noise profile and then run the main application loop"""
    # Add calibration notification
    chat_window.signals.add_status_message.emit("Initializing noise calibration...")
    
    # Ask if user wants to calibrate for noise
    app = QApplication.instance()
    calibration_box = QMessageBox()
    calibration_box.setIcon(QMessageBox.Icon.Question)
    calibration_box.setWindowTitle("Noise Calibration")
    calibration_box.setText("Would you like to calibrate for industrial noise?")
    calibration_box.setInformativeText("This will help filter out pneumatic machine and air compressor noise.")
    calibration_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    button = calibration_box.exec()
    
    # If yes, perform calibration
    if button == QMessageBox.StandardButton.Yes:
        chat_window.signals.add_status_message.emit(
            "Starting noise calibration. Please ensure only background machinery noise is present (no speech)."
        )
        
        # Countdown for preparation
        for i in range(5, 0, -1):
            chat_window.signals.add_status_message.emit(f"Calibrating in {i}...")
            await asyncio.sleep(1)
            
        # Perform calibration
        voice_handler.calibrate_noise_profile(duration=5)
        chat_window.signals.add_status_message.emit("Noise calibration complete!")
    
    # Start the main application loop
    chat_window.signals.add_status_message.emit("Waiting for voice commands...")
    
    # Initialize conversation history
    conversation_history = []

    # Main interaction loop
    while True:
        # Process voice input
        chat_window.signals.add_status_message.emit("Listening...")
        voice_text = voice_handler.listen_for_command()
        
        # Skip if no voice detected
        if not voice_text or voice_text.strip() == '':
            chat_window.signals.add_status_message.emit("No speech detected, try again...")
            continue
            
        # Add user input to chat history
        chat_window.signals.add_user_message.emit(voice_text)
        
        # Process the command
        response = await process_command(voice_text, ur_controller, voice_handler, chat_window)
        
        # Check for exit command
        if response.lower() == "goodbye":
            chat_window.signals.add_status_message.emit("Exiting program...")
            break
            
        # Give time for UI to process
        await asyncio.sleep(0.1)


async def enhanced_main():
    """Start the noise-suppressing voice control application."""
    try:
        # Initialize the PyQt application
        app = QApplication.instance() or QApplication(sys.argv)
        chat_window = ChatWindow()
        chat_window.show()
        
        # Initialize the UR controller
        ur_controller = URController(chat_window=chat_window)
        
        # Initialize the enhanced voice handler with noise suppression
        voice_handler = EnhancedVoiceHandler()
        
        # Start the main application loop with calibration
        await calibrate_and_run(chat_window, ur_controller, voice_handler)
        
        # Clean up resources
        ur_controller.close()
        voice_handler.close()
        app.quit()
        
    except Exception as e:
        error_msg = f"Error in main application: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        show_error_dialog(error_msg)


def run_application():
    """Start the enhanced voice control application with error handling."""
    try:
        # Check if OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY") and not os.path.exists(".env"):
            print("WARNING: No OpenAI API key found in environment or .env file.")
            app = show_error_dialog("No OpenAI API key found. Please set OPENAI_API_KEY in your environment or .env file.")
            return
        
        # Run the enhanced main function
        asyncio.run(enhanced_main())
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        app = show_error_dialog(error_msg)
        return


if __name__ == "__main__":
    try:
        run_application()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1) 