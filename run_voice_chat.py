#!/usr/bin/env python

"""
UR3 Robot Voice Control with Chat UI - Optimized Version

A simple launcher script to start the voice-controlled robot application with a chat UI.
This version includes better error handling and performance optimization.
"""

import asyncio
import sys
import traceback
import os
from PyQt6.QtWidgets import QApplication, QMessageBox

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

def run_application():
    """Start the voice control application with error handling."""
    try:
        # Check if OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY") and not os.path.exists(".env"):
            print("WARNING: No OpenAI API key found in environment or .env file.")
            app = show_error_dialog("No OpenAI API key found. Please set OPENAI_API_KEY in your environment or .env file.")
            return
        
        # Import the main function after the API key check
        try:
            from ur_llm_control_voice import main
            asyncio.run(main())
        except ImportError as e:
            error_msg = f"Failed to import application modules: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            app = show_error_dialog(error_msg)
            return
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        app = show_error_dialog(error_msg)
        return
    finally:
        # Make sure to properly exit
        QApplication.instance().quit()

if __name__ == "__main__":
    try:
        run_application()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)