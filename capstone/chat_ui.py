import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                            QFrame, QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QSize
from PyQt6.QtGui import QFont, QPixmap

class ChatSignals(QObject):
    """Signals for updating the chat window from other threads"""
    add_user_message = pyqtSignal(str)
    add_robot_message = pyqtSignal(str)
    add_status_message = pyqtSignal(str)

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Create signals for thread-safe updates
        self.signals = ChatSignals()
        
        # Connect signals to slots
        self.signals.add_user_message.connect(self.display_user_message)
        self.signals.add_robot_message.connect(self.display_robot_message)
        self.signals.add_status_message.connect(self.display_status_message)
        
        # Set up the UI
        self.setWindowTitle("UR3 Robot Voice Control")
        self.setGeometry(100, 100, 800, 600)
        
        # Set dark theme application style
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #ecf0f1;
            }
            QTextEdit {
                background-color: #2d2d44;
                color: #ecf0f1;
                border: 1px solid #555555;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #2980b9;
                color: white;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton#clearButton {
                background-color: #2c3e50;
            }
            QPushButton#quitButton {
                background-color: #c0392b;
            }
        """)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Add header with logo
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add logo - use absolute path to improve loading
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icslab_logo.jpg")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            logo_pixmap = QPixmap(logo_path)
            # Use fixed size scaling to improve performance
            logo_pixmap = logo_pixmap.scaled(150, 50, Qt.AspectRatioMode.KeepAspectRatio, 
                                            Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setFixedSize(150, 50)  # Fix size to prevent layout recalculations
            header_layout.addWidget(logo_label)
        else:
            # Fallback if logo is not found
            logo_label = QLabel("ICSLab")
            logo_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            logo_label.setStyleSheet("color: #3498db;")
            header_layout.addWidget(logo_label)
        
        # Add title
        title_label = QLabel("UR3 Robot Voice Control System")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #3498db;")
        header_layout.addWidget(title_label, 1)
        
        # Add header to main layout
        main_layout.addLayout(header_layout)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #555555;")
        main_layout.addWidget(separator)
        
        # Create chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 11))  # Slightly smaller font for better performance
        self.chat_display.setMinimumHeight(350)
        
        # Optimize the text edit for better performance
        self.chat_display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.chat_display.setUndoRedoEnabled(False)  # Disable undo/redo for better performance
        self.chat_display.document().setDocumentMargin(8)
        
        # Use simpler CSS styling for better rendering performance
        self.chat_display.setStyleSheet("background-color: #2d2d44; color: #ecf0f1; border: 1px solid #555555; border-radius: 5px;")
        
        main_layout.addWidget(self.chat_display, 1)
        
        # Create status bar with simpler styling
        self.statusBar().setFont(QFont("Arial", 10))
        self.statusBar().setStyleSheet("background-color: #2d2d44; color: #a4b0be;")
        self.statusBar().showMessage("Ready")
        
        # Create button layout with simpler styling
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 5, 0, 0)
        button_layout.setSpacing(10)
        
        # Create clear button with simpler styling
        clear_button = QPushButton("Clear Chat")
        clear_button.setObjectName("clearButton")
        clear_button.setFont(QFont("Arial", 11))
        clear_button.setFixedHeight(35)  # Fixed height for better layout performance
        clear_button.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_button.setStyleSheet("background-color: #2c3e50; color: white; border-radius: 4px;")
        clear_button.clicked.connect(self.clear_chat)
        
        # Create quit button with simpler styling
        quit_button = QPushButton("Quit")
        quit_button.setObjectName("quitButton")
        quit_button.setFont(QFont("Arial", 11))
        quit_button.setFixedHeight(35)  # Fixed height for better layout performance
        quit_button.setCursor(Qt.CursorShape.PointingHandCursor)
        quit_button.setStyleSheet("background-color: #c0392b; color: white; border-radius: 4px;")
        quit_button.clicked.connect(self.close)
        
        # Add buttons to layout with spacer for right alignment
        button_layout.addStretch(1)
        button_layout.addWidget(clear_button)
        button_layout.addWidget(quit_button)
        main_layout.addLayout(button_layout)
        
        # Limit the number of messages in the chat display to prevent performance degradation
        self.max_messages = 100
        self.message_count = 0
        
        # Initialize with welcome message
        self.display_status_message("Welcome to UR3 Robot Voice Control")
    
    def display_user_message(self, message):
        """Display a message from the user with simpler HTML for better performance"""
        self.chat_display.append(f"<p style='margin:5px 0'><b style='color:#3498db'>You:</b> <span style='background-color:#364156;padding:3px;border-radius:3px;color:#ecf0f1'>{message}</span></p>")
        self._manage_message_count()
        self.chat_display.ensureCursorVisible()
    
    def display_robot_message(self, message):
        """Display a message from the robot with simpler HTML for better performance"""
        self.chat_display.append(f"<p style='margin:5px 0;text-align:right'><b style='color:#2ecc71'>Robot:</b> <span style='background-color:#2c3c3f;padding:3px;border-radius:3px;color:#ecf0f1'>{message}</span></p>")
        self._manage_message_count()
        self.chat_display.ensureCursorVisible()
    
    def display_status_message(self, message):
        """Display a status message with simpler HTML for better performance"""
        self.chat_display.append(f"<p style='margin:5px 0;color:#a4b0be;font-style:italic;text-align:center'>{message}</p>")
        self._manage_message_count()
        self.statusBar().showMessage(message)
        self.chat_display.ensureCursorVisible()
    
    def _manage_message_count(self):
        """Manage the number of messages to prevent performance degradation"""
        self.message_count += 1
        if self.message_count > self.max_messages:
            # Clear half of the oldest messages when limit is reached
            doc = self.chat_display.document()
            cursor = self.chat_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(cursor.MoveOperation.Down, cursor.MoveMode.KeepAnchor, self.max_messages // 2)
            cursor.removeSelectedText()
            self.message_count = self.max_messages // 2
    
    def clear_chat(self):
        """Clear the chat display"""
        self.chat_display.clear()
        self.message_count = 0
        self.display_status_message("Chat cleared")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())