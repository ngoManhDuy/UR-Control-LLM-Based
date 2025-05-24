import time
import logging
import webrtcvad
import pyaudio
import wave
import threading
import queue
import tempfile
import os
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from openai import OpenAI

from voice_handler import VoiceHandler
from audio_noise_suppression import IndustrialNoiseSuppressor

class EnhancedVoiceHandler(VoiceHandler):
    """
    Enhanced Voice Handler with industrial noise suppression capabilities.
    Extends the base VoiceHandler to add noise filtering for pneumatic machines
    and air compressor environments.
    """
    def __init__(self):
        # Initialize the parent VoiceHandler
        super().__init__()
        
        # Initialize the noise suppressor
        self.noise_suppressor = IndustrialNoiseSuppressor(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size
        )
        
        # Flag for if calibration has been performed
        self.is_calibrated = False
        
        logging.info("Enhanced Voice Handler with noise suppression initialized")
    
    def calibrate_noise_profile(self, duration=5):
        """
        Record ambient noise to build a suppression profile.
        This should be done when only background industrial noise is present.
        """
        print(f"Starting noise calibration. Please ensure only the industrial noise is present...")
        
        # Start the audio stream if not already running
        self.start_audio_stream()
        
        # Calibrate using the current audio stream
        self.noise_suppressor.calibrate(self.stream, duration)
        
        # Set the flag to indicate calibration is done
        self.is_calibrated = True
        
        print("Noise calibration complete. Voice recognition should now be improved.")
        return True
        
    def start_audio_stream(self):
        """Start the audio stream for VAD with noise suppression."""
        # Use the parent method to start the stream
        super().start_audio_stream()
    
    def record_audio(self):
        """
        Record audio while speech is detected, with noise suppression applied.
        Overrides the parent method to add noise filtering.
        """
        # Pre-allocate buffer for better performance
        buffer_size = 20 * self.chunk_size
        frames = bytearray(buffer_size)
        total_bytes = 0
        
        silence_frames = 0
        max_silence_frames = 20  # About 0.6 seconds of silence to stop recording
        
        # Variables for tracking timeout
        start_time = time.time()
        speech_detected = False
        notification_sent = False
        
        while self.recording and not self.stop_event.is_set():
            try:
                # Read audio chunk
                audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Apply noise suppression if calibrated
                if self.is_calibrated:
                    audio_chunk = self.noise_suppressor.process_chunk(audio_chunk)
                
                chunk_len = len(audio_chunk)
                
                # Check if speech is detected using WebRTC VAD
                if self.is_speech(audio_chunk):
                    # Reset timeout counter when speech is detected
                    speech_detected = True
                    
                    # Extend buffer if needed - more efficient growth
                    if total_bytes + chunk_len > len(frames):
                        frames.extend(bytearray(chunk_len * 5))
                    
                    # Add chunk to buffer
                    frames[total_bytes:total_bytes+chunk_len] = audio_chunk
                    total_bytes += chunk_len
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames > max_silence_frames and speech_detected:
                        # Only break if we've already detected some speech followed by silence
                        break
                
                # Check for timeout if no speech has been detected yet
                if not speech_detected and not notification_sent:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= self.speech_timeout_seconds:
                        print("No speech detected. Please speak now.")
                        self.speak("I don't hear anything. Please speak louder or check your microphone.")
                        notification_sent = True
            
            except Exception as e:
                logging.error(f"Error recording audio: {e}")
                break
        
        # Return empty bytes if no speech was detected at all
        if not speech_detected:
            return b''
            
        return bytes(frames[:total_bytes])
    
    def listen_for_command(self):
        """
        Listen for voice command with noise suppression and return the recognized text.
        Overrides the parent method to add noise filtering.
        """
        # If not calibrated, suggest calibration first time
        if not self.is_calibrated:
            print("Noise profile not calibrated. For optimal performance in noisy environments,")
            print("consider calibrating first with the calibrate_noise_profile() method.")
        
        # Use the parent method for listening
        return super().listen_for_command()
    
    def filter_audio_file(self, input_file, output_file=None):
        """
        Apply noise suppression to an existing audio file.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file (if None, will use input_file_processed.wav)
        
        Returns:
            Path to the processed audio file
        """
        return self.noise_suppressor.process_file(input_file, output_file)
        
    def close(self):
        """Clean up resources."""
        # Clean up the noise suppressor
        if hasattr(self, 'noise_suppressor'):
            self.noise_suppressor.close()
            
        # Call parent close method
        super().close()
        
        logging.info("Enhanced Voice Handler closed") 