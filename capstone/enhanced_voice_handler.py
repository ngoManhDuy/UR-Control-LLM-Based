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
    and air compressor environments using RNNoise.
    """
    def __init__(self):
        # Initialize the parent VoiceHandler
        super().__init__()
        
        # Initialize the noise suppressor with 48kHz sample rate
        self.sample_rate = 48000  # Update to 48kHz for RNNoise
        self.chunk_size = 480     # 10ms chunks at 48kHz
        
        self.noise_suppressor = IndustrialNoiseSuppressor(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size
        )
        
        logging.info("Enhanced Voice Handler with RNNoise suppression initialized")
    
    def start_audio_stream(self):
        """Start the audio stream for VAD with noise suppression."""
        if self.stream is None:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
    
    def record_audio(self):
        """
        Record audio while speech is detected, with RNNoise suppression applied.
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
                
                # Apply RNNoise suppression
                filtered_chunk = self.noise_suppressor.process_chunk(audio_chunk)
                
                chunk_len = len(filtered_chunk)
                
                # Check if speech is detected using WebRTC VAD
                if self.is_speech(filtered_chunk):
                    # Reset timeout counter when speech is detected
                    speech_detected = True
                    
                    # Extend buffer if needed - more efficient growth
                    if total_bytes + chunk_len > len(frames):
                        frames.extend(bytearray(chunk_len * 5))
                    
                    # Add filtered chunk to buffer
                    frames[total_bytes:total_bytes+chunk_len] = filtered_chunk
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
        Listen for voice command with RNNoise suppression and return the recognized text.
        """
        return super().listen_for_command()
    
    def filter_audio_file(self, input_file, output_file=None):
        """
        Apply RNNoise suppression to an existing audio file.
        
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