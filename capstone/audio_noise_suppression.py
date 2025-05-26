import numpy as np
import scipy.signal as signal
import librosa
import soundfile as sf
import logging
import time
import threading
from collections import deque
import pyaudio
import wave
import tempfile
import os
from typing import Optional
from .rnnoise_wrapper import RNNoiseWrapper

class IndustrialNoiseSuppressor:
    def __init__(self, 
                 sample_rate=48000,  # RNNoise requires 48kHz
                 chunk_size=480,     # 10ms chunks at 48kHz
                 buffer_size=10):    # seconds of audio to keep in buffer
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_frames = int(buffer_size * (sample_rate / chunk_size))
        self.audio_buffer = deque(maxlen=self.buffer_frames)
        
        # Initialize RNNoise
        try:
            self.denoiser = RNNoiseWrapper()
            logging.info("RNNoise initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize RNNoise: {e}")
            raise
        
        # Create temporary directory for file processing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "temp_audio.wav")
        
        logging.info("Noise suppressor initialized with sample rate: %d Hz", sample_rate)
    
    def process_chunk(self, audio_chunk: bytes) -> bytes:
        """Process a single chunk of audio using RNNoise"""
        # Convert to float32 array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        
        # RNNoise expects 48kHz sample rate
        if self.sample_rate != 48000:
            # Resample to 48kHz
            audio_data = librosa.resample(audio_data, 
                                        orig_sr=self.sample_rate, 
                                        target_sr=48000)
        
        # Process through RNNoise in chunks of 480 samples
        denoised_chunks = []
        for i in range(0, len(audio_data), self.denoiser.frame_size):
            chunk = audio_data[i:i + self.denoiser.frame_size]
            
            # If this is the last chunk and it's incomplete, pad with zeros
            if len(chunk) < self.denoiser.frame_size:
                chunk = np.pad(chunk, (0, self.denoiser.frame_size - len(chunk)))
            
            # Process through RNNoise
            denoised_chunk, vad_prob = self.denoiser.process_frame(chunk)
            denoised_chunks.append(denoised_chunk)
        
        # Concatenate all chunks
        denoised = np.concatenate(denoised_chunks)
        
        # Resample back to original sample rate if needed
        if self.sample_rate != 48000:
            denoised = librosa.resample(denoised,
                                      orig_sr=48000,
                                      target_sr=self.sample_rate)
        
        # Convert back to int16
        denoised = np.clip(denoised * 32768.0, -32768, 32767).astype(np.int16)
        return denoised.tobytes()
    
    def process_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        """Process an entire audio file using RNNoise"""
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_filtered{ext}"
        
        # Load audio file
        audio_data, sr = librosa.load(input_file, sr=48000, mono=True)  # RNNoise requires 48kHz
        
        # Process through RNNoise in chunks
        denoised_chunks = []
        for i in range(0, len(audio_data), self.denoiser.frame_size):
            chunk = audio_data[i:i + self.denoiser.frame_size]
            
            # If this is the last chunk and it's incomplete, pad with zeros
            if len(chunk) < self.denoiser.frame_size:
                chunk = np.pad(chunk, (0, self.denoiser.frame_size - len(chunk)))
            
            # Process through RNNoise
            denoised_chunk, _ = self.denoiser.process_frame(chunk)
            
            # If this was a padded chunk, only keep the valid part
            if i + self.denoiser.frame_size > len(audio_data):
                denoised_chunk = denoised_chunk[:len(audio_data) - i]
            
            denoised_chunks.append(denoised_chunk)
        
        # Concatenate all chunks
        denoised = np.concatenate(denoised_chunks)
        
        # Resample to original sample rate if needed
        if sr != 48000:
            denoised = librosa.resample(denoised, orig_sr=48000, target_sr=sr)
        
        # Save the output
        sf.write(output_file, denoised, sr)
        
        return output_file
    
    def close(self):
        """Clean up resources"""
        try:
            # Close RNNoise
            if hasattr(self, 'denoiser'):
                self.denoiser = None
            
            # Clean up temporary files
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except Exception as e:
            logging.error(f"Error cleaning up resources: {e}") 