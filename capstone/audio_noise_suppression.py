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
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

class IndustrialNoiseSuppressor:
    def __init__(self, 
                 sample_rate=16000,  # Denoiser works with 16kHz
                 chunk_size=1024,    # Larger chunks for better denoising
                 buffer_size=10):    # seconds of audio to keep in buffer
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_frames = int(buffer_size * (sample_rate / chunk_size))
        self.audio_buffer = deque(maxlen=self.buffer_frames)
        
        # Initialize denoiser model
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logging.info("Using GPU for denoising")
            else:
                self.device = torch.device('cpu')
                logging.info("Using CPU for denoising")
            
            self.model = pretrained.dns64().to(self.device)
            self.model.eval()  # Set to evaluation mode
            logging.info("Denoiser model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to initialize denoiser model: {e}")
            raise
        
        # Create temporary directory for file processing
        self.temp_dir = tempfile.mkdtemp()
        
        logging.info("Noise suppressor initialized with sample rate: %d Hz", sample_rate)
    
    def process_chunk(self, audio_chunk: bytes) -> bytes:
        """Process a single chunk of audio using Facebook Denoiser"""
        # Convert to float32 array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample to 16kHz if needed
        if self.sample_rate != 16000:
            audio_data = librosa.resample(audio_data, 
                                        orig_sr=self.sample_rate, 
                                        target_sr=16000)
        
        # Convert to torch tensor
        audio_tensor = torch.FloatTensor(audio_data).to(self.device)
        
        # Add batch and channel dimensions
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        
        try:
            # Process through denoiser
            with torch.no_grad():
                denoised_tensor = self.model(audio_tensor)
            
            # Remove batch and channel dimensions
            denoised = denoised_tensor.squeeze().cpu().numpy()
            
            # Resample back to original sample rate if needed
            if self.sample_rate != 16000:
                denoised = librosa.resample(denoised,
                                          orig_sr=16000,
                                          target_sr=self.sample_rate)
            
            # Convert back to int16
            denoised = np.clip(denoised * 32768.0, -32768, 32767).astype(np.int16)
            return denoised.tobytes()
            
        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            return audio_chunk
    
    def process_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        """Process an entire audio file using Facebook Denoiser"""
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_filtered{ext}"
        
        try:
            # Load audio file
            audio_data, sr = librosa.load(input_file, sr=16000, mono=True)
            
            # Convert to torch tensor
            audio_tensor = torch.FloatTensor(audio_data).to(self.device)
            
            # Add batch and channel dimensions
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            
            # Process through denoiser
            with torch.no_grad():
                denoised_tensor = self.model(audio_tensor)
            
            # Remove batch and channel dimensions
            denoised = denoised_tensor.squeeze().cpu().numpy()
            
            # Resample to original sample rate if needed
            if sr != 16000:
                denoised = librosa.resample(denoised, orig_sr=16000, target_sr=sr)
            
            # Save the output
            sf.write(output_file, denoised, sr)
            return output_file
            
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return input_file
    
    def close(self):
        """Clean up resources"""
        try:
            # Clean up temporary files
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
            
            # Clear GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Error cleaning up resources: {e}") 