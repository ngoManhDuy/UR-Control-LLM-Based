import numpy as np
import scipy.signal as signal
import librosa
import soundfile as sf
import noisereduce
import logging
import time
import threading
from collections import deque
import pyaudio
import wave
import tempfile
import os

class IndustrialNoiseSuppressor:
    def __init__(self, 
                 sample_rate=16000, 
                 chunk_size=480,  # 30ms at 16kHz
                 calibration_time=5,  # seconds to capture noise profile
                 buffer_size=10,  # seconds of audio to keep in buffer
                 spectral_gate_threshold=0.5,  # Reduced from 1.5 to be less aggressive
                 pneumatic_freq_range=(2000, 4000),  # Adjusted to avoid speech frequencies
                 compressor_freq_range=(50, 500)):   # Adjusted to avoid speech frequencies
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.calibration_time = calibration_time
        self.noise_profile = None
        self.pneumatic_freq_range = pneumatic_freq_range
        self.compressor_freq_range = compressor_freq_range
        self.buffer_frames = int(buffer_size * (sample_rate / chunk_size))
        self.audio_buffer = deque(maxlen=self.buffer_frames)
        self.spectral_gate_threshold = spectral_gate_threshold
        
        # Adjusted bandpass filter for clearer speech (80-3000 Hz is the main speech range)
        self.speech_bandpass = self._create_speech_bandpass()
        self.pneumatic_notch = self._create_pneumatic_notch()
        self.compressor_notch = self._create_compressor_notch()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "temp_audio.wav")
        
    def _create_speech_bandpass(self):
        """Create bandpass filter for human speech range (80-3000 Hz)"""
        nyquist = self.sample_rate // 2
        low = 80 / nyquist    # Changed from 300 to 80 Hz
        high = 3000 / nyquist # Changed from 3400 to 3000 Hz
        return signal.butter(2, [low, high], btype='band')  # Reduced order from 4 to 2 for gentler filtering
    
    def _create_pneumatic_notch(self):
        """Create gentler notch filter for pneumatic machine noise"""
        nyquist = self.sample_rate // 2
        low = self.pneumatic_freq_range[0] / nyquist
        high = self.pneumatic_freq_range[1] / nyquist
        return signal.butter(4, [low, high], btype='bandstop')  # Reduced order from 6 to 4
    
    def _create_compressor_notch(self):
        """Create gentler notch filter for air compressor noise"""
        nyquist = self.sample_rate // 2
        low = self.compressor_freq_range[0] / nyquist
        high = self.compressor_freq_range[1] / nyquist
        return signal.butter(4, [low, high], btype='bandstop')  # Reduced order from 6 to 4
    
    def calibrate(self, audio_stream, duration=None):
        if duration is None:
            duration = self.calibration_time
        
        frames = []
        chunks_to_record = int((self.sample_rate / self.chunk_size) * duration)
        
        for _ in range(chunks_to_record):
            data = audio_stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
        
        self.noise_profile = np.concatenate(frames)
        return self.noise_profile
    
    def process_chunk(self, audio_chunk):
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        processed_data = self._apply_filters(audio_data)
        return processed_data.astype(np.int16).tobytes()
    
    def _apply_filters(self, audio_data):
        """Apply filters with smoother transitions"""
        self.audio_buffer.append(audio_data)
        
        # Apply speech bandpass first
        b, a = self.speech_bandpass
        speech_filtered = signal.filtfilt(b, a, audio_data)  # Changed from lfilter to filtfilt
        
        # Apply notch filters
        b, a = self.pneumatic_notch
        pneumatic_filtered = signal.filtfilt(b, a, speech_filtered)
        
        b, a = self.compressor_notch
        compressor_filtered = signal.filtfilt(b, a, pneumatic_filtered)
        
        if self.noise_profile is not None:
            # Use more conservative noise reduction settings
            noise_reduced = noisereduce.reduce_noise(
                y=compressor_filtered,
                y_noise=self.noise_profile,
                sr=self.sample_rate,
                stationary=True,  # Use stationary noise reduction for more stability
                prop_decrease=self.spectral_gate_threshold,
                n_fft=2048,
                win_length=1024,
                n_jobs=-1  # Use all CPU cores for faster processing
            )
            return noise_reduced
        
        return compressor_filtered
    
    def process_file(self, input_file, output_file=None):
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_filtered{ext}"
        
        audio_data, _ = librosa.load(input_file, sr=self.sample_rate, mono=True)
        
        if self.noise_profile is not None:
            processed_audio = noisereduce.reduce_noise(
                y=audio_data,
                y_noise=self.noise_profile,
                sr=self.sample_rate,
                stationary=False,
                prop_decrease=self.spectral_gate_threshold
            )
        else:
            b, a = self.speech_bandpass
            speech_filtered = signal.lfilter(b, a, audio_data)
            
            b, a = self.pneumatic_notch
            pneumatic_filtered = signal.lfilter(b, a, speech_filtered)
            
            b, a = self.compressor_notch
            processed_audio = signal.lfilter(b, a, pneumatic_filtered)
        
        sf.write(output_file, processed_audio, self.sample_rate)
        return output_file
    
    def close(self):
        try:
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except Exception as e:
            logging.error(f"Error cleaning up temp files: {e}") 