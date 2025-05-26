import ctypes
import numpy as np
from ctypes import c_int, c_float, c_void_p, POINTER, Structure
import os

class RNNoiseState(Structure):
    pass

class RNNoiseWrapper:
    def __init__(self):
        # Load the RNNoise library
        try:
            if os.name == 'posix':  # Linux/Mac
                self.lib = ctypes.CDLL('librnnoise.so.0')
            else:  # Windows
                self.lib = ctypes.CDLL('rnnoise.dll')
        except OSError as e:
            raise RuntimeError("Could not load RNNoise library. Make sure it's installed.") from e

        # Define function signatures
        self.lib.rnnoise_create.argtypes = []
        self.lib.rnnoise_create.restype = POINTER(RNNoiseState)
        
        self.lib.rnnoise_process_frame.argtypes = [POINTER(RNNoiseState), 
                                                  POINTER(c_float), 
                                                  POINTER(c_float)]
        self.lib.rnnoise_process_frame.restype = c_float
        
        self.lib.rnnoise_destroy.argtypes = [POINTER(RNNoiseState)]
        self.lib.rnnoise_destroy.restype = None

        # Create RNNoise state
        self.state = self.lib.rnnoise_create()
        if not self.state:
            raise RuntimeError("Could not create RNNoise state")

        # RNNoise constants
        self.frame_size = 480  # 10ms at 48kHz

    def process_frame(self, input_frame):
        """Process a single frame of audio through RNNoise.
        
        Args:
            input_frame: numpy array of float32 values in range [-1, 1]
                        Must be exactly 480 samples (10ms at 48kHz)
        
        Returns:
            Processed frame as numpy array
        """
        if len(input_frame) != self.frame_size:
            raise ValueError(f"Input frame must be exactly {self.frame_size} samples")

        # Prepare input buffer
        input_buffer = input_frame.astype(np.float32)
        input_ptr = input_buffer.ctypes.data_as(POINTER(c_float))
        
        # Prepare output buffer
        output_buffer = np.zeros(self.frame_size, dtype=np.float32)
        output_ptr = output_buffer.ctypes.data_as(POINTER(c_float))
        
        # Process frame
        vad_prob = self.lib.rnnoise_process_frame(self.state, output_ptr, input_ptr)
        
        return output_buffer, vad_prob

    def __del__(self):
        """Clean up RNNoise state when object is destroyed"""
        if hasattr(self, 'state') and self.state:
            self.lib.rnnoise_destroy(self.state)
            self.state = None 