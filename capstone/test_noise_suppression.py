#!/usr/bin/env python

"""
Test script for noise suppression functionality.
This script only tests the audio processing without running the full system.
"""

import argparse
import sys
import time
import pyaudio
import wave
from audio_noise_suppression import IndustrialNoiseSuppressor

def test_realtime():
    """Test real-time noise suppression with microphone input"""
    # Initialize audio parameters
    sample_rate = 16000  # Standard sample rate for speech
    chunk_size = 480     # 30ms chunks at 16kHz
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    try:
        # Initialize noise suppressor
        suppressor = IndustrialNoiseSuppressor(
            sample_rate=sample_rate,
            chunk_size=chunk_size
        )
        
        # Open input stream
        input_stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        # Open output stream (to hear the filtered audio in real-time)
        output_stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            frames_per_buffer=chunk_size
        )
        
        print("\nStarting real-time noise suppression test...")
        print("First 10 chunks will be used to learn noise profile...")
        print("Please be quiet during this time.")
        time.sleep(2)  # Give user time to read the message
        
        chunks_processed = 0
        while True:
            # Read input
            input_data = input_stream.read(chunk_size, exception_on_overflow=False)
            
            # Process through noise suppressor
            filtered_data = suppressor.process_chunk(input_data)
            
            # Play filtered audio
            output_stream.write(filtered_data)
            
            # Show status during noise learning
            chunks_processed += 1
            if chunks_processed <= 10:
                sys.stdout.write(f"\rLearning noise profile: {chunks_processed}/10 chunks")
                sys.stdout.flush()
            elif chunks_processed == 11:
                print("\nNoise profile learned! You can speak now. Press Ctrl+C to stop.")
            
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        # Clean up
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        audio.terminate()
        suppressor.close()

def test_file(input_file, output_file):
    """Test noise suppression on an audio file"""
    try:
        # Initialize noise suppressor
        suppressor = IndustrialNoiseSuppressor()
        
        print(f"Processing file: {input_file}")
        # Process the file
        result_file = suppressor.process_file(input_file, output_file)
        print(f"Filtered audio saved to: {result_file}")
        
    finally:
        suppressor.close()

def main():
    parser = argparse.ArgumentParser(description="Test noise suppression functionality")
    parser.add_argument("--mode", choices=["realtime", "file"], default="realtime",
                      help="Test mode: 'realtime' for microphone input or 'file' for audio file processing")
    parser.add_argument("--input", help="Input audio file (required for file mode)")
    parser.add_argument("--output", help="Output audio file (for file mode)")
    
    args = parser.parse_args()
    
    if args.mode == "realtime":
        test_realtime()
    elif args.mode == "file":
        if not args.input:
            print("Error: Input file is required for file mode")
            sys.exit(1)
        test_file(args.input, args.output)

if __name__ == "__main__":
    main() 