#!/usr/bin/env python

"""
Industrial Noise Suppression Tool for Pneumatic Machines and Air Compressors

This utility can be used to:
1. Process existing audio files to remove industrial noise
2. Record and filter audio in real-time
3. Process live audio input
"""

import argparse
import os
import sys
import time
import pyaudio
import wave
import numpy as np
from pathlib import Path
import tempfile
import logging

# Import our noise suppression module
from audio_noise_suppression import IndustrialNoiseSuppressor


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(message)s",
        level=logging.INFO
    )


def record_and_filter(duration=None, output_file="filtered_output.wav"):
    """
    Record audio and apply industrial noise suppression in real-time
    
    Args:
        duration: Recording duration in seconds (None for continuous until Ctrl+C)
        output_file: Path to save filtered output
    """
    # Initialize audio parameters
    sample_rate = 48000  # RNNoise works best with 48kHz
    chunk_size = 480    # 10ms chunks at 48kHz
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    try:
        # Initialize noise suppressor
        suppressor = IndustrialNoiseSuppressor(
            sample_rate=sample_rate,
            chunk_size=chunk_size
        )
        
        # Open audio stream
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        print("\nStarting real-time noise suppression...")
        print("Recording continuously... Press Ctrl+C to stop")
        
        # Record and filter audio
        frames = []
        start_time = time.time()
        recorded_chunks = 0
        
        try:
            while True:
                # Read and process audio chunk
                data = stream.read(chunk_size, exception_on_overflow=False)
                filtered_data = suppressor.process_chunk(data)
                frames.append(filtered_data)
                recorded_chunks += 1
                
                # Show progress every second
                if recorded_chunks % (sample_rate // chunk_size) == 0:
                    elapsed = time.time() - start_time
                    sys.stdout.write(f"\rRecording time: {elapsed:.1f} seconds")
                    sys.stdout.flush()
                    
                # If duration is specified, check if we should stop
                if duration and time.time() - start_time >= duration:
                    break
                    
        except KeyboardInterrupt:
            print("\nRecording stopped by user")
            
        actual_duration = time.time() - start_time
        print(f"\nFinished recording! Actual duration: {actual_duration:.1f} seconds")
        
        # Close resources
        stream.stop_stream()
        stream.close()
        
        # Save the output file
        print(f"Saving filtered audio to {output_file}...")
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
            
        print(f"Successfully saved to {output_file}")
        
    finally:
        # Clean up resources
        audio.terminate()
        if 'suppressor' in locals():
            suppressor.close()


def filter_audio_file(input_file, output_file=None):
    """
    Apply industrial noise suppression to an existing audio file
    
    Args:
        input_file: Path to the input audio file
        output_file: Path to save the filtered audio (if None, auto-generated)
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return False
        
    # Generate output filename if not provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_filtered{ext}"
    
    # Initialize noise suppressor
    suppressor = IndustrialNoiseSuppressor()
    
    try:
        # Process the file
        print(f"Processing {input_file}...")
        result_file = suppressor.process_file(input_file, output_file)
        
        print(f"Successfully filtered audio to {result_file}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        logging.error(f"Error processing file: {str(e)}")
        return False
    finally:
        suppressor.close()


def main():
    """Main entry point for the command line tool"""
    parser = argparse.ArgumentParser(
        description="Industrial Noise Suppression Tool for Pneumatic Machines and Air Compressors"
    )
    
    # Create subparsers for different operations
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Subparser for recording and filtering
    record_parser = subparsers.add_parser("record", help="Record and filter audio in real-time")
    record_parser.add_argument("--duration", type=int, default=None,
                              help="Recording duration in seconds (default: continuous until Ctrl+C)")
    record_parser.add_argument("--output", type=str, default="filtered_output.wav",
                              help="Output file path (default: filtered_output.wav)")
    
    # Subparser for filtering existing files
    filter_parser = subparsers.add_parser("filter", help="Filter an existing audio file")
    filter_parser.add_argument("input", type=str, help="Input audio file path")
    filter_parser.add_argument("--output", type=str, default=None,
                              help="Output file path (default: input_filtered.wav)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Execute command
    if args.command == "record":
        record_and_filter(args.duration, args.output)
    elif args.command == "filter":
        filter_audio_file(args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 