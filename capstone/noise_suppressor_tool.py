#!/usr/bin/env python

"""
Industrial Noise Suppression Tool for Pneumatic Machines and Air Compressors

This utility can be used to:
1. Process existing audio files to remove industrial noise
2. Record and filter audio in real-time
3. Create and save noise profiles for reuse
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


def record_and_filter(duration=None, output_file="filtered_output.wav", calibration_time=5):
    """
    Record audio and apply industrial noise suppression in real-time
    
    Args:
        duration: Recording duration in seconds (None for continuous until Ctrl+C)
        output_file: Path to save filtered output
        calibration_time: Time to calibrate noise profile in seconds
    """
    # Initialize audio parameters
    sample_rate = 16000
    chunk_size = 480  # 30ms chunks at 16kHz
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    try:
        # Initialize noise suppressor
        suppressor = IndustrialNoiseSuppressor(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            calibration_time=calibration_time
        )
        
        # Open audio stream
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        print(f"\nPHASE 1: Calibrating for {calibration_time} seconds...")
        print("Please ensure ONLY industrial noise is present (no speech)...")
        
        # Countdown before calibration
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
            
        # Calibrate the noise profile
        suppressor.calibrate(stream, calibration_time)
        
        print("\nPHASE 2: Now recording speech with noise suppression...")
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


def filter_audio_file(input_file, output_file=None, noise_profile=None):
    """
    Apply industrial noise suppression to an existing audio file
    
    Args:
        input_file: Path to the input audio file
        output_file: Path to save the filtered audio (if None, auto-generated)
        noise_profile: Path to a saved noise profile file (if None, uses default filters)
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
        # Load noise profile if provided
        if noise_profile and os.path.exists(noise_profile):
            print(f"Loading noise profile from {noise_profile}...")
            import soundfile as sf
            import librosa
            
            # Load noise profile audio
            noise_data, sr = librosa.load(noise_profile, sr=suppressor.sample_rate, mono=True)
            suppressor.noise_profile = noise_data
            
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


def create_noise_profile(output_file, duration=10):
    """
    Create and save a noise profile for later use
    
    Args:
        output_file: Path to save the noise profile
        duration: Duration in seconds to record noise
    """
    # Initialize audio parameters
    sample_rate = 16000
    chunk_size = 480
    
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
        
        print(f"\nRecording industrial noise profile for {duration} seconds...")
        print("Please ensure ONLY the machinery noise sources are active (no speech)...")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
        
        # Record noise profile
        frames = []
        chunks_to_record = int((sample_rate / chunk_size) * duration)
        
        for i in range(chunks_to_record):
            # Show progress
            if i % (chunks_to_record // 10) == 0:
                progress = int((i / chunks_to_record) * 100)
                sys.stdout.write(f"\rProgress: {progress}%")
                sys.stdout.flush()
            
            # Read audio chunk
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
        
        print("\nNoise profile recording complete!")
        
        # Close stream
        stream.stop_stream()
        stream.close()
        
        # Save the noise profile
        print(f"Saving noise profile to {output_file}...")
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
            
        print(f"Successfully saved noise profile to {output_file}")
        print("This file can be used with the --noise-profile option when filtering audio files.")
        
        return True
        
    except Exception as e:
        print(f"Error creating noise profile: {str(e)}")
        logging.error(f"Error creating noise profile: {str(e)}")
        return False
        
    finally:
        audio.terminate()


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
    record_parser.add_argument("--calibration-time", type=int, default=5,
                              help="Time for noise calibration in seconds (default: 5)")
    
    # Subparser for filtering existing files
    filter_parser = subparsers.add_parser("filter", help="Filter an existing audio file")
    filter_parser.add_argument("input", type=str, help="Input audio file path")
    filter_parser.add_argument("--output", type=str, default=None,
                              help="Output file path (default: input_filtered.wav)")
    filter_parser.add_argument("--noise-profile", type=str, default=None,
                              help="Path to a saved noise profile (optional)")
    
    # Subparser for creating noise profiles
    profile_parser = subparsers.add_parser("profile", help="Create a noise profile for later use")
    profile_parser.add_argument("--output", type=str, default="noise_profile.wav",
                               help="Output noise profile path (default: noise_profile.wav)")
    profile_parser.add_argument("--duration", type=int, default=10,
                               help="Duration to record noise in seconds (default: 10)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Execute command
    if args.command == "record":
        record_and_filter(args.duration, args.output, args.calibration_time)
    elif args.command == "filter":
        filter_audio_file(args.input, args.output, args.noise_profile)
    elif args.command == "profile":
        create_noise_profile(args.output, args.duration)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 