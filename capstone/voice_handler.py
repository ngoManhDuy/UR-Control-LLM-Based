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


class VoiceHandler:
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # TTS voice settings
        self.voice = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
        self.tts_model = "tts-1"
        self.stt_model = "whisper-1"
        
        # VAD parameters
        self.vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (highest)
        self.sample_rate = 16000
        self.chunk_duration_ms = 30  # ms
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        # Audio recording parameters
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Create output directory for speech files at initialization
        self.output_dir = Path("speech_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Reusable format parameters for wave files
        self.wave_params = {
            "nchannels": 1,
            "sampwidth": 2,  # 16-bit audio
            "framerate": self.sample_rate
        }
        
        # Timeout configuration for speech detection
        self.speech_timeout_seconds = 3  # 3 seconds timeout for speech detection
        
        # Performance optimization
        self._setup_temp_file()  # Pre-create temp file path
        
        logging.info("Voice handler initialized")

    def _setup_temp_file(self):
        """Pre-create temporary file path for reuse"""
        self.temp_wav_path = os.path.join(tempfile.gettempdir(), "temp_speech.wav")

    def start_audio_stream(self):
        """Start the audio stream for VAD."""
        # Don't recreate stream if already exists
        if self.stream is None:
            try:
                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                self.recording = True
                self.stop_event.clear()
            except Exception as e:
                logging.error(f"Failed to open audio stream: {e}")
                print(f"Error opening audio stream: {e}")
                self.recording = False

    def stop_audio_stream(self):
        """Stop the audio stream."""
        self.recording = False
        self.stop_event.set()
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                logging.error(f"Error stopping audio stream: {e}")

    def is_speech(self, audio_chunk):
        """Check if the audio chunk contains speech."""
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except:
            return False

    def record_audio(self):
        """Record audio while speech is detected."""
        # Pre-allocate buffer for better performance - made smaller for less memory usage
        buffer_size = 20 * self.chunk_size  # Initial smaller buffer 
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
                audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                chunk_len = len(audio_chunk)
                
                if self.is_speech(audio_chunk):
                    # Reset timeout counter when speech is detected
                    speech_detected = True
                    
                    # Extend buffer if needed - more efficient growth
                    if total_bytes + chunk_len > len(frames):
                        frames.extend(bytearray(chunk_len * 5))  # Grow by 5x chunk size instead of 10x
                    
                    # Add chunk to buffer
                    frames[total_bytes:total_bytes+chunk_len] = audio_chunk
                    total_bytes += chunk_len
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames > max_silence_frames and speech_detected:
                        # Only break if we've already detected some speech followed by silence
                        break
                
                # Check for timeout if no speech has been detected yet - less frequent checks
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
        """Listen for voice command using VAD and return the recognized text using OpenAI."""
        try:
            print("Listening... (speak when ready)")
            self.start_audio_stream()
            
            # Record audio while speech is detected
            audio_data = self.record_audio()
            self.stop_audio_stream()
            
            if not audio_data or len(audio_data) < 100:  # Add minimum size check
                print("No speech detected or too short")
                return None
            
            # Convert audio data to WAV format
            try:
                with wave.open(self.temp_wav_path, "wb") as wf:
                    wf.setnchannels(self.wave_params["nchannels"])
                    wf.setsampwidth(self.wave_params["sampwidth"])
                    wf.setframerate(self.wave_params["framerate"])
                    wf.writeframes(audio_data)
                
                # Use OpenAI's Whisper API for speech recognition
                with open(self.temp_wav_path, "rb") as audio_file:
                    transcription = self.client.audio.transcriptions.create(
                        model=self.stt_model,
                        file=audio_file
                    )
                    text = transcription.text
                    print(f"You said: {text}")
                    return text.lower()
            except Exception as e:
                print(f"OpenAI STT Error: {e}")
                logging.error(f"OpenAI STT Error: {e}")
                return None
                    
        except Exception as e:
            print(f"Error listening for command: {e}")
            logging.error(f"Error listening for command: {e}")
            return None

    def speak(self, text):
        """Convert text to speech using OpenAI's TTS API and play it with sounddevice."""
        try:
            print(f"Robot: {text}")
            
            # Create filename based on timestamp
            timestamp = int(time.time())
            output_path = self.output_dir / f"speech_{timestamp}.wav"
            
            # Generate speech using OpenAI TTS
            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice=self.voice,
                input=text
            )
            
            # Save to WAV file
            response.stream_to_file(str(output_path))
            
            # Play the audio file using sounddevice and soundfile
            data, samplerate = sf.read(str(output_path))
            sd.play(data, samplerate)
            sd.wait()  # Wait until playback is finished
            
        except Exception as e:
            print(f"Error in OpenAI TTS: {e}")
            logging.error(f"Error in OpenAI TTS: {e}")

    def close(self):
        """Cleanup resources."""
        try:
            self.stop_audio_stream()
            self.audio.terminate()
            
            # Clean up temp file if it exists
            try:
                if os.path.exists(self.temp_wav_path):
                    os.unlink(self.temp_wav_path)
            except:
                pass
                
            logging.info("Voice handler closed")
        except Exception as e:
            logging.error(f"Error closing voice handler: {e}") 