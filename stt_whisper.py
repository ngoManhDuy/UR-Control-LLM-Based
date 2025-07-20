import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pyaudio
import numpy as np
import soundfile as sf
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor
import os
import librosa
from scipy.signal import butter, filtfilt
import webrtcvad
import collections
import time

# Import denoising functions
from denoise_audio import (
    comprehensive_denoise
)


class STT_module:
    """Speech-to-Text class using Whisper model with Voice Activity Detection - GUI Compatible"""
    
    def __init__(self, model_name="openai/whisper-medium", enable_denoising=True, status_callback=None):
        # Audio parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1        # Mono
        self.RATE = 16000        # 16kHz for VAD compatibility
        self.CHUNK = 320         # 20ms frames for VAD (16000 * 0.02)
        self.OUTPUT_FILENAME = "output.wav"
        self.DENOISED_FILENAME = "output_denoised.wav"
        
        # VAD parameters
        self.vad_aggressiveness = 2  # 0-3, higher = more aggressive
        self.min_speech_duration = 0.5  # Minimum seconds of speech to process
        self.max_silence_duration = 2.0  # Max seconds of silence before stopping
        self.energy_threshold = 500  # Energy threshold for basic voice detection
        
        # Denoising settings
        self.enable_denoising = enable_denoising
        
        # Callback for status updates
        self.status_callback = status_callback
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        
        # Load Whisper model components
        self._update_status(f"Loading {model_name} model...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Initialize ASR pipeline
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            stride_length_s=0
        )
        self._update_status("Model loaded successfully!")

    def _update_status(self, message):
        """Update status through callback if available"""
        if self.status_callback:
            self.status_callback(message)

    def calculate_energy(self, audio_frame):
        """Calculate energy of audio frame"""
        audio_data = np.frombuffer(audio_frame, dtype=np.int16)
        return np.sqrt(np.mean(audio_data**2))

    def is_speech(self, audio_frame):
        """
        Determine if audio frame contains speech using both VAD and energy
        """
        # Energy-based detection
        energy = self.calculate_energy(audio_frame)
        energy_speech = energy > self.energy_threshold
        
        # WebRTC VAD detection (requires specific frame size)
        try:
            vad_speech = self.vad.is_speech(audio_frame, self.RATE)
        except:
            vad_speech = False
        
        # Combine both methods (OR logic)
        return energy_speech or vad_speech

    def record_with_vad(self, max_duration=30):
        """
        Record audio with Voice Activity Detection
        Returns the recorded audio data or None if no speech detected
        """
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            self._update_status("🎤 Listening for speech...")
            
            # States
            is_recording = False
            speech_frames = []
            silence_counter = 0
            speech_counter = 0
            max_frames = int(self.RATE / self.CHUNK * max_duration)
            
            # Buffers
            silence_threshold_frames = int(self.max_silence_duration * self.RATE / self.CHUNK)
            speech_threshold_frames = int(self.min_speech_duration * self.RATE / self.CHUNK)
            
            for frame_count in range(max_frames):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                
                # Check if current frame contains speech
                has_speech = self.is_speech(data)
                
                if has_speech:
                    if not is_recording:
                        self._update_status("🗣️ Speech detected, recording...")
                        is_recording = True
                    
                    speech_frames.append(data)
                    speech_counter += 1
                    silence_counter = 0
                    
                else:  # No speech
                    if is_recording:
                        silence_counter += 1
                        speech_frames.append(data)  # Keep recording during brief silences
                        
                        # Check if silence duration exceeded threshold
                        if silence_counter >= silence_threshold_frames:
                            self._update_status("⏹️ Speech ended, processing...")
                            break
                    else:
                        # Reset counters if we're not recording
                        speech_counter = 0
                        silence_counter = 0
            
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
        
        # Check if we have enough speech
        if speech_counter < speech_threshold_frames:
            self._update_status("Not enough speech detected")
            return None
        
        # Convert frames to audio data
        if speech_frames:
            audio_data = np.frombuffer(b"".join(speech_frames), dtype=np.int16)
            return audio_data
        
        return None

    def save_audio(self, audio_data, filename=None):
        """Save audio numpy array to file"""
        filename = filename or self.OUTPUT_FILENAME
        sf.write(filename, audio_data, self.RATE, subtype='PCM_16')

    def denoise_audio(self, audio_data):
        """
        Apply denoising to audio data using comprehensive denoising pipeline
        """
        if not self.enable_denoising:
            return audio_data
            
        self._update_status("🔧 Applying audio denoising...")
        
        try:
            # Convert int16 to float for processing
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Apply comprehensive denoising
            denoised_audio = comprehensive_denoise(audio_float, self.RATE, method='all')
            
            # Convert back to int16
            denoised_audio_int16 = (denoised_audio * 32767).astype(np.int16)
            
            self._update_status("✅ Denoising completed")
            return denoised_audio_int16
            
        except Exception as e:
            self._update_status(f"Denoising error: {e}, using original audio")
            return audio_data

    def transcribe_audio(self, audio_path=None):
        """Transcribe audio file to text"""
        audio_path = audio_path or self.OUTPUT_FILENAME
        
        self._update_status("🔄 Transcribing audio...")
        output = self.transcriber(audio_path, generate_kwargs={"language": "en"})
        return output["text"]

    def process_single_recording(self, max_duration=30):
        """
        Process a single recording session and return results
        Returns the transcribed text or None if no speech
        """ 
        # Record with VAD
        audio_data = self.record_with_vad(max_duration=max_duration)
        
        if audio_data is None:
            return None
        
        # Save original audio before denoising
        original_filename = "output_original.wav"
        self.save_audio(audio_data, original_filename)
        
        # Apply denoising if enabled
        denoised_audio = self.denoise_audio(audio_data)
        
        # Save processed audio for transcription
        final_filename = self.DENOISED_FILENAME if self.enable_denoising else self.OUTPUT_FILENAME
        self.save_audio(denoised_audio, final_filename)
        
        # Transcribe using the processed audio
        transcribed_text = self.transcribe_audio(final_filename)
        
        return transcribed_text


def main():
    """Main function for testing"""
    def status_update(message):
        print(f"Status: {message}")
    
    print("STT Module with Integrated Denoising Test")
    print("=" * 50)
    
    # Test with denoising enabled
    print("Testing with denoising enabled...")
    stt = STT_module(
        model_name="openai/whisper-medium", 
        enable_denoising=True,
        status_callback=status_update
    )
    
    # Test single recording
    transcribed_text = stt.process_single_recording()
    
    if transcribed_text:
        print(f"\n📝 Transcription: {transcribed_text}")
        print(f"\n📁 Denoised audio saved as: output_denoised.wav")
    else:
        print("❌ No speech detected")


if __name__ == "__main__":
    main()
