import numpy as np
import librosa
import scipy.io.wavfile as wav
import scipy.signal
from scipy.signal import butter, filtfilt, wiener
import argparse
import os

def load_audio(file_path):
    """Load audio file"""
    try:
        # Load with librosa for better compatibility
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading with librosa: {e}")
        # Fallback to scipy
        try:
            sr, y = wav.read(file_path)
            # Normalize if integer format
            if y.dtype == np.int16:
                y = y / 32768.0
            elif y.dtype == np.int32:
                y = y / 2147483648.0
            return y, sr
        except Exception as e2:
            print(f"Error loading with scipy: {e2}")
            raise

def save_audio(y, sr, output_path):
    """Save audio file"""
    # Convert back to int16 for WAV format
    y_int = (y * 32767).astype(np.int16)
    wav.write(output_path, sr, y_int)

def spectral_subtraction(y, sr, noise_factor=2.0, alpha=2.0):
    """
    Simple spectral subtraction for noise reduction
    """
    # Compute STFT
    stft = librosa.stft(y)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise from the first 0.5 seconds (assuming it's mostly noise)
    noise_frames = int(0.5 * sr / 512)  # 512 is default hop_length
    noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Apply spectral subtraction
    clean_magnitude = magnitude - alpha * noise_spectrum
    
    # Ensure we don't go below a fraction of original
    clean_magnitude = np.maximum(clean_magnitude, magnitude / noise_factor)
    
    # Reconstruct signal
    clean_stft = clean_magnitude * np.exp(1j * phase)
    clean_audio = librosa.istft(clean_stft)
    
    return clean_audio

def wiener_filter_denoise(y, noise_est_length=0.5):
    """
    Apply Wiener filtering for noise reduction
    """
    # This is a simplified version - estimate noise from beginning
    sr_est = int(len(y) * noise_est_length / 5.0)  # Assuming 5 second audio
    noise_sample = y[:sr_est] if len(y) > sr_est else y[:len(y)//4]
    
    # Estimate noise variance
    noise_var = np.var(noise_sample)
    
    # Apply Wiener filter (simplified)
    signal_var = np.var(y)
    wiener_gain = signal_var / (signal_var + noise_var)
    
    return y * wiener_gain

def bandpass_filter(y, sr, low_freq=80, high_freq=8000):
    """
    Apply bandpass filter to remove frequencies outside speech range
    """
    nyquist = sr / 2
    
    # Ensure frequencies are within valid range
    low_freq = max(low_freq, 1)  # Minimum 1 Hz
    high_freq = min(high_freq, nyquist * 0.95)  # Maximum 95% of Nyquist
    
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure normalized frequencies are in valid range (0, 1)
    low = max(low, 0.01)  # At least 1% of Nyquist
    high = min(high, 0.95)  # At most 95% of Nyquist
    
    # Skip filtering if range is invalid
    if low >= high:
        print(f"Warning: Invalid filter range, skipping bandpass filter")
        return y
    
    try:
        # Design Butterworth bandpass filter
        b, a = butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered = filtfilt(b, a, y)
        return filtered
    except Exception as e:
        print(f"Warning: Bandpass filter failed ({e}), returning original audio")
        return y

def noise_gate(y, threshold=0.01, attack_time=0.01, release_time=0.1, sr=22050):
    """
    Apply noise gate to reduce quiet background noise
    """
    # Convert times to samples
    attack_samples = int(attack_time * sr)
    release_samples = int(release_time * sr)
    
    # Calculate envelope (RMS with sliding window)
    window_size = int(0.02 * sr)  # 20ms window
    envelope = np.sqrt(np.convolve(y**2, np.ones(window_size)/window_size, mode='same'))
    
    # Create gate signal
    gate = np.ones_like(y)
    below_threshold = envelope < threshold
    
    # Apply attack/release smoothing
    for i in range(1, len(gate)):
        if below_threshold[i]:
            if gate[i-1] > 0:
                # Start of gate closing
                gate[i] = max(0, gate[i-1] - 1/attack_samples)
            else:
                gate[i] = 0
        else:
            if gate[i-1] < 1:
                # Start of gate opening
                gate[i] = min(1, gate[i-1] + 1/release_samples)
            else:
                gate[i] = 1
    
    return y * gate

def adaptive_filter_denoise(y, sr):
    """
    Simple adaptive filtering approach
    """
    # Estimate noise from quiet sections
    # Find RMS for overlapping windows
    window_size = int(0.1 * sr)  # 100ms windows
    hop_size = window_size // 2
    
    rms_values = []
    for i in range(0, len(y) - window_size, hop_size):
        window = y[i:i + window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    
    # Assume noise threshold is 20th percentile of RMS values
    noise_threshold = np.percentile(rms_values, 20)
    
    # Apply noise gate based on this threshold
    return noise_gate(y, threshold=noise_threshold * 2, sr=sr)

def comprehensive_denoise(y, sr, method='all'):
    """
    Apply comprehensive denoising pipeline
    """
    print(f"Original audio: {len(y)} samples, {len(y)/sr:.2f} seconds")
    print(f"Sample rate: {sr} Hz, Nyquist frequency: {sr/2} Hz")
    
    if method in ['all', 'bandpass']:
        print("Applying bandpass filter...")
        y = bandpass_filter(y, sr)
    
    if method in ['all', 'spectral']:
        print("Applying spectral subtraction...")
        y = spectral_subtraction(y, sr)
    
    if method in ['all', 'gate']:
        print("Applying noise gate...")
        y = adaptive_filter_denoise(y, sr)
    
    if method in ['all', 'wiener']:
        print("Applying Wiener filter...")
        y = wiener_filter_denoise(y)
    
    # Normalize
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val * 0.8
    
    return y

def analyze_noise_profile(y, sr):
    """
    Analyze the noise characteristics of the audio
    """
    print("\n=== Noise Analysis ===")
    
    # Overall statistics
    print(f"RMS level: {np.sqrt(np.mean(y**2)):.4f}")
    print(f"Peak level: {np.max(np.abs(y)):.4f}")
    print(f"Dynamic range: {20 * np.log10(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-10)):.1f} dB")
    
    # Noise floor estimation (lowest 10% of RMS values)
    window_size = int(0.1 * sr)
    hop_size = window_size // 2
    rms_values = []
    
    for i in range(0, len(y) - window_size, hop_size):
        window = y[i:i + window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    
    noise_floor = np.percentile(rms_values, 10)
    signal_level = np.percentile(rms_values, 90)
    
    print(f"Estimated noise floor: {noise_floor:.4f}")
    print(f"Estimated signal level: {signal_level:.4f}")
    print(f"Signal-to-noise ratio: {20 * np.log10(signal_level / (noise_floor + 1e-10)):.1f} dB")
    print("=" * 30)

def main():
    parser = argparse.ArgumentParser(description='Denoise audio files')
    parser.add_argument('input', nargs='?', default='output.wav',
                       help='Input WAV file (default: output.wav)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output WAV file (default: input_denoised.wav)')
    parser.add_argument('--method', '-m', choices=['all', 'bandpass', 'spectral', 'gate', 'wiener'],
                       default='all', help='Denoising method to apply')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Analyze noise profile of input file')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Set output filename
    if args.output is None:
        name, ext = os.path.splitext(args.input)
        args.output = f"{name}_denoised{ext}"
    
    try:
        # Load audio
        print(f"Loading audio file: {args.input}")
        y, sr = load_audio(args.input)
        
        # Analyze if requested
        if args.analyze:
            analyze_noise_profile(y, sr)
        
        # Apply denoising
        print(f"Applying denoising method: {args.method}")
        y_clean = comprehensive_denoise(y, sr, method=args.method)
        
        # Save result
        print(f"Saving denoised audio to: {args.output}")
        save_audio(y_clean, sr, args.output)
        
        print("Denoising complete!")
        print(f"You can now compare:")
        print(f"  Original: {args.input}")
        print(f"  Denoised: {args.output}")
        
    except Exception as e:
        print(f"Error processing audio: {e}")

if __name__ == "__main__":
    main()
