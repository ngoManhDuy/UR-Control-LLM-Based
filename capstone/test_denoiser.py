import torch
import torchaudio
from denoiser import pretrained
import os

def denoise_audio(input_file, output_file):
    # Load model
    model = pretrained.dns64()
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Using device: {device}")
    
    # Load audio file
    print(f"Loading audio file: {input_file}")
    waveform, sample_rate = torchaudio.load(input_file)
    print(f"Audio loaded - Sample rate: {sample_rate}, Shape: {waveform.shape}")
    
    # Resample if needed (model expects 16kHz)
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate}Hz to 16000Hz")
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Ensure input is mono
    if waveform.size(0) > 1:
        print("Converting to mono")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Move to device
    waveform = waveform.to(device)
    
    print("Processing audio...")
    
    # Process through denoiser
    with torch.no_grad():
        denoised_waveform = model(waveform)
    
    # Move back to CPU and ensure correct format (2D tensor)
    denoised_waveform = denoised_waveform.cpu()
    if denoised_waveform.dim() == 1:
        denoised_waveform = denoised_waveform.unsqueeze(0)
    elif denoised_waveform.dim() == 3:
        denoised_waveform = denoised_waveform.squeeze(0)
    
    print(f"Denoised waveform shape: {denoised_waveform.shape}")
    print(f"Denoised waveform min: {denoised_waveform.min()}, max: {denoised_waveform.max()}")
    
    # Normalize if needed
    if denoised_waveform.abs().max() > 1:
        denoised_waveform = denoised_waveform / denoised_waveform.abs().max()
    
    # Save the denoised audio
    torchaudio.save(output_file, denoised_waveform, sample_rate)
    print(f"Denoised audio saved to: {output_file}")

if __name__ == "__main__":
    input_file = "test.wav"
    output_file = "denoised_output.wav"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        exit(1)
    
    denoise_audio(input_file, output_file) 