# Industrial Noise Suppression System

A specialized audio processing system designed to suppress noise from pneumatic machines and air compressors in industrial environments. This system enhances voice recognition in noisy industrial settings.

## Components

1. **IndustrialNoiseSuppressor**: Core noise suppression engine specialized for industrial environments
2. **EnhancedVoiceHandler**: Extended voice handler with noise suppression capabilities
3. **Run Scripts**:
   - `run_enhanced_voice.py`: Main application with UI and robot control integration
   - `noise_suppressor_tool.py`: Standalone tool for noise profile creation and audio processing

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure you have a working microphone and speakers.

## Usage

### Main Application with Robot Control

To run the full system with the robotic arm control interface and noise suppression:

```bash
python run_enhanced_voice.py
```

This will:
1. Start the chat UI
2. Ask if you want to calibrate for industrial noise
3. If yes, record a noise profile for 5 seconds
4. Allow you to control the robot with voice commands in the noisy environment

### Standalone Noise Suppressor Tool

The `noise_suppressor_tool.py` utility can be used independently for:

#### 1. Create a Noise Profile

Record a sample of your industrial environment noise (without speech) to create a profile:

```bash
python noise_suppressor_tool.py profile --output my_factory_noise.wav --duration 10
```

#### 2. Filter an Existing Audio File

Apply noise suppression to an existing audio recording:

```bash
python noise_suppressor_tool.py filter my_recording.wav --noise-profile my_factory_noise.wav
```

#### 3. Record and Filter in Real-time

Record audio with noise suppression applied in real-time:

```bash
python noise_suppressor_tool.py record --duration 30 --output clean_speech.wav
```

## How It Works

The system uses multiple techniques to suppress industrial noise:

1. **Spectral Gating**: Learns the noise profile during calibration and suppresses those frequencies
2. **Bandpass Filtering**: Preserves the human voice frequency range (300Hz-3.4kHz)
3. **Notch Filtering**: Specifically targets frequency ranges common to:
   - Pneumatic machinery (1kHz-4kHz)
   - Air compressors (100Hz-800Hz)

## Optimization

The noise suppressor is optimized for:
- Real-time processing with minimal latency
- Preserving speech intelligibility 
- Reducing fatigue from continuous industrial noise

## Integration

To integrate with your own applications, you can:

1. Use `EnhancedVoiceHandler` as a drop-in replacement for `VoiceHandler`
2. Use `IndustrialNoiseSuppressor` directly for custom audio processing pipelines
3. Process files in batch using the standalone tool

## License

This software is provided for educational and research purposes. 