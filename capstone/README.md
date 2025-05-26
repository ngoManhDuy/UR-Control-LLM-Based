# Industrial Noise Suppression System

A specialized audio processing system designed to suppress noise in industrial environments, particularly optimized for robotic control applications. The system uses state-of-the-art deep learning models (Facebook Denoiser) combined with traditional signal processing techniques for real-time noise suppression.

## Key Features

1. **Advanced Noise Suppression**:
   - Facebook Denoiser model for deep learning-based noise removal
   - Real-time processing capability
   - Optimized for industrial environments
   - WebRTC VAD (Voice Activity Detection) integration

2. **Modern User Interface**:
   - PyQt6-based chat interface
   - Real-time audio visualization
   - Status updates and system feedback
   - Dark theme for better visibility

3. **Robust Audio Processing**:
   - Support for multiple audio formats
   - Real-time audio streaming
   - Configurable sample rates and chunk sizes
   - Multi-threaded processing for better performance

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for optimal performance)
- Working microphone and speakers
- Sufficient RAM (minimum 4GB recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ngoManhDuy/UR-Control-LLM-Based/tree/kido
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

### 1. Main Application

Run the enhanced voice control system:
```bash
python run_enhanced_voice.py
```

This will:
- Start the chat UI
- Initialize the noise suppression system
- Begin voice command processing

### 2. Testing Tools

#### Test Noise Suppression:
```bash
python test_noise_suppression.py --mode realtime
# or
python test_noise_suppression.py --mode file --input input.wav --output output.wav
```

#### Test Denoiser:
```bash
python test_denoiser.py
```

### 3. Standalone Noise Suppressor

Process audio files independently:
```bash
python noise_suppressor_tool.py filter input.wav --output filtered.wav
```

Record and filter in real-time:
```bash
python noise_suppressor_tool.py record --duration 30 --output recording.wav
```

## Architecture

The system consists of several key components:

1. **IndustrialNoiseSuppressor**: Core noise suppression engine using Facebook Denoiser
2. **EnhancedVoiceHandler**: Voice processing with integrated noise suppression
3. **ChatWindow**: Modern PyQt6-based user interface
4. **URController**: Robot control integration

## Performance Optimization

The system is optimized for:
- Real-time processing with minimal latency
- GPU acceleration when available
- Memory-efficient audio processing
- Threaded UI for responsive interaction

## Troubleshooting

Common issues and solutions:

1. **Audio Device Issues**:
   - Ensure your microphone is properly connected
   - Check system audio settings
   - Verify PyAudio installation

2. **GPU-related Issues**:
   - Update CUDA drivers if using GPU
   - Check torch installation matches CUDA version
   - Fall back to CPU if needed

3. **Performance Issues**:
   - Adjust chunk size for better latency
   - Monitor CPU/GPU usage
   - Close unnecessary applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook Denoiser team for their excellent model
- PyQt team for the UI framework
- The WebRTC team for VAD implementation 