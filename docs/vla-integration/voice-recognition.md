# Voice Recognition

## Overview

Voice recognition is the first step in the Vision-Language-Action (VLA) pipeline, converting spoken language into text that can be processed by the language understanding system. This technology enables natural human-robot interaction through speech commands and conversations.

## Voice Recognition Fundamentals

### Automatic Speech Recognition (ASR)

Automatic Speech Recognition (ASR) is the technology that converts speech to text. Modern ASR systems use deep learning models trained on vast amounts of audio data to achieve high accuracy.

### Key Components

- **Audio Preprocessing**: Cleaning and preparing audio signals
- **Feature Extraction**: Converting audio to features suitable for neural networks
- **Acoustic Model**: Mapping audio features to phonemes
- **Language Model**: Converting phonemes to likely word sequences
- **Decoder**: Combining models to produce final text output

## Voice Recognition Technologies

### Cloud-Based Services

Cloud-based ASR services offer high accuracy and low development overhead:

- **Google Cloud Speech-to-Text**: Highly accurate with multiple language support
- **Amazon Transcribe**: AWS-based speech recognition service
- **Microsoft Azure Speech**: Comprehensive speech services
- **OpenAI Whisper**: Open-source, state-of-the-art model

### Local Solutions

Local ASR provides privacy and reduced latency:

- **Vosk**: Lightweight, open-source speech recognition
- **SpeechRecognition (Python library)**: Supports multiple engines
- **Coqui STT**: Open-source speech-to-text engine
- **Picovoice Porcupine**: Wake word detection and speech recognition

## Implementation with ROS 2

### Voice Recognition Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import speech_recognition as sr
import threading
import queue

class VoiceRecognitionNode(Node):
    def __init__(self):
        super().__init__('voice_recognition_node')

        # Publisher for recognized text
        self.text_pub = self.create_publisher(String, 'recognized_text', 10)

        # Subscriber for audio data
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000  # Adjust for environment
        self.recognizer.dynamic_energy_threshold = True

        # Microphone setup (if using direct microphone access)
        self.microphone = sr.Microphone()

        # Audio processing queue
        self.audio_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info('Voice Recognition Node initialized')

    def audio_callback(self, msg):
        """Process incoming audio data"""
        try:
            # Convert ROS audio message to AudioData format
            audio_data = sr.AudioData(
                msg.data,
                sample_rate=16000,  # Adjust based on your audio source
                sample_width=2      # 16-bit audio
            )

            # Add to processing queue
            self.audio_queue.put(audio_data)
        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

    def process_audio(self):
        """Process audio data in a separate thread"""
        while rclpy.ok():
            try:
                # Get audio from queue
                audio_data = self.audio_queue.get(timeout=1.0)

                # Recognize speech
                text = self.recognize_speech(audio_data)

                if text:
                    # Publish recognized text
                    text_msg = String()
                    text_msg.data = text
                    self.text_pub.publish(text_msg)
                    self.get_logger().info(f'Recognized: {text}')

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in audio processing: {e}')

    def recognize_speech(self, audio_data):
        """Recognize speech from audio data"""
        try:
            # Use Google Web Speech API (requires internet)
            # For offline use, consider Vosk or other offline engines
            text = self.recognizer.recognize_google(
                audio_data,
                language='en-US',
                show_all=False
            )
            return text
        except sr.UnknownValueError:
            self.get_logger().info('Speech recognition could not understand audio')
            return None
        except sr.RequestError as e:
            self.get_logger().error(f'Could not request results from speech service; {e}')
            return None

    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        with self.microphone as source:
            self.get_logger().info('Calibrating microphone...')
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            self.get_logger().info('Microphone calibrated')
```

## Using Vosk for Offline Recognition

Vosk provides excellent offline speech recognition capabilities:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import json
from vosk import Model, KaldiRecognizer

class VoskVoiceRecognitionNode(Node):
    def __init__(self):
        super().__init__('vosk_voice_recognition_node')

        # Publisher for recognized text
        self.text_pub = self.create_publisher(String, 'recognized_text', 10)

        # Subscriber for audio data
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        # Initialize Vosk model (download model first)
        try:
            self.model = Model(lang="en-us")  # Download English model
            self.rec = KaldiRecognizer(self.model, 16000)
            self.get_logger().info('Vosk model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load Vosk model: {e}')
            raise

        self.get_logger().info('Vosk Voice Recognition Node initialized')

    def audio_callback(self, msg):
        """Process incoming audio data with Vosk"""
        try:
            # Process audio chunk
            if self.rec.AcceptWaveform(msg.data):
                # Get final result
                result = self.rec.Result()
                result_dict = json.loads(result)

                if 'text' in result_dict and result_dict['text']:
                    text_msg = String()
                    text_msg.data = result_dict['text']
                    self.text_pub.publish(text_msg)
                    self.get_logger().info(f'Recognized: {result_dict["text"]}')
            else:
                # Get partial result (interim result)
                partial_result = self.rec.PartialResult()
                partial_dict = json.loads(partial_result)
                # Optionally handle partial results for real-time feedback
        except Exception as e:
            self.get_logger().error(f'Error in Vosk recognition: {e}')
```

## Wake Word Detection

Implement wake word detection to activate the system:

```python
from pvporcupine import Porcupine
import pyaudio
import struct

class WakeWordDetector:
    def __init__(self, keyword_paths, sensitivities):
        self.porcupine = Porcupine(
            keyword_paths=keyword_paths,
            sensitivities=sensitivities
        )

        self.audio = pyaudio.PyAudio()
        self.mic_stream = self.audio.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )

    def detect_wake_word(self):
        """Detect wake word from microphone input"""
        while True:
            pcm = self.mic_stream.read(self.porcupine.frame_length)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)

            keyword_index = self.porcupine.process(pcm)
            if keyword_index >= 0:
                return True  # Wake word detected
```

## Audio Preprocessing

### Noise Reduction

Implement noise reduction for better recognition:

```python
import numpy as np
from scipy import signal
import webrtcvad

class AudioPreprocessor:
    def __init__(self):
        # Initialize WebRTC VAD for voice activity detection
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # Aggressive mode

    def denoise_audio(self, audio_data, sample_rate=16000):
        """Apply noise reduction to audio data"""
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Apply noise reduction (simplified example)
        # In practice, use libraries like pyAudioAnalysis or scikit-dsp
        denoised = self.apply_spectral_gating(audio_array)

        return denoised.tobytes()

    def voice_activity_detection(self, audio_data, sample_rate=16000):
        """Detect voice activity in audio"""
        # WebRTC VAD expects 10, 20, or 30 ms frames
        frame_duration = 20  # ms
        frame_size = int(sample_rate * frame_duration / 1000)

        if len(audio_data) >= frame_size:
            frame = audio_data[:frame_size]
            return self.vad.is_speech(frame, sample_rate)

        return False

    def apply_spectral_gating(self, audio_array):
        """Apply basic spectral gating for noise reduction"""
        # This is a simplified example
        # In practice, use more sophisticated noise reduction algorithms
        return audio_array
```

## Integration with VLA Pipeline

### Voice Command Processing

```python
class VoiceCommandProcessor:
    def __init__(self):
        self.voice_recognizer = None  # Initialize with your chosen recognizer
        self.command_history = []
        self.is_listening = False

    def start_listening(self):
        """Start listening for voice commands"""
        self.is_listening = True
        self.command_history = []  # Clear previous commands
        self.get_logger().info('Voice command processor started')

    def stop_listening(self):
        """Stop listening for voice commands"""
        self.is_listening = False
        self.get_logger().info('Voice command processor stopped')

    def process_command(self, recognized_text):
        """Process recognized voice command"""
        if not self.is_listening:
            return None

        # Add to command history
        self.command_history.append({
            'text': recognized_text,
            'timestamp': self.get_clock().now().to_msg()
        })

        # Filter out wake words or activation phrases
        command = self.extract_command(recognized_text)

        if command:
            self.get_logger().info(f'Processing command: {command}')
            return self.parse_command(command)

        return None

    def extract_command(self, full_text):
        """Extract command from full text (remove wake words, etc.)"""
        # Common wake words/phrases to remove
        wake_words = ['robot', 'hey robot', 'hello robot', 'please']

        text = full_text.lower().strip()

        for wake_word in wake_words:
            if text.startswith(wake_word):
                text = text[len(wake_word):].strip()
                break

        return text if text else None

    def parse_command(self, command_text):
        """Parse command text into structured format"""
        # Simple command parsing (in practice, use NLP)
        command_parts = command_text.split()

        if not command_parts:
            return None

        # Basic command structure: [action] [object] [location/direction]
        parsed_command = {
            'action': command_parts[0] if command_parts else None,
            'object': command_parts[1] if len(command_parts) > 1 else None,
            'modifiers': command_parts[2:] if len(command_parts) > 2 else []
        }

        return parsed_command
```

## Performance Optimization

### Real-time Processing

Optimize for real-time performance:

```python
import asyncio
import concurrent.futures
from collections import deque

class RealTimeVoiceProcessor:
    def __init__(self):
        self.audio_buffer = deque(maxlen=10)  # Circular buffer
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.is_processing = False

    async def process_audio_stream(self, audio_stream):
        """Process continuous audio stream in real-time"""
        loop = asyncio.get_event_loop()

        async for audio_chunk in audio_stream:
            # Add to buffer
            self.audio_buffer.append(audio_chunk)

            # Process if not already processing
            if not self.is_processing:
                self.is_processing = True

                # Submit to thread pool for recognition
                future = self.executor.submit(
                    self.recognize_audio,
                    list(self.audio_buffer)
                )

                # Process result asynchronously
                result = await loop.run_in_executor(None, future.result)
                self.handle_recognition_result(result)

                self.is_processing = False

    def recognize_audio(self, audio_chunks):
        """Perform speech recognition on audio chunks"""
        # Combine audio chunks if needed
        combined_audio = b''.join(audio_chunks)

        # Perform recognition (implementation depends on chosen engine)
        recognized_text = self.perform_recognition(combined_audio)
        return recognized_text
```

## Troubleshooting Common Issues

### Audio Quality Problems

**Issue**: Poor recognition accuracy due to audio quality.

**Solutions**:
1. Check microphone placement and environment
2. Adjust recognition thresholds
3. Implement noise reduction preprocessing
4. Use directional microphones for better signal-to-noise ratio

**Issue**: High latency in recognition.

**Solutions**:
1. Use streaming recognition instead of batch processing
2. Optimize audio buffer sizes
3. Use lightweight recognition models for initial processing
4. Implement parallel processing where possible

### Environmental Challenges

**Issue**: Recognition fails in noisy environments.

**Solutions**:
1. Use beamforming microphones
2. Implement adaptive noise cancellation
3. Increase model sensitivity settings
4. Use context-specific language models

## Privacy and Security

### Data Handling

- **Local Processing**: Process sensitive conversations locally when possible
- **Encryption**: Encrypt audio data in transit and at rest
- **Retention Policies**: Implement data retention limits
- **User Consent**: Obtain explicit consent for voice data processing

## Best Practices

### System Design

- **Modular Architecture**: Keep recognition components separate for easy replacement
- **Fallback Mechanisms**: Implement fallback recognition methods
- **Continuous Learning**: Update models based on usage patterns
- **Performance Monitoring**: Track recognition accuracy and latency

### User Experience

- **Feedback**: Provide audio/visual feedback when listening
- **Confirmation**: Confirm understood commands before execution
- **Error Handling**: Gracefully handle unrecognized commands
- **Customization**: Allow users to customize wake words and commands

## Next Steps

Continue to [LLM Integration](./llm-integration.md) to learn how to connect large language models with your voice recognition system for advanced natural language understanding.