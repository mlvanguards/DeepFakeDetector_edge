# DeepfakeGuard - Real-time Deepfake Audio Detection

An Android application that detects deepfake audio in real-time during phone calls using on-device machine learning.

## Features

- **Real-time Detection**: Analyzes audio during phone calls to detect synthetic/deepfake audio
- **Fully Local**: All processing happens on-device using PyTorch Mobile for maximum privacy
- **Visual Overlay**: Shows detection results in real-time during calls
- **Call Monitoring**: Automatically starts monitoring when phone calls begin/end
- **Audio Analysis**: Uses advanced audio features (MFCC, spectral features) for detection

## Setup Instructions

### 1. Convert Your CRNN Model

Convert your trained CRNN `.pth` model file to `.pt` format for mobile deployment:

```bash
# Basic conversion
python convert_model.py --input your_model.pth --output deepfake_detector.pt --analyze

# If your model has different hyperparameters
python convert_model.py --input your_model.pth --hidden_size 128 --num_layers 1 --dropout 0.2
```

### 2. Test Your Conversion (Recommended)

Verify the conversion works correctly:

```bash
# Test with synthetic audio
python convert_and_test.py --model_path your_model.pth --visualize

# Test with real audio file
python convert_and_test.py --model_path your_model.pth --audio_file test_audio.wav
```

### 3. Add Model to App

1. Copy the generated `deepfake_detector.pt` file to `app/src/main/assets/models/`
2. The model expects mel spectrogram input: `[batch, 2_channels, 64_mel_bins, 300_time_steps]`

### 4. Build and Install

```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 5. Grant Permissions

The app requires several permissions for call monitoring:
- Phone state access
- Audio recording
- Overlay permissions (for showing detection results)
- Foreground service permissions

## Architecture

### Core Components

- **PhoneStateReceiver**: Detects incoming/outgoing calls
- **DeepfakeDetectionService**: Main service that runs detection during calls
- **AudioProcessor**: Generates mel spectrograms from stereo audio
- **OverlayView**: Shows real-time detection results during calls
- **MainActivity**: Main UI for app configuration and status

### Audio Processing Pipeline

1. **Audio Capture**: Records stereo call audio using `AudioRecord` with `VOICE_CALL` source
2. **Mel Spectrogram Generation**: Converts 4-second stereo audio chunks to mel spectrograms:
   - Input: Stereo audio (2 channels) at 16kHz
   - STFT with n_fft=780, hop_length=195
   - Mel filter bank with 64 mel bins
   - Convert to dB scale (top_db=80)
   - Output shape: [2, 64, 300] (channels, mel_bins, time_steps)
3. **Model Inference**: Runs mel spectrograms through CRNN PyTorch Mobile model
4. **Result Display**: Shows probability scores and detection status in overlay

### CRNN Model Requirements

Your CRNN model should:
- **Architecture**: ResNet18 backbone + BiGRU + Attention pooling
- **Input shape**: `[batch_size, 2, 64, 300]` (stereo mel spectrograms)
- **Output**: Binary classification `[real_probability, fake_probability]`
- **Audio specs**: 16kHz stereo, 4-second duration
- **Compatible with**: PyTorch 2.1.0, CPU inference

## Usage

1. **Start Monitoring**: Open the app and tap "Start Monitoring"
2. **Automatic Detection**: The service will automatically activate during phone calls
3. **View Results**: Real-time detection results appear as an overlay during calls
4. **Check History**: View detection logs and statistics in the app

## Model Training

The conversion scripts support the CRNN architecture with ResNet18 + BiGRU + Attention:

1. **Architecture**: Use the provided `CRNNWithAttn` class or ensure your model matches this structure
2. **Training Data**: 4-second stereo audio at 16kHz sample rate
3. **Preprocessing**: Mel spectrograms with n_mels=64, n_fft=780, hop_length=195
4. **Output**: Binary classification with sigmoid activation
5. **Augmentation**: Time shift, noise, pitch shift, spectral masking (as shown in your training code)

## Privacy & Security

- **No Data Collection**: All processing happens locally on the device
- **No Network Requests**: Model inference is completely offline
- **Secure Audio**: Audio is processed in memory and not stored permanently
- **Permission Transparency**: All required permissions are clearly explained

## Technical Specifications

- **Minimum Android Version**: API 24 (Android 7.0)
- **Target Architecture**: ARM64-v8a, ARMv7
- **Model Format**: PyTorch Mobile (.pt)
- **Audio Format**: 16kHz stereo PCM
- **Input Dimensions**: [2, 64, 300] mel spectrogram per 4-second chunk
- **Model Architecture**: ResNet18 + BiGRU + Attention (~11M parameters)
- **Detection Latency**: ~200-800ms per chunk (depending on device)
- **Memory Usage**: ~50-100MB during inference

## Troubleshooting

### Model Loading Issues
- Ensure model file is in `app/src/main/assets/models/deepfake_detector.pt`
- Check that model input shape is [1, 2, 64, 300] (stereo mel spectrograms)
- Verify model was converted with the correct CRNN architecture
- Test conversion with `convert_and_test.py` before deploying

### Permission Issues
- Grant all required permissions through the app settings
- Enable overlay permissions for call-time display
- Check phone state access permissions
- Ensure foreground service permissions are granted

### Audio Capture Issues
- Test on different devices (call audio access varies by manufacturer)
- Ensure `VOICE_CALL` audio source is available
- Check if device supports stereo call recording
- Some devices may only provide mono audio from calls

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test thoroughly on real devices
4. Submit a pull request

## License

This project is for educational and research purposes. Ensure compliance with local laws regarding call recording and audio monitoring.

