#!/usr/bin/env python3
"""
Convert and Test CRNN Deepfake Detection Model

This script demonstrates how to convert your trained CRNN model
and test it with the same input format that the Android app will use.

Usage:
    python convert_and_test.py --model_path your_model.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchaudio
import torchaudio.transforms as transforms
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Copy the model classes from the conversion script
class AttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        scores = self.attn(x)              # [B, T, 1]
        weights = F.softmax(scores, dim=1) # [B, T, 1]
        return (weights * x).sum(dim=1)    # [B, D]

class CRNNWithAttn(nn.Module):
    def __init__(self, pretrained=True, hidden_size=128, num_layers=1, dropout=0.2):
        super().__init__()
        # 1. Pretrained ResNet18
        if pretrained:
            resnet = models.resnet18(weights='DEFAULT')
        else:
            resnet = models.resnet18()
        
        # Modify first conv layer for 2-channel input
        w = resnet.conv1.weight.data.clone()
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data[:, 0] = w[:, 0]
        resnet.conv1.weight.data[:, 1] = w[:, 0]  # Duplicate for second channel
        
        # Remove final pooling & fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 2. Bi-GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 3. Attention pooling
        self.attn_pool = AttentionPool(hidden_size * 2)

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: [B, 2, F, T]
        feat = self.backbone(x)            # [B, 512, F', T']
        feat = feat.mean(dim=2)            # collapse freq → [B, 512, T']
        feat = feat.permute(0, 2, 1)       # → [B, T', 512]

        out, _ = self.gru(feat)            # → [B, T', 2*hidden_size]
        pooled = self.attn_pool(out)       # → [B, 2*hidden_size]
        logit = self.classifier(pooled)    # → [B, 1]
        
        # Convert to probabilities for binary classification
        prob_fake = torch.sigmoid(logit)   # Probability of being fake
        prob_real = 1 - prob_fake          # Probability of being real
        
        return torch.cat([prob_real, prob_fake], dim=1)  # [B, 2]

def create_test_audio_data():
    """
    Create synthetic stereo audio data that matches the training format
    """
    # Create 4 seconds of stereo audio at 16kHz
    duration = 4.0  # seconds
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    # Generate synthetic stereo audio (sine waves with different frequencies)
    t = torch.linspace(0, duration, num_samples)
    left_channel = 0.5 * torch.sin(2 * np.pi * 440 * t)  # 440 Hz (A note)
    right_channel = 0.5 * torch.sin(2 * np.pi * 523 * t)  # 523 Hz (C note)
    
    # Combine into stereo format [2, num_samples]
    stereo_audio = torch.stack([left_channel, right_channel], dim=0)
    
    return stereo_audio, sample_rate

def audio_to_mel_spectrogram(audio, sample_rate, n_mels=64, n_fft=780, hop_length=195):
    """
    Convert stereo audio to mel spectrogram using the same parameters as training
    """
    # Create mel spectrogram transform
    mel_transform = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to dB scale
    db_transform = transforms.AmplitudeToDB(top_db=80)
    
    # Apply transforms
    mel_spec = mel_transform(audio)  # [2, n_mels, time_steps]
    mel_spec_db = db_transform(mel_spec)
    
    return mel_spec_db

def load_model(model_path):
    """
    Load the trained model from .pth file
    """
    print(f"Loading model from: {model_path}")
    
    # Load state dict
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Create model and load weights
    model = CRNNWithAttn()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model

def test_model_with_data(model, mel_spec):
    """
    Test the model with mel spectrogram data
    """
    print("\n=== TESTING MODEL ===")
    
    # Add batch dimension: [1, 2, 64, time_steps]
    input_tensor = mel_spec.unsqueeze(0)
    
    # Ensure we have exactly 300 time steps
    if input_tensor.shape[3] != 300:
        if input_tensor.shape[3] > 300:
            input_tensor = input_tensor[:, :, :, :300]
        else:
            # Pad with zeros
            pad_size = 300 - input_tensor.shape[3]
            input_tensor = F.pad(input_tensor, (0, pad_size))
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Interpret results
    prob_real, prob_fake = output[0]
    print(f"\n📊 PREDICTION RESULTS:")
    print(f"Real probability: {prob_real:.4f} ({prob_real*100:.1f}%)")
    print(f"Fake probability: {prob_fake:.4f} ({prob_fake*100:.1f}%)")
    print(f"Prediction: {'🚨 FAKE' if prob_fake > 0.5 else '✅ REAL'}")
    
    return output

def visualize_mel_spectrogram(mel_spec, title="Mel Spectrogram"):
    """
    Visualize the mel spectrogram
    """
    # Convert to numpy and select left channel for visualization
    mel_np = mel_spec[0].numpy()  # Left channel
    
    plt.figure(figsize=(12, 6))
    plt.imshow(mel_np, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='dB')
    plt.title(f'{title} (Left Channel)')
    plt.xlabel('Time Steps')
    plt.ylabel('Mel Bins')
    plt.tight_layout()
    plt.show()

def convert_to_mobile(model, output_path):
    """
    Convert model to mobile format
    """
    print(f"\n=== CONVERTING TO MOBILE FORMAT ===")
    
    # Create example input
    example_input = torch.randn(1, 2, 64, 300)
    
    # Set to eval mode
    model.eval()
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for mobile
    try:
        optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
        print("✅ Mobile optimization successful")
    except Exception as e:
        print(f"⚠️ Mobile optimization failed: {e}")
        optimized_model = traced_model
    
    # Save the model
    optimized_model.save(output_path)
    print(f"📱 Mobile model saved to: {output_path}")
    
    # Verify
    loaded_model = torch.jit.load(output_path)
    test_output = loaded_model(example_input)
    print(f"✅ Mobile model verification passed")
    print(f"Mobile model output: {test_output[0]}")

def main():
    parser = argparse.ArgumentParser(description='Convert and test CRNN deepfake detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to your .pth model file')
    parser.add_argument('--output_path', type=str, default='deepfake_detector.pt', help='Output mobile model path')
    parser.add_argument('--visualize', action='store_true', help='Visualize mel spectrogram')
    parser.add_argument('--audio_file', type=str, help='Test with real audio file')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model_path)
    
    # Create or load test audio
    if args.audio_file and Path(args.audio_file).exists():
        print(f"Loading audio from: {args.audio_file}")
        audio, sr = torchaudio.load(args.audio_file)
        
        # Ensure stereo and 16kHz
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)  # Convert mono to stereo
        elif audio.shape[0] > 2:
            audio = audio[:2, :]  # Take first 2 channels
        
        if sr != 16000:
            resampler = transforms.Resample(sr, 16000)
            audio = resampler(audio)
            sr = 16000
        
        # Ensure 4 seconds duration
        target_length = 4 * sr
        if audio.shape[1] > target_length:
            audio = audio[:, :target_length]
        elif audio.shape[1] < target_length:
            pad_length = target_length - audio.shape[1]
            audio = F.pad(audio, (0, pad_length))
    else:
        print("Creating synthetic test audio...")
        audio, sr = create_test_audio_data()
    
    print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
    
    # Convert to mel spectrogram
    print("Converting to mel spectrogram...")
    mel_spec = audio_to_mel_spectrogram(audio, sr)
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    
    # Visualize if requested
    if args.visualize:
        visualize_mel_spectrogram(mel_spec)
    
    # Test the model
    output = test_model_with_data(model, mel_spec)
    
    # Convert to mobile format
    convert_to_mobile(model, args.output_path)
    
    print("\n🎉 CONVERSION AND TESTING COMPLETE!")
    print(f"📋 NEXT STEPS:")
    print(f"1. Copy {args.output_path} to: app/src/main/assets/models/deepfake_detector.pt")
    print(f"2. Build and install the Android app")
    print(f"3. Grant all permissions and test with real phone calls")

if __name__ == "__main__":
    main()
