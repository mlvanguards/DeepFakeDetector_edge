#!/usr/bin/env python3
"""
Convert PyTorch .pth CRNN model file to .pt format for Android deployment

This script converts your CRNN deepfake detection model (with ResNet18 + BiGRU + Attention)
to a format that can be used with PyTorch Mobile on Android.

Usage:
    python convert_model.py --input model.pth --output deepfake_detector.pt --model_class CRNNWithAttn

Requirements:
    - PyTorch
    - torchvision
    - The original model class definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import os
import sys
from pathlib import Path

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
        
        # Modify first conv layer for 2-channel input (stereo audio)
        w = resnet.conv1.weight.data.clone()
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data[:, 0] = w[:, 0]  # Copy first channel weights
        resnet.conv1.weight.data[:, 1] = w[:, 0]  # Duplicate for second channel
        
        # Remove final pooling & fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 2. Bi-GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=512,          # ResNet last block outputs 512 channels
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
        # x: [B, 2, F, T] - stereo mel spectrogram
        feat = self.backbone(x)            # [B, 512, F', T']
        feat = feat.mean(dim=2)            # collapse freq → [B, 512, T']
        feat = feat.permute(0, 2, 1)       # → [B, T', 512]

        out, _ = self.gru(feat)            # → [B, T', 2*hidden_size]
        pooled = self.attn_pool(out)       # → [B, 2*hidden_size]
        logit = self.classifier(pooled)    # → [B, 1]
        
        # Convert to probabilities for binary classification
        prob_fake = torch.sigmoid(logit)   # Probability of being fake
        prob_real = 1 - prob_fake          # Probability of being real
        
        return torch.cat([prob_real, prob_fake], dim=1)  # [B, 2] format expected by Android

def load_model_from_pth(pth_path, model_class, **model_kwargs):
    """
    Load model from .pth state dict file
    
    Args:
        pth_path: Path to .pth file
        model_class: Model class to instantiate
        **model_kwargs: Arguments for model constructor
        
    Returns:
        Loaded PyTorch model
    """
    print(f"Loading model state dict from: {pth_path}")
    
    # Load the state dict
    if torch.cuda.is_available():
        state_dict = torch.load(pth_path)
    else:
        state_dict = torch.load(pth_path, map_location='cpu')
    
    # Handle different state dict formats
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        # Training checkpoint format
        state_dict = state_dict['model_state_dict']
        print("Found model_state_dict in checkpoint")
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        # Alternative checkpoint format
        state_dict = state_dict['state_dict']
        print("Found state_dict in checkpoint")
    
    # Create model instance
    model = model_class(**model_kwargs)
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Successfully loaded state dict (strict mode)")
    except Exception as e:
        print(f"⚠️ Strict loading failed: {e}")
        try:
            model.load_state_dict(state_dict, strict=False)
            print("✅ Successfully loaded state dict (non-strict mode)")
        except Exception as e2:
            print(f"❌ Error loading state dict: {e2}")
            print("Available keys in state dict:")
            for key in state_dict.keys():
                print(f"  - {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else type(state_dict[key])}")
            print("\nModel parameters:")
            for name, param in model.named_parameters():
                print(f"  - {name}: {param.shape}")
            raise
    
    return model

def convert_to_mobile(model, output_path, input_shape=(1, 2, 64, 300)):
    """
    Convert model to TorchScript and optimize for mobile deployment
    
    Args:
        model: PyTorch model
        output_path: Path to save .pt file
        input_shape: Input tensor shape for tracing [batch, channels, freq_bins, time_steps]
    """
    print("Converting model to TorchScript...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create example input for tracing
    example_input = torch.randn(input_shape)
    print(f"Using example input shape: {example_input.shape}")
    
    try:
        # Option 1: Tracing (recommended for feedforward models)
        print("Attempting tracing...")
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        
        # Verify the traced model works
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = traced_model(example_input)
            
            # Check if outputs are close
            if torch.allclose(original_output, traced_output, atol=1e-4):
                print("✅ Tracing verification passed")
            else:
                print("⚠️ Tracing verification failed, outputs differ")
                print(f"Original output shape: {original_output.shape}")
                print(f"Traced output shape: {traced_output.shape}")
                print(f"Max difference: {torch.max(torch.abs(original_output - traced_output))}")
        
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        print("Attempting scripting instead...")
        
        # Option 2: Scripting (fallback)
        try:
            traced_model = torch.jit.script(model)
            print("✅ Scripting successful")
        except Exception as e2:
            print(f"❌ Scripting also failed: {e2}")
            raise
    
    # Optimize for mobile
    print("Optimizing for mobile deployment...")
    try:
        optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
        print("✅ Mobile optimization successful")
    except Exception as e:
        print(f"⚠️ Mobile optimization failed: {e}")
        print("Proceeding with non-optimized model...")
        optimized_model = traced_model
    
    # Save the model
    print(f"Saving optimized model to: {output_path}")
    optimized_model.save(output_path)
    
    # Verify the saved model can be loaded
    try:
        loaded_model = torch.jit.load(output_path)
        test_output = loaded_model(example_input)
        print("✅ Saved model verification passed")
        print(f"Model output shape: {test_output.shape}")
        print(f"Model output sample: {test_output[0]}")
        
        # Check if output is in probability format
        if test_output.shape[1] == 2:
            prob_real, prob_fake = test_output[0]
            print(f"Sample prediction - Real: {prob_real:.4f}, Fake: {prob_fake:.4f}")
        
    except Exception as e:
        print(f"❌ Saved model verification failed: {e}")
        raise
    
    return optimized_model

def analyze_model(model, example_input):
    """
    Analyze model properties for deployment
    """
    print("\n=== MODEL ANALYSIS ===")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Estimate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    print(f"Estimated model size: {model_size_mb:.2f} MB")
    
    # Test inference time
    model.eval()
    with torch.no_grad():
        import time
        
        # Warmup
        for _ in range(10):
            _ = model(example_input)
        
        # Time inference
        start_time = time.time()
        num_runs = 100
        for _ in range(num_runs):
            output = model(example_input)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / num_runs * 1000
        print(f"Average inference time: {avg_time_ms:.2f} ms")
        print(f"Output shape: {output.shape}")
        
        # Show prediction
        if output.shape[1] == 2:
            prob_real, prob_fake = output[0]
            print(f"Example prediction - Real: {prob_real:.4f}, Fake: {prob_fake:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Convert CRNN .pth model to .pt for Android deployment')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input .pth file path')
    parser.add_argument('--output', '-o', type=str, help='Output .pt file path (default: input.pt)')
    parser.add_argument('--hidden_size', type=int, default=128, help='GRU hidden size (default: 128)')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of GRU layers (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained ResNet18')
    parser.add_argument('--input_shape', type=str, default='1,2,64,300', help='Input shape for tracing (default: 1,2,64,300)')
    parser.add_argument('--analyze', action='store_true', help='Analyze model properties')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.with_suffix('.pt'))
    
    # Parse input shape
    try:
        input_shape = tuple(map(int, args.input_shape.split(',')))
    except:
        print(f"❌ Invalid input shape format: {args.input_shape}")
        print("Use format like: 1,2,64,300")
        sys.exit(1)
    
    print("🚀 Starting CRNN model conversion...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Input shape: {input_shape}")
    
    try:
        # Load model from .pth
        model_kwargs = {
            'pretrained': args.pretrained,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
        
        model = load_model_from_pth(args.input, CRNNWithAttn, **model_kwargs)
        
        # Analyze model if requested
        if args.analyze:
            example_input = torch.randn(input_shape)
            analyze_model(model, example_input)
        
        # Convert to mobile format
        mobile_model = convert_to_mobile(model, args.output, input_shape)
        
        print(f"\n✅ Conversion successful!")
        print(f"📱 Mobile-optimized model saved to: {args.output}")
        print(f"📊 File size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")
        
        # Instructions for Android integration
        print("\n🔧 ANDROID INTEGRATION INSTRUCTIONS:")
        print("1. Copy the .pt file to: app/src/main/assets/models/deepfake_detector.pt")
        print("2. The model expects mel spectrogram input shape:", input_shape)
        print("3. Input format: [batch, 2_channels, 64_mel_bins, 300_time_steps]")
        print("4. Output: 2 classes [real_probability, fake_probability]")
        print("5. Audio specs: 16kHz, stereo, 4-second duration")
        print("6. Mel spectrogram: n_mels=64, n_fft=780, hop_length=195")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
