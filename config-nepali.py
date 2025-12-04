import json
import os

# This config EXACTLY matches your FastSpeech2 mel parameters
hifigan_config = {
    "resblock": "1",
    "num_gpus": 1,
    "batch_size": 16,  # Optimized for T4 GPU
    "learning_rate": 0.0002,
    "adam_!git clone https://github.com/jik876/hifi-ganb1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.999,
    "seed": 1234,
    
    # CRITICAL: These MUST match your FastSpeech2 hparams.py
    "upsample_rates": [8, 8, 2, 2],  # Product = 256 (your hop_length)
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    
    # Audio parameters matching your FastSpeech2
    "segment_size": 8192,
    "num_mels": 80,  # Your n_mel_channels
    "num_freq": 513,  # (filter_length // 2) + 1 = (1024 // 2) + 1
    "n_fft": 1024,  # Your filter_length
    "hop_size": 256,  # Your hop_length
    "win_size": 1024,  # Your win_length
    "sampling_rate": 22050,  # Your sample_rate
    "fmin": 0.0,  # Your mel_fmin
    "fmax": 8000.0,  # Your mel_fmax
    "fmax_for_loss": None,
    
    "num_workers": 4,
    "dist_config": {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "world_size": 1
    }
}