import os
import random
from pathlib import Path

def prepare_hifigan_data(wav_dir, mel_dir, output_dir, train_ratio=0.95, verify_mels=True):
    wav_dir = Path(wav_dir)
    mel_dir = Path(mel_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wav_files = sorted(list(wav_dir.glob('*.wav')))
    print(f"Found {len(wav_files)} wav files")
    
    if verify_mels:
        valid_wav_files = []
        for wav_file in wav_files:
            mel_file = mel_dir / f"{wav_file.stem}.npy"
            if mel_file.exists():
                valid_wav_files.append(wav_file)
        wav_files = valid_wav_files
        print(f"Verified {len(wav_files)} wav-mel pairs")
    
    random.seed(1234)
    random.shuffle(wav_files)
    
    split_idx = int(len(wav_files) * train_ratio)
    train_files = wav_files[:split_idx]
    val_files = wav_files[split_idx:]
    
    train_path = output_dir / 'training.txt'
    with open(train_path, 'w') as f:
        for wav_file in train_files:
            f.write(f"{wav_file.name}\n")
    
    val_path = output_dir / 'validation.txt'
    with open(val_path, 'w') as f:
        for wav_file in val_files:
            f.write(f"{wav_file.name}\n")
    
    print(f"✓ Train: {len(train_files)} | Val: {len(val_files)}")
    return str(train_path), str(val_path)

# Run it
WAV_DIR = '/Users/ashutoshbhattarai/Downloads/data/wavs'
MEL_DIR = '//Users/ashutoshbhattarai/Downloads/mel'  
OUTPUT_DIR = './filelists'

prepare_hifigan_data(WAV_DIR, MEL_DIR, OUTPUT_DIR)