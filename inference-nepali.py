
import torch
import numpy as np
from scipy.io.wavfile import write
from models import Generator
from meldataset import MAX_WAV_VALUE
import json

class HiFiGANVocoder:
    def __init__(self, checkpoint_path, config_path, device='cuda'):
        """
        Initialize HiFi-GAN vocoder
        
        Args:
            checkpoint_path: Path to generator checkpoint (e.g., 'g_00500000')
            config_path: Path to config file
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        # Load config
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        
        # Load generator
        self.generator = Generator(self.h).to(device)
        state_dict_g = torch.load(checkpoint_path, map_location=device)
        self.generator.load_state_dict(state_dict_g['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()
        
        print(f"✓ HiFi-GAN vocoder loaded from {checkpoint_path}")
    
    def infer(self, mel_spectrogram):
        """
        Convert mel-spectrogram to audio
        
        Args:
            mel_spectrogram: numpy array or torch tensor of shape (n_mels, time)
                            This is the output from your FastSpeech2 model
        
        Returns:
            audio: numpy array of audio samples
        """
        # Convert to torch tensor if needed
        if isinstance(mel_spectrogram, np.ndarray):
            mel = torch.FloatTensor(mel_spectrogram)
        else:
            mel = mel_spectrogram
        
        # Add batch dimension if needed
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        
        mel = mel.to(self.device)
        
        # Generate audio
        with torch.no_grad():
            audio = self.generator(mel)
            audio = audio.squeeze()
            audio = audio.cpu().numpy()
        
        # Denormalize
        audio = audio * MAX_WAV_VALUE
        audio = audio.astype('int16')
        
        return audio
    
    def save_audio(self, mel_spectrogram, output_path, sample_rate=22050):
        """
        Generate and save audio file
        
        Args:
            mel_spectrogram: mel output from FastSpeech2
            output_path: where to save the wav file
            sample_rate: audio sample rate (should match your training)
        """
        audio = self.infer(mel_spectrogram)
        write(output_path, sample_rate, audio)
        print(f"✓ Audio saved to {output_path}")
        return audio


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# EXAMPLE USAGE WITH YOUR FASTSPEECH2
# ====================================
if __name__ == '__main__':
    # Initialize vocoder
    vocoder = HiFiGANVocoder(
        checkpoint_path='',
        config_path='',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Example: Load mel from your FastSpeech2 output
    # mel_output = your_fastspeech2_model.inference(text)
    # audio = vocoder.infer(mel_output)
    
    # Or load from saved numpy file
    mel = np.load('your_fastspeech2_mel_output.npy')
    vocoder.save_audio(mel, 'output.wav')
