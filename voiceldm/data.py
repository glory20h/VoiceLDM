import os
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from .audio import raw_waveform_to_fbank, TacotronSTFT


class AudioDataset(Dataset):
    def __init__(self, args, df, df_noise, clap_processor):
        self.df = df
        self.df_noise = df_noise
        
        self.paths = args.paths
        self.noise_paths = args.noise_paths

        self.uncond_text_prob = args.uncond_text_prob
        self.add_noise_prob = args.add_noise_prob
        
        self.duration = 10
        self.target_length = int(self.duration * 102.4)
        self.stft = TacotronSTFT(
            filter_length=1024,
            hop_length=160,
            win_length=1024,
            n_mel_channels=64,
            sampling_rate=16000,
            mel_fmin=0,
            mel_fmax=8000,
        )

        self.clap_processor = clap_processor
        
    def pad_wav(self, wav, target_len, random_cut=False):
        n_channels, wav_len = wav.shape
        if n_channels == 2:
            wav = wav.mean(-2, keepdim=True)

        if wav_len > target_len:
            if random_cut:
                i = random.randint(0, wav_len - target_len)
                return wav[:, i:i+target_len]
            return wav[:, :target_len]
        elif wav_len < target_len:
            wav = F.pad(wav, (0, target_len-wav_len))
        return wav
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        file_path = os.path.join(self.paths[row.data], row.file_path)
        
        waveform, sr = torchaudio.load(file_path)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        waveform = self.pad_wav(waveform, self.target_length * 160)
        
        if random.random() < self.add_noise_prob and row.data in ['cv'] and len(self.df_noise) > 0:
            noise_row = self.df_noise.iloc[random.randint(0, len(self.df_noise)-1)]
            noise, sr = torchaudio.load(os.path.join(self.noise_paths[noise_row.data], noise_row.file_path))
            noise = torchaudio.functional.resample(noise, orig_freq=sr, new_freq=16000)
            noise = self.pad_wav(noise, self.target_length * 160, random_cut=True)
            
            snr = torch.Tensor(1).uniform_(4, 20)
            waveform = torchaudio.functional.add_noise(waveform, noise, snr)

        fbank, _, waveform = raw_waveform_to_fbank(
            waveform[0], 
            target_length=self.target_length, 
            fn_STFT=self.stft
        )
        
        text = row.text
        if random.random() < self.uncond_text_prob:
            text = ""

        # resample to 48k for clap
        wav_48k = torchaudio.functional.resample(waveform, orig_freq=16000, new_freq=48000)
        clap_inputs = self.clap_processor(audios=wav_48k, return_tensors="pt", sampling_rate=48000)
        
        return fbank, waveform, text, clap_inputs
        
    def __len__(self):
        return len(self.df)


class CollateFn:
    def __init__(self, text_processor):
        self.text_processor = text_processor

    def __call__(self, examples):
        fbank = torch.stack([example[0] for example in examples])
        waveform = torch.stack([example[1] for example in examples])
        clap_input_features = torch.cat([example[3].input_features for example in examples])
        clap_is_longer = torch.cat([example[3].is_longer for example in examples])
        
        text_tokens = self.text_processor(
            text=[example[2] for example in examples], 
            padding=True,
            truncation=True,
            max_length=1000,
            return_tensors="pt"
        )

        return {
            "fbank": fbank, 
            "waveform": waveform, 
            "text_tokens": text_tokens,
            "clap_input_features": clap_input_features,
            "clap_is_longer": clap_is_longer,
        }