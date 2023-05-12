#imports
import sys
import soundfile as sf
import librosa
import numpy as np
import torchvision as tv
import torch

from scripts.ref_repo.AudioCLIP.model import *
from scripts.ref_repo.AudioCLIP.utils.transforms import ToTensor1D, RandomPadding, RandomCrop

### AudioClip ###
def get_model(args):
    DIR = 'data/input/pretrained/'
    MODEL_FILENAME = DIR+'AudioCLIP-Full-Training.pt'
    aclp = AudioCLIP(pretrained=MODEL_FILENAME)
    aclp.to(args.device)
    return aclp, None

def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    return new_value

audio_transforms = []
audio_transforms.append(ToTensor1D())
audio_transforms.append(RandomPadding(out_len=176400, train=False))
audio_transforms.append(RandomCrop(out_len=176400, train=False))
transforms_test = tv.transforms.Compose(audio_transforms)

def get_norm_audio_embd(paths_to_audio, aclp, mono=True):
    sample_rate = 44100
    audio_list = list()

    for path_to_audio in paths_to_audio:
        wav, sample_rate_ = sf.read(
            path_to_audio,
            dtype='float32',
            always_2d=True
        )
        wav = librosa.resample(wav.T, sample_rate_, sample_rate)

        if wav.shape[0] == 1 and not mono:
            wav = np.concatenate((wav, wav), axis=0)

        wav = wav[:, :sample_rate * 4]
        wav = scale(wav, wav.min(), wav.max(), -32768.0, 32767.0).astype(np.float32)
        audio = transforms_test(wav)
        audio_list.append(audio)

    audio_batch = torch.stack([track.reshape(1, -1) for track in audio_list])
    aclp.eval()
    with torch.no_grad():
        ((audio_features, _, _), _), _ = aclp(audio=audio_batch)

    audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
    return audio_features