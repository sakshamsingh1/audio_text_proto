# imports
import soundfile as sf
import sys
import numpy as np
import torchvision as tv

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import torch
import soundata
import pickle
import argparse

# model imports
repo_path = '/scratch/sk8974/experiments/audio_text/audio_text_dhh/ref_repo/AudioCLIP'
sys.path.append(repo_path)
from model import AudioCLIP
from utils.transforms import ToTensor1D, RandomPadding, RandomCrop

torch.set_grad_enabled(False)

# derived from ESResNeXt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
audio_transforms = []

audio_transforms.append(ToTensor1D())
audio_transforms.append(RandomPadding(out_len=176400, train=False))
audio_transforms.append(RandomCrop(out_len=176400, train=False))
transforms_test = tv.transforms.Compose(audio_transforms)

sample_rate = 44100


# model
def get_model():
    DIR = '/scratch/sk8974/experiments/audio_text/audio_text_dhh/data/models/'
    MODEL_FILENAME = DIR + 'AudioCLIP-Full-Training.pt'
    aclp = AudioCLIP(pretrained=MODEL_FILENAME)
    aclp.to(device)
    return aclp, None


def get_text_embd(model, labels):
    # expects all the prompts attached to the labels

    model.eval()
    with torch.no_grad():
        ((_, _, text_features), _), _ = model(text=labels)

    text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

    return text_features


# audioClip helper function
def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    return new_value


def get_norm_audio_embd(paths_to_audio, aclp, mono=True):
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


# audio
class FSD50k(Dataset):
    def __init__(self):
        self.paths = []
        self.labels = []
        self.folds = []
        self.fill()

    def fill(self):
        data_path = '/scratch/sk8974/experiments/audio_text/audio_text_dhh/data/fsd50k/'
        soundataset = soundata.initialize('fsd50k', data_home=data_path)
        sound_clips = soundataset.load_clips()

        label_file = '/scratch/sk8974/experiments/audio_text/audio_text_dhh/data/fsd50k/processed/name_id_label_map.pkl'
        with open(label_file, 'rb') as f:
            label_id_data = pickle.load(f)

        for _, info in sound_clips.items():
            self.paths.append(info.audio_path)

            curr_label = []
            for label in info.tags.labels:
                curr_label.append(label_id_data[label])
            self.labels.append(curr_label)

            self.folds.append(info.split)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        fold = self.folds[idx]

        return path, label, fold
