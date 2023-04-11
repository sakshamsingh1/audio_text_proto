#imports
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import sys
import librosa
import torch

sys.path.append('/scratch/sk8974/experiments/audio_text/audio_text_dhh/ref_repo/CLAP/src/')
from open_clip import create_model
from training.data import get_audio_features
from training.data import int16_to_float32, float32_to_int16


#constants and variables
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = 'model_K2C_fusion'
enable_fusion = True
data_truncating = 'fusion' # 'rand_trunc' # if run in unfusion mode

base_path = f'/scratch/sk8974/experiments/audio_text/audio_text_dhh/scripts/CLAP/pretrain_models/{model_name}/'
param_path = base_path + 'params.txt'
pretrained = base_path + "checkpoints/epoch_top_0.pt"

data_base_path = '/scratch/sk8974/experiments/audio_text/audio_text_dhh/data/esc/'
label_path = data_base_path+'esc50.csv'
audio_base_path = data_base_path+'audio/'

def find_params_value(file, key):
    # find value of params in params_file
    with open(file, 'r') as f:
        for line in f:
            if key + ': ' in line:
                return line.split(': ')[1].strip()
    return None

tokenize = RobertaTokenizer.from_pretrained('roberta-base')
def tokenizer(text):
    result = tokenize(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in result.items()}


# model
def get_model(param_path):
    precision = 'fp32'

    amodel = find_params_value(param_path, 'amodel')
    tmodel = find_params_value(param_path, 'tmodel')

     # False if you do not want to use the fusion model
    fusion_type = 'aff_2d'
     # the checkpoint name, the unfusion model can also be loaded.

    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision=precision,
        device=device,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type
    )

    model.to(device)
    model.eval()
    return model, model_cfg

def get_audio_embd(audio_paths, model, model_cfg):
    audio_input = []
    for audio_path in audio_paths:
        audio_waveform, sr = librosa.load(audio_path, sr=48000)
        audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        audio_dict = {}

        audio_dict = get_audio_features(
            audio_dict, audio_waveform, 480000,
            data_truncating=data_truncating,
            data_filling='repeatpad',
            audio_cfg=model_cfg['audio_cfg']
        )
        audio_input.append(audio_dict)
    # can send a list to the model, to process many audio tracks in one time (i.e. batch size)
    audio_embed = model.get_audio_embedding(audio_input)
    return audio_embed.detach().cpu()

# audio
class ESC(Dataset):
    def __init__(self):
        audio_base_path = '/scratch/sk8974/experiments/audio_text/audio_text_dhh/data/esc/'
        self.paths = glob.glob(audio_base_path+'audio/*.wav')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return path