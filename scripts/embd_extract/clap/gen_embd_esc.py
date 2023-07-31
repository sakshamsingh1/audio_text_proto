# imports
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

from common_utils import ESC
from common_utils import get_clap_model

def gen_esc_audioclip_embd(args):
    model = get_model(args)
    save_path = 'data/processed/esc_audioclip_embd.pt'

    feat_data = {}

    train_set = ESC()
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=False)

    for paths in tqdm(train_dataloader):
        audio_embd = get_norm_audio_embd(paths, model, mono=False)

        for idx, embd in enumerate(audio_embd):
            path = paths[idx]

            file_name = path.split('/')[-1]
            label = int(file_name.split('.')[0].split('-')[-1])
            fold = int(file_name.split('.')[0].split('-')[0])

            feat_data[file_name] = {}
            feat_data[file_name]['class_gt'] = label
            feat_data[file_name]['embd'] = embd
            feat_data[file_name]['fold'] = fold

    torch.save(feat_data, save_path)