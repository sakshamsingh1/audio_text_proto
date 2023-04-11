# imports
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# imports from repo
from gen_embd_helpers import *

# constants and variables
model, model_cfg = get_model(param_path)


def gen_esc_embd():
    save_path = '/scratch/sk8974/experiments/audio_text/audio_text_dhh/data/CLAP/esc_feat/'

    feat_data = {}

    train_set = ESC()
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=False)

    for paths in tqdm(train_dataloader):
        # print(f'Batch {i}')
        # import pdb; pdb.set_trace()
        audio_embd = get_audio_embd(paths, model, model_cfg)

        for idx, embd in enumerate(audio_embd):
            path = paths[idx]

            file_name = path.split('/')[-1]
            label = int(file_name.split('.')[0].split('-')[-1])
            fold = int(file_name.split('.')[0].split('-')[0])

            feat_data[file_name] = {}
            feat_data[file_name]['class_gt'] = label
            feat_data[file_name]['embd'] = embd
            feat_data[file_name]['fold'] = fold

    torch.save(feat_data, save_path + 'feat_data.pt')