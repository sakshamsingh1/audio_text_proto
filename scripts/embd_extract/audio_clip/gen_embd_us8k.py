# imports
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# imports from repo
from common_utils import US8k
from scripts.embd_extract.audio_clip.utils import get_norm_audio_embd, get_model

def gen_us8k_audioclip_embd(args):
    model = get_model(args)
    save_path = '../data/processed/us8k_audioclip_embd.pt'

    feat_data = {}

    train_set = US8k()
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=False)

    for paths in tqdm(train_dataloader):
        audio_embd = get_norm_audio_embd(paths, model, mono=False)

        for idx, embd in enumerate(audio_embd):
            path = paths[idx]

            file_name = path.split('/')[-1]
            label = int(file_name.split('.')[0].split('-')[1])
            fold = int(path.split('/')[-2].split('fold')[1])

            feat_data[file_name] = {}
            feat_data[file_name]['class_gt'] = label
            feat_data[file_name]['embd'] = embd
            feat_data[file_name]['fold'] = fold

    torch.save(feat_data, save_path + 'feat_data.pt')
