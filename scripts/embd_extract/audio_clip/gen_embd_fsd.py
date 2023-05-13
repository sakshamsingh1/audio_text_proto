# imports
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from common_utils import FSD50k
from scripts.embd_extract.audio_clip.utils import get_norm_audio_embd, get_model

def gen_fsdk_audioclip_embd(args):
    save_path = 'data/processed/fds50k_audioclip_embd.pt'
    model = get_model(args)

    feat_data = {}

    train_set = FSD50k()
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=False, collate_fn=lambda x: x, num_workers=3)

    for batch_data in tqdm(train_dataloader):
        paths = [data[0] for data in batch_data]
        labels = [data[1] for data in batch_data]
        folds = [data[2] for data in batch_data]

        audio_embd = get_norm_audio_embd(paths, model, mono=False)

        for idx, embd in enumerate(audio_embd):
            path = paths[idx]

            file_name = path.split('/')[-1]
            label = labels[idx]
            fold = folds[idx]

            feat_data[file_name] = {}
            feat_data[file_name]['class_gt'] = label
            feat_data[file_name]['embd'] = embd
            feat_data[file_name]['fold'] = fold

    torch.save(feat_data, save_path)