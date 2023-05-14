# imports
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

from common_utils import ESC, FSD50k, US8k
from scripts.embd_extract.audio_clip.utils import get_norm_audio_embd, get_model

def gen_audioclip_embd(args):
    model = get_model(args)
    save_path = f'data/processed/{args.dataset_name}_audioclip_embd.pt'

    feat_data = {}

    train_set = None
    if args.dataset_name == 'esc50':
        train_set = ESC()
    elif args.dataset_name == 'us8k':
        train_set = US8k()
    elif args.dataset_name == 'fsd50k':
        train_set = FSD50k()
    else:
        raise ValueError(f'Invalid dataset name: {args.dataset_name}')

    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=False)

    if args.dataset_name == 'fsd50k':
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

    else:
        for paths in tqdm(train_dataloader):
            audio_embd = get_norm_audio_embd(paths, model, mono=False)

            for idx, embd in enumerate(audio_embd):
                path = paths[idx]

                file_name, label, fold = None, None, None
                if args.dataset_name == 'esc50':
                    file_name = path.split('/')[-1]
                    label = int(file_name.split('.')[0].split('-')[-1])
                    fold = int(file_name.split('.')[0].split('-')[0])

                elif args.dataset_name == 'us8k':
                    file_name = path.split('/')[-1]
                    label = int(file_name.split('.')[0].split('-')[1])
                    fold = int(path.split('/')[-2].split('fold')[1])

                feat_data[file_name] = {}
                feat_data[file_name]['class_gt'] = label
                feat_data[file_name]['embd'] = embd
                feat_data[file_name]['fold'] = fold

    torch.save(feat_data, save_path)