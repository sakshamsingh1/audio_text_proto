from torch.utils.data import Dataset
import soundata
import pickle
from glob import glob
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
import numpy as np

def get_cos_sim(mean_embd, curr_embd):
    cos = nn.CosineSimilarity(dim=1, eps=1e-8)
    return cos(mean_embd, curr_embd)

def get_map(pred, target, use_sig=True):
    if use_sig:
        pred = torch.sigmoid(pred).numpy()
    else:
        pred = pred.numpy()
    target = target.cpu().numpy()
    return np.mean(average_precision_score(target, pred, average=None))


def get_embd_path(dataset_name, model_type):
    return f'data/processed/{dataset_name}_{model_type}_embd.pt'

class FSD50k(Dataset):
    def __init__(self):
        self.paths = []
        self.labels = []
        self.folds = []
        self.fill()

    def fill(self):
        data_path = 'data/input/FSD50K/'
        soundataset = soundata.initialize('fsd50k', data_home=data_path)
        sound_clips = soundataset.load_clips()

        label_file = 'data/processed/fsd50k/name_id_label_map.pkl'
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

class ESC(Dataset):
    def __init__(self):
        audio_base_path = 'data/input/ESC-50/'
        self.paths = glob(audio_base_path+'audio/*.wav')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return path

class US8k(Dataset):
    def __init__(self):
        audio_base_path = 'data/input/US8k/'
        self.paths = glob(audio_base_path+'audio/fold*/*.wav')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return path

def get_embdDim(model_type):
    if model_type == "audioclip":
        return 1024
    elif model_type == "clap":
        return 512
    return None

def get_labelCount(data_type):
    if data_type == "esc50":
        return 50
    elif data_type == "us8k":
        return 10
    elif data_type == "fsd50k":
        return 200
    return None

def read_pkl(file):
    with open(file, "rb") as file:
        data = pickle.load(file)
    return data

def get_fsd_labels():
    base_dir = 'data/processed/fsd50k'
    id_origLabel_map = read_pkl(base_dir+'label_map.pkl')
    orig_clapLabel_map = read_pkl(base_dir+'label_to_clap_label.pkl')

    id_clapLabel_map = {}
    labels = []
    for i in range(200):
        orig_label = id_origLabel_map[i]
        id_clapLabel_map[i] = orig_clapLabel_map[orig_label]
        labels.append(orig_label)
    return id_clapLabel_map

def get_esc50_labels():
    label_path = 'data/input/ESC-50/meta/esc50.csv'
    df = pd.read_csv(label_path)
    df['category'] = df['category'].apply(lambda x : " ".join(x.split('_')))
    label_map = dict(zip(list(df.target),list(df.category)))
    return label_map

def get_us8k_labels():
    label_path = 'data/input/US8k/metadata/UrbanSound8K.csv'
    df = pd.read_csv(label_path)
    df['class_'] = df['class'].apply(lambda x : " ".join(x.split('_')))
    label_map = dict(zip(list(df.classID),list(df.class_)))
    return label_map

def get_label_map(data_type):
    if data_type == "esc50":
        return get_esc50_labels()
    elif data_type == "us8k":
        return get_us8k_labels()
    elif data_type == "fsd50k":
        return get_fsd_labels()
    return None


class Fold_var:
    def __init__(self, data_type, model_type, FOLD='train'):
        # fold can only be {'train', 'validation', 'test'}
        embd_file = get_embd_path(data_type, model_type)
        self.embds = torch.load(embd_file, map_location='cpu')
        self.curr_fold = FOLD
        embd_dim = get_embdDim(model_type)
        self.label_id_name_map = get_label_map(data_type)
        curr_size = self.get_size()
        self.curr_norm_feat = torch.empty(curr_size, embd_dim)
        self.curr_true_labels = []
        self.curr_true_labels_name = []
        self.fill_embd_data()

    def fill_embd_data(self):
        curr_idx = 0
        for file, info in self.embds.items():
            if self.curr_fold == info['fold']:
                self.curr_norm_feat[curr_idx, :] = info['embd']
                curr_label = info['class_gt']
                self.curr_true_labels.append(curr_label)
                self.curr_true_labels_name.append([self.label_id_name_map[l] for l in curr_label])
                curr_idx += 1

    def get_size(self):
        curr_size = 0
        for _, info in self.embds.items():
            if self.curr_fold == info['fold']:
                curr_size += 1
        return curr_size


class Fold_proto:
    def __init__(self, data_type, model_type):
        self.num_class = get_labelCount(data_type)

        train_data = Fold_var(data_type, model_type, FOLD='train')
        self.train_norm_feat = train_data.curr_norm_feat
        self.train_true_labels = train_data.curr_true_labels

        test_data = Fold_var(data_type, model_type, FOLD='test')
        self.test_norm_feat = test_data.curr_norm_feat
        self.test_true_labels = test_data.curr_true_labels


