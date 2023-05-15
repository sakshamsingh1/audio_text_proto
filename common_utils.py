from torch.utils.data import Dataset
import soundata
import pickle
from glob import glob
import pandas as pd

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