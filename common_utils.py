from torch.utils.data import Dataset
import soundata
import pickle

class FSD50k(Dataset):
    def __init__(self):
        self.paths = []
        self.labels = []
        self.folds = []
        self.fill()

    def fill(self):
        data_path = 'data/input/fsd50k/'
        soundataset = soundata.initialize('fsd50k', data_home=data_path)
        sound_clips = soundataset.load_clips()

        label_file = 'data/processed/name_id_label_map.pkl'
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