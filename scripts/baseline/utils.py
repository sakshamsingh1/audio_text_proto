import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from scripts.common_utils import Fold_var


def get_map(pred, target, use_sig=True):
    if use_sig:
        pred = torch.sigmoid(pred).numpy()
    else:
        pred = pred.numpy()
    target = target.cpu().numpy()
    return np.mean(average_precision_score(target, pred, average=None))


class FSD_data(Dataset):
    def __init__(self, fold='train', transform=None, overfit=None):
        base_path = '/scratch/sk8974/experiments/audio_text/audio_text_dhh/data/'
        embd_file = base_path+'audioClip/fsd_feat/feat_data.pt'

        self.obj = Fold_var(embd_file, FOLD=fold)
        
        self.labels = self.obj.curr_true_labels
        self.audios = self.obj.curr_norm_feat
    
        if overfit is not None:
            self.audios = self.audios[:overfit,:]
            self.labels = self.labels[:overfit]
            
        self.transform = transform
        self.num_class = 200

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        audio = self.audios[idx]

        label_oh = torch.zeros(self.num_class)
        label_oh = label_oh.scatter_(0, torch.tensor(label), 1)

        if self.transform is not None:
            audio = self.transform(audio)

        return audio, label_oh


class MLPLayers(nn.Module):
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X        

class US8k_data(Dataset):
    def __init__(self, fold=1, transform=None, train=True, overfit=None):
        label_path = '../../data/us-8k/US8k/metadata/UrbanSound8K.csv'
        path_dir = '../../data/us-8k/US8k_feat/'

        label, label_map = get_us_labels(label_path)
        self.obj = Fold_var(path_dir, label_map, FOLD=fold)
        if train:
            self.labels = self.obj.train_true_labels
            self.audios = self.obj.train_norm_feat
        else:
            self.labels = self.obj.test_true_labels
            self.audios = self.obj.test_norm_feat
        
        if overfit is not None:
            self.audios = self.audios[:overfit,:]
            self.labels = self.labels[:overfit]
            
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        audio = self.audios[idx]
        if self.transform is not None:
            audio = self.transform(audio)
        return audio, label

def val_epoch(model, dataloader, criterion):

    model.eval()  
    loss_sum, cor, total = 0.0,0,0
    with torch.no_grad():
        for data, labels in dataloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            out = model(data)
            loss = criterion(out, labels)
            loss_sum += loss.item()

            pred = out.data.max(1, keepdim=True)[1]
            cor += pred.eq(labels.data.view_as(pred)).cpu().sum()
            total += labels.shape[0]

    epoch_loss = round(loss_sum / len(dataloader),2)
    curr_acc = round((cor*100/total).item(),2)
    return epoch_loss, curr_acc

def val_epoch_fsd(model, dataloader, criterion):
    model.eval()
    loss_sum, map_sum = 0.0,0
    with torch.no_grad():
        for data, labels in dataloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            out = model(data)
            loss = criterion(out, labels)
            loss_sum += loss.item()
            map_sum += get_map(out.detach().cpu(),labels)

    epoch_loss = round(loss_sum / len(dataloader),2)
    epoch_map = round(map_sum / len(dataloader),4)
    
    return epoch_loss, epoch_map