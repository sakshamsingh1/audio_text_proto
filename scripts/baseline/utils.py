import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset  

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

