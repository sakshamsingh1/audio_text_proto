import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common_utils import get_embdDim, get_clap_model, Fold_esc_us8k_basedata, get_fold_count, get_num_class, MLPLayers, get_label_map
from scripts.audioclip_utils import get_audioclip_model, get_text_embd

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

def train_sv_us8k_esc50(data_type, model_type):
    embd_dim = get_embdDim(model_type)
    criterion = nn.CrossEntropyLoss()
    
    EPOCHS = 30
    BATCH_SIZE = 128
    NUM_WORKERS = 0
    
    fold_count = get_fold_count(data_type)
    
    best_vals = []
    for fold in range(1, fold_count+1):
        train_set = Fold_esc_us8k_basedata(data_type, model_type, fold=fold)
        trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        val_set = Fold_esc_us8k_basedata(data_type, model_type, fold=fold, train=False)
        valloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        net = MLPLayers([embd_dim, embd_dim*2, get_num_class(data_type)])
        if torch.cuda.is_available():
            net = net.cuda()

        optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1.0e-8)

        for epoch in range(EPOCHS): 
            net.train()
            loss_sum, cor = 0.0, 0

            for data, labels in trainloader:
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                optimizer.zero_grad()
                out = net(data)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            train_loss = round(loss_sum / len(trainloader), 2)

        val_loss, val_acc = val_epoch(net, valloader, criterion)
        best_vals.append(val_acc)
        print(f'fold:{fold}, val_acc:{val_acc}')

    print(f'Avg val_acc:{sum(best_vals)/len(best_vals)}')

def run_zs_us8k_esc50(data_type, model_type):
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # setting fold to 11 will give us all the data as train data
    test_set = Fold_esc_us8k_basedata(data_type, model_type, fold=11)
    testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    if model_type == 'audioclip':
        PROMPT = 'This is '
        model = get_audioclip_model()

    elif model_type == 'clap':
        PROMPT = 'This is a sound of '
        model = get_clap_model()

    label_map = get_label_map(data_type)

    labels = []
    for i in range(len(label_map)):
        labels.append(label_map[i])

    correct, total = 0, 0

    if model_type == 'audioclip':
        text_data = [[f'{PROMPT}{label}'] for label in labels]
        text_embd = get_text_embd(text_data)
        scale_audio_text = torch.clamp(model.logit_scale_at.exp(), min=1.0, max=100.0)

        for audio_embd, label in tqdm(testloader):
            logits_per_audio = (scale_audio_text * audio_embd @ text_embd.t()).detach().cpu()
            conf, idx = logits_per_audio.topk(1)
            curr_corr = (idx.squeeze() == label).sum()
            correct += curr_corr
            total += len(label)
    
    elif model_type == 'clap':
        text_data = [f'{PROMPT}{label}' for label in labels]
        text_features = model.get_text_embedding(text_data)

        for data, label in tqdm(testloader):
            logits_audio_text = (data @ torch.tensor(text_features).t()).detach().cpu()
            conf, idx = logits_audio_text.topk(1)
            curr_corr = (idx.squeeze() == label).sum()
            correct += curr_corr
            total += len(label)
    
    zs_acc = round(correct.item()*100 / total, 4)
    print(f'Model: {model_type}, Data:{data_type}  Zero shot Acc: {zs_acc}%')


