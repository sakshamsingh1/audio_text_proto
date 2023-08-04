import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm

from common_utils import get_embdDim, get_map, FSD_data, MLPLayers, get_clap_model, get_fsd_labels
from torch.utils.data import DataLoader
from scripts.audioclip_utils import get_audioclip_model, get_text_embd

import warnings
warnings.filterwarnings('ignore')


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


def train_sv_fsd(model_type):
    embd_dim = get_embdDim(model_type)
    net = MLPLayers([embd_dim, embd_dim*2, 200])
    criterion = nn.BCEWithLogitsLoss()
    
    EPOCHS = 30
    BATCH_SIZE = 128
    NUM_WORKERS = 0
    
    model_path = 'NN_baseline_fsd.pt'
    data_type = 'fsd50k'
    trainset = FSD_data(data_type, model_type)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    valset = FSD_data(data_type, model_type,fold='validation')
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    testset = FSD_data(data_type, model_type, fold='test')
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1.0e-8)

    best_val_map = 0
    for epoch in range(EPOCHS): 
        net.train()
        loss_sum= 0.0

        for data, labels in trainloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            optimizer.zero_grad()
            out = net(data)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        train_loss, train_map = val_epoch_fsd(net, trainloader, criterion)
        val_loss, val_map = val_epoch_fsd(net, valloader, criterion)
        
        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(net.state_dict(), model_path)
        
        if epoch%10==0:
            print(f'epoch:{epoch}, loss|mAP train:{train_loss}|{train_map}, Val:{val_loss}|{val_map}')

    test_net = MLPLayers([embd_dim, embd_dim*2, 200])
    if torch.cuda.is_available():
        test_net = test_net.cuda()
    test_net.load_state_dict(torch.load(model_path))
    test_loss, test_map = val_epoch_fsd(test_net, testloader, criterion)
    print(f'test map: {test_map}')

def run_zs_fsd(model_type):
    BATCH_SIZE = 32
    NUM_WORKERS = 1
    data_type = 'fsd50k'
    test_set = FSD_data(data_type, model_type, fold='test')
    testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    if model_type == 'audioclip':
        PROMPT = 'This is '
        model = get_audioclip_model()

    elif model_type == 'clap':
        PROMPT = 'This is a sound of '
        model = get_clap_model()

    label_map = get_fsd_labels()

    labels = []
    for i in range(len(label_map)):
        labels.append(label_map[i])

    text_data = [[f'{PROMPT}{label}'] for label in labels]
    map_sum = 0

    if model_type == 'audioclip':

        text_embd = get_text_embd(text_data)
        scale_audio_text = torch.clamp(model.logit_scale_at.exp(), min=1.0, max=100.0)
        
        for data, batch_label in tqdm(testloader):
            logits_audio_text = scale_audio_text * data @ text_embd.T
            map_sum += get_map(logits_audio_text.detach().cpu(),batch_label,use_sig=False)

        zs_map = round(map_sum / len(testloader),4)
    
    elif model_type == 'clap':
        for data, batch_label in tqdm(testloader):
            text_features = model.get_text_embedding(text_data)
            logits_audio_text = (data @ torch.tensor(text_features).t()).detach().cpu()
            map_sum += get_map(logits_audio_text.detach().cpu(),batch_label,use_sig=False)
        zs_map = round(map_sum / len(testloader),4)

    print(f'Zero shot map: {zs_map}')