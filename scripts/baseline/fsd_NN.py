import torch.nn as nn
import torch
import torch.optim as optim

from scripts.baseline.utils import MLPLayers, FSD_data, val_epoch_fsd, val_epoch
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

#helper functions
def run_fsd_NN(model_type):
    net = MLPLayers([1024,1024*2,200])
    criterion = nn.BCEWithLogitsLoss()
    
    EPOCHS = 100
    BATCH_SIZE = 160
    NUM_WORKERS = 4
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_path = 'NN_baseline_fsd.pt'

    trainset = FSD_data()
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    valset = FSD_data(fold='validation')
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    testset = FSD_data(fold='test')
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
            print(f'epoch:{epoch}, loss|mAP train:{train_loss}|{train_map}; Val:{val_loss}|{val_map}')

    test_loss, test_map = val_epoch(net, testloader, criterion)
    print(f'test map: {test_map}')



    



