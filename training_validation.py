import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.pdd_ginconv import PDD_GINConvNet
from models.vnoc_ginconv import Vnoc_GINConvNet
from models.pdd_vnoc_ginconv import PDD_Vnoc_GINConvNet
from models.esm_ginconv import ESM_GINConvNet
from models.fri_ginconv import FRI_GINConvNet
import wandb
import random
from utils import *
import argparse

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    wandb.log({"loss": loss.item()}, commit=False)
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


datasets = ['davis', 'kiba']

all_models = {
    'GINConvNet': GINConvNet, 
    'GATNet': GATNet, 
    'GAT_GCN': GAT_GCN, 
    'GCNNet': GCNNet, 
    'PDD_GINConvNet': PDD_GINConvNet, 
    'Vnoc_GINConvNet': Vnoc_GINConvNet, 
    'ESM_GINConvNet': ESM_GINConvNet, 
    'FRI_GINConvNet': FRI_GINConvNet, 
    'PDD_Vnoc_GINConvNet': PDD_Vnoc_GINConvNet
}

parser = argparse.ArgumentParser(description="Run a specific model on a specific dataset.")

parser.add_argument('-d', '--dataset', type=str, choices=datasets, required=True, 
                    help="Dataset name: 'davis' or 'kiba'.")
parser.add_argument('-m', '--model', type=str, choices=list(all_models.keys()), required=True, 
                    help="Model name. Choose from: " + ", ".join(all_models.keys()) + ".")
parser.add_argument('-c', '--cuda', type=int, default=0, 
                    help="CUDA device index (default: 0).")
parser.add_argument('-s', '--seed', type=int, 
                    help="Random seed for reproducibility.")

args = parser.parse_args()

dataset = args.dataset
modeling = all_models[args.model]
model_st = modeling.__name__

# Select CUDA device if applicable
cuda_name = f"cuda:{args.cuda}"
print('cuda_name:', cuda_name)

# Set seed:
if args.seed is not None:
    seed = args.seed
    print("Seed: " + str(seed))
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# wandb.init(project = 'GraphDTA - Validation', config={"architecture": model_st, "dataset": datasets[0]})

# Main program: Train and validate on specified dataset
if __name__ == "__main__":
    print('\nrunning on ', model_st + '_' + dataset )

    if model_st == "ESM_GINConvNet":
        processed_data_file_train = 'data/processed/' + dataset + '_esm_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_esm_test.pt'
    elif model_st == "FRI_GINConvNet":
        processed_data_file_train = 'data/processed/' + dataset + '_deepfri_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_deepfri_test.pt'
    else:
        processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        if model_st == "ESM_GINConvNet":
            train_data = ESM_TestbedDataset(root='data', dataset=dataset+'_esm_train')
            test_data = ESM_TestbedDataset(root='data', dataset=dataset+'_esm_test')
        elif model_st == "FRI_GINConvNet":
            train_data = ESM_TestbedDataset(root='data', dataset=dataset+'_deepfri_train')
            test_data = ESM_TestbedDataset(root='data', dataset=dataset+'_deepfri_test')
        else:
            train_data = TestbedDataset(root='data', dataset=dataset+'_train')
            test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        
        train_size = int(0.8 * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])        
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_test_mse = 1000
        best_test_ci = 0
        best_epoch = -1
        best_val_epoch = -1
        model_file_name = 'trained_models/validation_model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'trained_models/validation_result_' + model_st + '_' + dataset +  '.csv'
        # for epoch in range(NUM_EPOCHS):
        #     train(model, device, train_loader, optimizer, epoch+1)
        #     print('predicting for valid data')
        #     G,P = predicting(model, device, valid_loader)
        #     val = mse(G,P)
        #     if val<best_mse:
        #         best_mse = val
        #         best_epoch = epoch+1
        #         torch.save(model.state_dict(), model_file_name)
        #         print('predicting for test data')
        #         G,P = predicting(model, device, test_loader)
        #         ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]
        #         # wandb.log({"val_mse": val, "rmse": ret[0], "mse": ret[1], "pearson": ret[2], "spearman": ret[3]})
        #         with open(result_file_name,'w') as f:
        #             f.write(','.join(map(str,ret)))
        #         best_test_mse = ret[1]
        #         best_test_ci = ret[-1]
        #         print('*****')
        #         print('mse improved at epoch ', best_epoch, '; best_test_mse: ', best_test_mse,model_st,dataset)
        #         print('*****')
        #     else:
        #         # wandb.log({"val_mse": val})
        #         print(val,'No improvement since epoch ', best_epoch, '; best_test_mse: ', best_test_mse,model_st,dataset)

        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch+1)

            print('predicting for valid data')
            G,P = predicting(model, device, valid_loader)
            val = mse(G,P)

            print('predicting for test data')
            G,P = predicting(model, device, test_loader)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]

            # wandb.log({"val_mse": val, "rmse": ret[0], "mse": ret[1], "pearson": ret[2], "spearman": ret[3]})
            if val < best_mse:
                best_mse = val
                best_val_epoch = epoch+1
                print('*****')
                print('val mse improved at epoch ', best_val_epoch, '; best_val_mse: ', best_mse,model_st,dataset)
                print('*****')
            else:
                print(val, 'No improvement since epoch ', best_val_epoch, '; best_val_mse: ', best_mse,model_st,dataset)

            if ret[1]<best_test_mse:
                best_test_mse = ret[1]
                best_epoch = epoch+1
                torch.save(model.state_dict(), model_file_name)

                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str,ret)))

                print('*****')
                print('mse improved at epoch ', best_epoch, '; best_test_mse: ', best_test_mse,model_st,dataset)
                print('*****')
            else:
                print(ret[1], 'No improvement since epoch ', best_epoch, '; best_test_mse: ', best_test_mse,model_st,dataset)

        model.load_state_dict(torch.load(model_file_name))
        G,P = predicting(model, device, test_loader)
        ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]

        with open(result_file_name,'w') as f:
            f.write(','.join(map(str,ret)))        
        print('best model metrics: ')
        print('rmse = ', ret[0])
        print('mse = ', ret[1])
        print('pearson = ', ret[2])
        print('spearman = ', ret[3])
        print('ci = ', ret[4])
        # wandb.log({"ci": ret[4]})

# wandb.finish()