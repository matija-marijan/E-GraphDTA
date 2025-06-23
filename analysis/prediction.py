import torch
from torch.utils.data import ConcatDataset
import argparse
from utils import *
import matplotlib.pyplot as plt

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

from models.pdc_ginconv import PDC_GINConvNet
from models.vnoc_ginconv import Vnoc_GINConvNet
from models.pdc_vnoc_ginconv import PDC_Vnoc_GINConvNet
from models.esm_ginconv import ESM_GINConvNet
from models.fri_ginconv import FRI_GINConvNet

import csv
import random

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

def plot_histograms(labels, predictions, bin_count = 50, data_dir = 'tmp.png'):
    G = labels
    P = predictions
    xmin = min(G.min(), P.min())
    xmax = max(G.max(), P.max())

    # Define specific bin edges based on combined min and max
    bin_edges = np.linspace(np.floor(xmin), np.ceil(xmax), bin_count)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Histogram for labels
    ax1.hist(G, bins=bin_edges)
    ax1.set_xlabel('Label value - G')
    ax1.set_ylabel('Bin count')
    ax1.grid(True)

    # Histogram for predictions
    ax2.hist(P, bins=bin_edges)
    ax2.set_xlabel('Prediction value - P')
    ax2.set_ylabel('Bin count')
    ax2.grid(True)

    # Set the same axis range for both subplots
    ymin = 0
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([ymin, ymax])
    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([ymin, ymax])

    # Title for the entire figure
    fig.suptitle('Histogram of predictions and labels')
    plt.savefig(data_dir)
    plt.show()

datasets = ['davis', 'kiba']

all_models = {
    'GINConvNet': GINConvNet, 
    'GATNet': GATNet, 
    'GAT_GCN': GAT_GCN, 
    'GCNNet': GCNNet, 
    'PDC_GINConvNet': PDC_GINConvNet, 
    'Vnoc_GINConvNet': Vnoc_GINConvNet, 
    'ESM_GINConvNet': ESM_GINConvNet, 
    'FRI_GINConvNet': FRI_GINConvNet, 
    'PDC_Vnoc_GINConvNet': PDC_Vnoc_GINConvNet
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
parser.add_argument('-x', '--mutation', action='store_true', default=False,
                    help="Flag for including protein sequence mutations for the Davis dataset (default: False).")

args = parser.parse_args()

modeling = all_models[args.model]
model_st = modeling.__name__

dataset = args.dataset
mutation = ''
if dataset == 'davis' and args.mutation:
    mutation = '_mutation'

print(f"Mutation = {args.mutation}")
# Select CUDA device if applicable
cuda_name = f"cuda:{args.cuda}"
print('cuda_name:', cuda_name)
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

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

# TODO: add model_path as an argument
if mutation == '':
    model_path = 'trained_models/final_training_model_' + model_st + '_' + dataset + '.model'
elif mutation == '_mutation':
    model_path = 'trained_models/mutation_training_model_' + model_st + '_' + dataset + '.model'

BATCH_SIZE = 512

if __name__ == "__main__":
    print('\nrunning on ', model_st + '_' + dataset )

    if model_st == "ESM_GINConvNet":
        processed_data_file_train = 'data/processed/' + dataset + mutation + '_esm_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + mutation + '_esm_test.pt'
    elif model_st == "FRI_GINConvNet":
        processed_data_file_train = 'data/processed/' + dataset + mutation + '_deepfri_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + mutation + '_deepfri_test.pt'
    else:
        processed_data_file_train = 'data/processed/' + dataset + mutation + '_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + mutation + '_test.pt'
    test_data_file = 'data/' + dataset + mutation + '_test.csv'

    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        if model_st == "ESM_GINConvNet":
            train_data = ESM_TestbedDataset(root='data', dataset=dataset+ mutation +'_esm_train')
            test_data = ESM_TestbedDataset(root='data', dataset=dataset+ mutation +'_esm_test')
        elif model_st == "FRI_GINConvNet":
            train_data = ESM_TestbedDataset(root='data', dataset=dataset+ mutation +'_deepfri_train')
            test_data = ESM_TestbedDataset(root='data', dataset=dataset+ mutation +'_deepfri_test')
        else:
            train_data = TestbedDataset(root='data', dataset=dataset+ mutation +'_train')
            test_data = TestbedDataset(root='data', dataset=dataset+ mutation +'_test')

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # combined_data = ConcatDataset([train_data, test_data])
    # combined_loader = DataLoader(combined_data, batch_size=BATCH_SIZE, shuffle=False)

    # Load pre-trained model
    model = modeling().to(device)
    model.load_state_dict(torch.load(model_path))

    # Predict for combined dataset
    # G, P = predicting(model, device, combined_loader)
    # Plot histograms for combined dataset
    # plot_histograms(G, P, data_dir = f'images/{model_st}_{dataset}{mutation}_histogram.png')

    # Predict for test dataset
    G, P = predicting(model, device, test_loader)
    
    histogram_dir = 'images/histograms'
    os.makedirs(histogram_dir, exist_ok=True)
    plot_histograms(G, P, data_dir = f'{histogram_dir}/{model_st}_{dataset}{mutation}_histogram.png')

    # Save test predictions       
    output_dir = 'analysis/predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_data_file = os.path.join(output_dir, model_st + '_' + dataset + mutation + '_test_predictions.csv')

    with open(test_data_file, 'r') as infile, open(output_data_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Read the header and add the new column name
        header = next(reader)
        header.append('prediction')
        writer.writerow(header)
        
        # Iterate over the rows and append the predictions
        for i, row in enumerate(reader):
            row.append(P[i])
            writer.writerow(row)
    print("Predictions written to .csv file successfully.")