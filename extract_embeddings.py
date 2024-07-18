import torch
from torch.utils.data import ConcatDataset
import argparse
from utils import *
import matplotlib.pyplot as plt

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

from models.pdd_ginconv import PDD_GINConvNet
from models.vnoc_ginconv import Vnoc_GINConvNet
from models.pdd_vnoc_ginconv import PDD_Vnoc_GINConvNet
from models.esm_ginconv import ESM_GINConvNet
from models.fri_ginconv import FRI_GINConvNet

from models.flag.flag_ginconv import Flag_GINConvNet
from models.flag.flag_pdd_ginconv import Flag_PDD_GINConvNet
from models.flag.flag_vnoc_ginconv import Flag_Vnoc_GINConvNet
from models.flag.flag_pdd_vnoc_ginconv import Flag_PDD_Vnoc_GINConvNet
from models.flag.flag_esm_ginconv import Flag_ESM_GINConvNet
from models.flag.flag_fri_ginconv import Flag_FRI_GINConvNet

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

flag_models = {
    'Flag_GINConvNet': Flag_GINConvNet, 
    'Flag_PDD_GINConvNet': Flag_PDD_GINConvNet, 
    'Flag_Vnoc_GINConvNet': Flag_Vnoc_GINConvNet, 
    'Flag_ESM_GINConvNet': Flag_ESM_GINConvNet, 
    'Flag_FRI_GINConvNet': Flag_FRI_GINConvNet, 
    'Flag_PDD_Vnoc_GINConvNet': Flag_PDD_Vnoc_GINConvNet
}

parser = argparse.ArgumentParser(description="Run a specific model on a specific dataset.")

parser.add_argument('-d', '--dataset', type=str, choices=datasets, required=True, 
                    help="Dataset name: 'davis' or 'kiba'.")
parser.add_argument('-m', '--model', type=str, choices=list(all_models.keys()), required=True, 
                    help="Model name. Choose from: " + ", ".join(all_models.keys()) + ".")
parser.add_argument('-c', '--cuda', type=int, default=0, 
                    help="CUDA device index (default: 0).")
parser.add_argument('-x', '--mutation', type=int, default = 0, choices = {0, 1, 2},
                    help="Flag for including protein sequence mutations (1), and protein phosphorylation flags (2) (default: 0).")

args = parser.parse_args()

modeling = all_models[args.model]
model_st = modeling.__name__

dataset = args.dataset
mutation = ''
if dataset == 'davis':
    if args.mutation == 0:
        mutation = ''
    elif args.mutation == 1:
        mutation = '_mutation'
    elif args.mutation == 2:
        mutation = '_flag'
        modeling = flag_models['Flag_' + args.model]
        model_st = modeling.__name__

print(f"Mutation = {args.mutation}")
# Select CUDA device if applicable
cuda_name = f"cuda:{args.cuda}"
print('cuda_name:', cuda_name)
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

# Dodati mogucnost dodavanja argumenta za model_path!
if mutation == '':
    model_path = 'results/final_training_model_' + model_st + '_' + dataset + '.model'
elif mutation == '_mutation':
    model_path = 'results/mutation_training_model_' + model_st + '_' + dataset + '.model'

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

    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        if model_st == "ESM_GINConvNet":
            train_data = TestbedDataset(root='data', dataset=dataset+ mutation +'_esm_train')
            test_data = TestbedDataset(root='data', dataset=dataset+ mutation +'_esm_test')
        elif model_st == "FRI_GINConvNet":
            train_data = ESM_TestbedDataset(root='data', dataset=dataset+ mutation +'_deepfri_train')
            test_data = ESM_TestbedDataset(root='data', dataset=dataset+ mutation +'_deepfri_test')
        else:
            train_data = TestbedDataset(root='data', dataset=dataset+ mutation +'_train')
            test_data = TestbedDataset(root='data', dataset=dataset+ mutation +'_test')

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    combined_data = ConcatDataset([train_data, test_data])
    combined_loader = DataLoader(combined_data, batch_size=BATCH_SIZE, shuffle=False)

    # Load pre-trained model
    model = modeling().to(device)
    model.load_state_dict(torch.load(model_path))
    print(model)

    # Find the layer just before fc1 and register a hook
    embeddings = []
    target_layer_name = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name == 'fc1':
            target_layer_name = previous_layer_name
            break
        previous_layer_name = name

    if target_layer_name is None:
        raise ValueError("Could not find the layer before fc1")
    print(target_layer_name)

    # Define a hook function
    def hook(module, input, output):
        embeddings.append(output.detach().cpu())

    # Register the hook to the target layer
    hook_handle = getattr(model, target_layer_name).register_forward_hook(hook)

    # Predict for combined dataset to extract embeddings
    with torch.no_grad():
        for data in combined_loader:
            data = data.to(device)
            _ = model(data)

    # Remove the hook
    hook_handle.remove()

    # Save embeddings to a CSV file
    embeddings_np = torch.cat(embeddings).numpy()
    np.savetxt(f'data/interpretability/{dataset}{mutation}_{model_st}_embeddings.csv', embeddings_np, delimiter=',')

# TO-DO:
# load model - done
# load dataset - done (implicitly?)
# predict(dataset) - done (for combined data only!)
# Optional: histogram, output analysis (confusion matrix?)
# Optional: number of parameters, inference time?
# extract embeddings!!! (keract?)
# find out which embedding corresponds to which protein!
# save embeddings to davis_442x128.csv and kiba_223x128.csv