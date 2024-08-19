import argparse
import os

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
from rdkit import Chem

def find_matching_keys(file1_path, file2_path, canonicalize=False):
    """
    Finds keys in the second JSON file that have identical values to those in the first JSON file.
    
    Parameters:
        file1_path (str): Path to the first JSON file.
        file2_path (str): Path to the second JSON file.
    
    Returns:
        dict: A dictionary where keys from the first JSON file map to lists of keys from the second JSON file
              that have the same values.
    """
    # Load JSON files
    with open(file1_path, 'r') as f1:
        data1 = json.load(f1)

    data2 = json.load(open(file2_path), object_pairs_hook=OrderedDict) 

    if canonicalize==True:
        data2 = {key: Chem.MolToSmiles(Chem.MolFromSmiles(value)) for key, value in data2.items()}

    # Create a mapping of values to keys for the second JSON file
    value_to_keys = {}
    for key, value in data2.items():
        if value not in value_to_keys:
            value_to_keys[value] = []
        value_to_keys[value].append(key)

    # Create a result dictionary where keys from the second JSON file map to lists of values from the first JSON file
    result = {}
    for key, value in data2.items():
        if value in data1:
            if key not in result:
                result[key] = []
            result[key].append(value)

    return result

def calculate_mae(group):
    mae = (group['affinity'] - group['prediction']).abs().median()
    return mae

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

if __name__ == "__main__":
    print('\nrunning on ', model_st + '_' + dataset )

    data_file = 'analysis/predictions/' + model_st + '_' + dataset + mutation + '_test_predictions.csv'
    with open(data_file, 'r') as infile:
        results_df = pd.read_csv(infile)

    ground = np.asarray(results_df['affinity'])
    prediction = np.asarray(results_df['prediction'])

    print(ground)
    print(prediction)

    # Calculate MAE for each drug
    drug_mae = results_df.groupby('compound_iso_smiles').apply(calculate_mae).reset_index()
    drug_mae.columns = ['compound_iso_smiles', 'mae']
    drug_mae = drug_mae.sort_values(by='mae')

    # Calculate MAE for each protein
    protein_mae = results_df.groupby('target_sequence').apply(calculate_mae).reset_index()
    protein_mae.columns = ['target_sequence', 'mae']
    protein_mae = protein_mae.sort_values(by='mae')

    top_10_drug_mae = drug_mae.tail(10)
    top_10_drug_mae = top_10_drug_mae['compound_iso_smiles'].tolist()
    # if ((not os.path.isfile(f'analysis/predictions/annotations/{model_st}_{dataset}{mutation}_drugs.json'))):
    with open(f'analysis/predictions/annotations/{model_st}_{dataset}{mutation}_drugs.json', 'w') as f:
        json.dump(top_10_drug_mae, f, indent=4)

    top_10_prot_mae = protein_mae.tail(10)
    top_10_prot_mae = top_10_prot_mae['target_sequence'].tolist()
    # if ((not os.path.isfile(f'analysis/predictions/annotations/{model_st}_{dataset}{mutation}_proteins.json'))):
    with open(f'analysis/predictions/annotations/{model_st}_{dataset}{mutation}_proteins.json', 'w') as f:
        json.dump(top_10_prot_mae, f, indent=4)

    if mutation != '':
        proteins = f'data/{dataset}/new_proteins.json'
    else:
        proteins = f'data/{dataset}/proteins.txt'
    
    drugs = f'data/{dataset}/ligands_can.txt'

    annotations_drug = f'analysis/predictions/annotations/{model_st}_{dataset}{mutation}_drugs.json'
    annotations_prot = f'analysis/predictions/annotations/{model_st}_{dataset}{mutation}_proteins.json'

    matching_keys_drugs = find_matching_keys(annotations_drug, drugs, canonicalize=True)
    print(matching_keys_drugs.keys())
    drug_keys = list(matching_keys_drugs.keys())
    # with open(f'images/error contribution/{model_st}_{dataset}{mutation}_drugs.json', 'w') as f:
    #     json.dump(drug_keys, f)

    matching_keys_prots = find_matching_keys(annotations_prot, proteins)
    print(matching_keys_prots.keys())
    prot_keys = list(matching_keys_prots.keys())
    # with open(f'images/error contribution/{model_st}_{dataset}{mutation}_proteins.json', 'w') as f:
    #     json.dump(prot_keys, f)

    # Create the plot
    fig, axs = (plt.subplots(1, 2, figsize=(14, 6)))

    # Plot for Drug MAE
    axs[0].scatter(range(len(drug_mae)), drug_mae['mae'], s=10, color = 'black')
    for i in range(-10, 0):  # Annotate only the top 10 highest MAE
        axs[0].annotate(drug_keys[-i - 1], (len(drug_mae) + i, drug_mae['mae'].iloc[i]), fontsize=8,
                        xytext=(-10, 0), textcoords = 'offset points', ha='right')
    axs[0].set_xlabel('Drug')
    axs[0].set_ylabel('Median of Absolute Errors for Affinity Prediction')
    # axs[0].set_title(f'{model_st} Prediction Error for {dataset}{mutation} Test Data')
    axs[0].set_xticks([])

    # Plot for Protein MAE
    axs[1].scatter(range(len(protein_mae)), protein_mae['mae'], s=10, color='black')
    for i in range(-10, 0):  # Annotate only the top 10 highest MAE
        axs[1].annotate(prot_keys[-i - 1], (len(protein_mae) + i, protein_mae['mae'].iloc[i]), fontsize=8,
                        xytext=(-10, 0), textcoords = 'offset points', ha='right')
    axs[1].set_xlabel('Protein')
    # axs[1].set_ylabel('Median of Absolute Errors for Affinity Prediction')
    # axs[1].set_title(f'{model_st} Prediction Error for {dataset}{mutation} Test Data')
    axs[1].set_xticks([])
    
    plt.suptitle(f'{model_st} Prediction Error for {dataset}{mutation} Test Data')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'images/error contribution/{model_st}_{dataset}{mutation}_errors.png', dpi=500)

# TO-DO:
# load model - done
# load dataset - done (implicitly?)
# predict(dataset) - done (for combined data only!)
# Optional: histogram, output analysis (confusion matrix?)
# Optional: number of parameters, inference time?
# extract embeddings!!! (keract?)
# find out which embedding corresponds to which protein!
# save embeddings to davis_442x128.csv and kiba_223x128.csv