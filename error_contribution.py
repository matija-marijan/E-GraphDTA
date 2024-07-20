import argparse

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

def calculate_mae(group):
    mae = (group['affinity'] - group['prediction']).abs().median()
    return mae

# Function to plot MAE
def plot_mae(df, x_col, y_col, title, xlabel, ylabel, highlight_points, ax):
    # sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, s=10)
    ax.scatter(df[x_col], df[y_col], s=10)
    for point in highlight_points:
        x = df[df[x_col] == point][x_col].values[0]
        y = df[df[x_col] == point][y_col].values[0]
        ax.annotate(point, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

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

    data_file = 'predictions/' + model_st + '_' + dataset + mutation + '_test_predictions.csv'
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
    with open(f'predictions/annotations/{model_st}_{dataset}{mutation}_drugs.json', 'w') as f:
        json.dump(top_10_drug_mae, f, indent=4)

    top_10_prot_mae = protein_mae.tail(10)
    top_10_prot_mae = top_10_prot_mae['target_sequence'].tolist()
    with open(f'predictions/annotations/{model_st}_{dataset}{mutation}_proteins.json', 'w') as f:
        json.dump(top_10_prot_mae, f, indent=4)

    # Create the plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # # Plot for drug_mae_sorted
    # highlight_drugs = drug_mae.head(10)['compound_iso_smiles'].tolist()  # Highlight top 10 drugs with highest MAE
    # plot_mae(drug_mae, 'compound_iso_smiles', 'mae', 'Prediction Error for Drugs', 'Drug', 'Median of Absolute Errors', highlight_drugs, axs[0])

    # # Plot for protein_mae_sorted
    # highlight_proteins = protein_mae.head(10)['target_sequence'].tolist()  # Highlight top 10 proteins with highest MAE
    # plot_mae(protein_mae, 'target_sequence', 'mae', 'Prediction Error for Proteins', 'Protein', 'Median of Absolute Errors', highlight_proteins, axs[1])

    # # Plot for Drug MAE
    # axs[0].scatter(range(len(drug_mae)), drug_mae['mae'], s=10)
    # # for i in range(-10, 0):  # Annotate only the top 10 highest MAE
    # #     axs[0].annotate(drug_mae['compound_iso_smiles'].iloc[i], (len(drug_mae) + i, drug_mae['mae'].iloc[i]), fontsize=8,
    # #                     xytext=(-10, 0), textcoords = 'offset points', ha='right')
    # axs[0].set_xlabel('Drug')
    # axs[0].set_ylabel('Median of Absolute Errors for Affinity Prediction')
    # # axs[0].set_title(f'{model_st} Prediction Error for {dataset}{mutation} Test Data')
    # axs[0].set_xticks([])

    # # Plot for Protein MAE
    # axs[1].scatter(range(len(protein_mae)), protein_mae['mae'], s=10)
    # # for i in range(-10, 0):  # Annotate only the top 10 highest MAE
    # #     axs[1].annotate(protein_mae['target_sequence'].iloc[i], (len(protein_mae) + i, protein_mae['mae'].iloc[i]), fontsize=8,
    # #                     xytext=(-10, 0), textcoords = 'offset points', ha='right')
    # axs[1].set_xlabel('Protein')
    # # axs[1].set_ylabel('Median of Absolute Errors for Affinity Prediction')
    # # axs[1].set_title(f'{model_st} Prediction Error for {dataset}{mutation} Test Data')
    # axs[1].set_xticks([])
    
    # plt.suptitle(f'{model_st} Prediction Error for {dataset}{mutation} Test Data')
    # plt.tight_layout()
    # plt.show()

# TO-DO:
# load model - done
# load dataset - done (implicitly?)
# predict(dataset) - done (for combined data only!)
# Optional: histogram, output analysis (confusion matrix?)
# Optional: number of parameters, inference time?
# extract embeddings!!! (keract?)
# find out which embedding corresponds to which protein!
# save embeddings to davis_442x128.csv and kiba_223x128.csv