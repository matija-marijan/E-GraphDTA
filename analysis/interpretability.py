import argparse
import os
import pickle

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

from models.pdc_ginconv import PDC_GINConvNet
from models.vnoc_ginconv import Vnoc_GINConvNet
from models.pdc_vnoc_ginconv import PDC_Vnoc_GINConvNet
from models.esm_ginconv import ESM_GINConvNet
from models.fri_ginconv import FRI_GINConvNet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
from rdkit import Chem

from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors

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

if __name__ == "__main__":
    print('\nrunning on ', model_st + '_' + dataset )

    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e ]
    test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))

    param_path = f'analysis/interpretability/new_protein_parameters/{dataset}{mutation}_proteins_ProtParam.csv'
    emb_path = f'analysis/interpretability/protein_embeddings/test/{dataset}{mutation}_{model_st}_embeddings.csv'

    protparams_df = pd.read_csv(param_path)
    # print(protparams_df.columns)
    # protparams = pd.read_csv(param_path, usecols=[2,3,4,5,6,7,8]).to_numpy()
    # Exclude 2 AA columns 30 and 31 (no protein has them)
    protparams = protparams_df.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,32,33]].to_numpy()
    print(np.shape(protparams))
    print(protparams)

    embeddings_df = pd.read_csv(emb_path, header=None)
    embeddings = np.asarray(embeddings_df)
    print(np.shape(embeddings))
    # print(embeddings_df.shape)

    # latent_train_mapping = {index: latent_train[i] for i, index in enumerate(train_fold)}
    latent_test_mapping = {index: embeddings[i] for i, index in enumerate(test_fold)}

    if dataset == 'davis' and mutation != '':
        proteins_dir = 'data/' + dataset + '/new_proteins.json'
    else:
        proteins_dir = 'data/' + dataset + '/proteins.txt'
    proteins = json.load(open(proteins_dir), object_pairs_hook=OrderedDict)

    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)
    rows, cols = np.where(np.isnan(affinity)==False) 
    
    rows, cols = rows[test_fold], cols[test_fold]
    
    # rows su sad indeksi od 0 do 67 drugs, a cols su indeksi od 0 do 441 prots

    # print(*rows, sep=' ')
    print(np.max(rows))
    print(np.shape(rows))
    
    # print(*cols, sep=' ')
    print(np.max(cols))
    print(np.shape(cols))

    print(proteins.keys().__len__())
    print(list(proteins.keys())[0])
    # print(list(proteins.keys()))
    
    all_params = np.zeros((embeddings.shape[0], protparams.shape[1]))
    print(np.shape(all_params))

    for i in range(embeddings.shape[0]):
        all_params[i, :] = protparams[cols[i]]
        # print(protparams[cols[i]])
        # print(all_params[i, :])
        # print(i)
        # print(cols[i])
        # exit()
        
    print(np.shape(all_params))

    # affinity = 68 x 442 -> cols = proteins
    # affinity_rows[index_testfold], affinity_cols[index_testfold] -> find protein column number
    # find ID/sequence from column number
    # group embeddings for drugs by sequence/ID
    # average the embeddings 

    # OR open *test.csv from create_data.py that has already done test_fold and continue from there?

    # od 442 x 7 napraviti 5010 x 7
    # procitati test foldove
    # procitati koji je protein
    # naci njegove protparam
    #

    # CCA analysis
    n_components = 2
    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit(embeddings, all_params).transform(embeddings, all_params)
    print(np.shape(X_c))
    print(np.shape(Y_c))

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'CCA1_X': X_c[:, 0],
        'CCA2_X': X_c[:, 1],
        'CCA1_Y': Y_c[:, 0],
        'CCA2_Y': Y_c[:, 1]
    })

    emb_proj_x = df['CCA1_X']
    emb_proj_y = df['CCA2_X']
    param_proj_x = df['CCA1_Y']
    param_proj_y = df['CCA2_Y']
    
    # left singular vectors
    emb_vectors = cca.x_weights_
    # right singular vectors
    param_vectors = cca.y_weights_    

    print()
    print(np.shape(emb_proj_x))
    print(np.shape(emb_proj_y))
    print(np.shape(param_proj_x))
    print(np.shape(param_proj_y))
    print(np.shape(cca.x_weights_))
    print(np.shape(cca.y_weights_))

    plt.figure(figsize=(10, 10))

    # color1 = mcolors.TABLEAU_COLORS['tab:color1']
    # color2 = mcolors.TABLEAU_COLORS['tab:color2']
    
    color1 = "#1f77b4"
    color2 = "#d62728"
    color3 = "black"

    # Plot the latent variable projections
    plt.plot(emb_proj_x, emb_proj_y, '.', color=color1, alpha=1, label='Protein embedding projections', zorder = 1)

    # Plot the latent variable singular vectors
    plt.plot(emb_vectors[:, 0], emb_vectors[:, 1], '^', color=color2, alpha=1, label='Protein embedding singular vectors', zorder=2)
    
    # Calculate the Euclidean distance from (0, 0) for each point
    distances = np.sqrt(emb_vectors[:, 0]**2 + emb_vectors[:, 1]**2)

    # Find the indices of the three farthest points
    farthest_indices = np.argsort(distances)[-3:]

    # Circle the two farthest points
    for idx in farthest_indices:
        plt.scatter(emb_vectors[idx, 0], emb_vectors[idx, 1], s=200, facecolors='none', edgecolors=color3, linewidths=2, zorder=3)
        # text = plt.text(emb_vectors[idx, 0] + 0.025, emb_vectors[idx, 1] + 0.025, str(f'LV-{idx}'), color=color2, fontsize=10, fontweight='bold', zorder=5)
        # text.set_path_effects([path_effects.Stroke(linewidth=1, foreground=color3), path_effects.Normal()])

    # Add arrows for protein descriptors
    for i in range(all_params.shape[1]):
        arrows = plt.arrow(0, 0, param_vectors[i, 0], param_vectors[i, 1], color=color3, alpha=1, linewidth=1.5, head_width=0.015, head_length=0.0125, zorder = 4)
        # arrows.set_path_effects([path_effects.Stroke(linewidth=2, foreground=color3), path_effects.Normal()])

    # Calculate the Euclidean distance from (0, 0) for each arrow endpoint
    distances = np.sqrt(param_vectors[:, 0]**2 + param_vectors[:, 1]**2)

    # Find the indices of the three/four farthest arrows
    farthest_indices = np.argsort(distances)[-3:]

    # Circle the two farthest arrows and add text with their indices
    for i in farthest_indices:
        plt.scatter(param_vectors[i, 0], param_vectors[i, 1], s=200, facecolors='none', edgecolors=color2, linewidths=2, zorder=4)
        # text = plt.text(param_vectors[i, 0] + 0.025, param_vectors[i, 1] + 0.025, protparams_df.columns[i + 2], color=color3, fontsize=10, fontweight='bold', zorder=5)
        # text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])

    # Create a custom legend for descriptors
    handles = [
        plt.Line2D([0], [0], marker='.', color='w', markerfacecolor=color1, markersize=10, label='Protein embedding projections'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=color2, markersize=10, label='Protein embedding singular vectors'),
        plt.Line2D([0], [0], color=color3, lw=2, label='Protein parameter singular vectors'),
    ]

    plt.xlabel('CCA1', fontsize=12)
    plt.ylabel('CCA2', fontsize=12)
    plt.title(f'{model_st} Redundancy Analysis Triplot on the {dataset} dataset', fontsize=14)
    plt.legend(handles=handles, loc='upper right', fontsize=12)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.show()
    plt.savefig(f'analysis/interpretability/{model_st}_{dataset}{mutation}_CCA.png', dpi=300, bbox_inches='tight')
    # plt.savefig(f'analysis/interpretability/{model_st}_{dataset}{mutation}_CCA_annot.png', dpi=100)


# TO-DO:
# extract ProtParam - DONE
# extract GO terms - standby
# read .pt from from create_data to predict done?
# extract embeddings (save to new file if datadir doesnt exist) done
# extract from specific layer done
# read embedding matrix
# read prot param matrix
# cross-correlation emb/param
# cross-correlation emb/go -> pay attention to mutations!
# figure out what to do with PDC / PDC_Vnoc embeddings -> skip? -> average?

    