import argparse
import os
import pickle

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

    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e ]
    test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))

    param_path = f'interpretability/protein_parameters/{dataset}{mutation}_proteins_ProtParam.csv'
    emb_path = f'interpretability/protein_embeddings/test/{dataset}{mutation}_{model_st}_embeddings.csv'

    protparams = pd.read_csv(param_path, usecols=[2,3,4,5,6,7,8]).to_numpy()
    # protparams = protparams_df.iloc[:, [2,3,4,5,6,7,8]].to_numpy()
    # print(np.shape(protparams))
    # print(protparams)

    embeddings_df = pd.read_csv(emb_path, header=None)
    embeddings = np.asarray(embeddings_df)
    # print(np.shape(embeddings))
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

    # affinity = 68 x 442 -> cols = proteins
    # affinity_rows[index_testfold], affinity_cols[index_testfold] -> find protein column number
    # find ID/sequence from column number
    # group embeddings for drugs by sequence/ID
    # average the embeddings 

    # OR open *test.csv from create_data.py that has already done test_fold and continue from there?

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
# figure out what to do with PDD / PDD_Vnoc embeddings -> skip? -> average?

    