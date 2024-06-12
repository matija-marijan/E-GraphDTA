import os
import pandas as pd
import numpy as np
import esm
from rdkit import Chem
import networkx as nx
from utils import *

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

compound_iso_smiles = []
for dt_name in ['kiba','davis']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g


########################### PROTEIN PROCESSING ###########################

# Done in a separate conda environment with TensorFlow
# deepfri_embeddings.py
# ESM_TestbedDataset is appropriate for DeepFRI embeddings, only the dimensions differ (320 vs 4096)

######################## PYTORCH FILE FORMATTING #########################

datasets = ['davis', 'kiba']
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_deepfri_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_deepfri_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
       
        df = pd.read_csv('data/' + dataset + '_train.csv')
        train_drugs, train_Y = list(df['compound_iso_smiles']), list(df['affinity'])
        train_prots = pd.read_csv('data/' + dataset + '_deepfri_train.csv')
        train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
        
        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_drugs, test_Y = list(df['compound_iso_smiles']), list(df['affinity'])
        test_prots = pd.read_csv('data/' + dataset + '_deepfri_test.csv')
        test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)

        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_deepfri_train.pt in pytorch format!')
        train_data = ESM_TestbedDataset(root='data', dataset=dataset+'_deepfri_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph)
        print('preparing ', dataset + '_deepfri_test.pt in pytorch format!')
        test_data = ESM_TestbedDataset(root='data', dataset=dataset+'_deepfri_test', xd=test_drugs, xt=test_prots, y=test_Y,smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')        
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')        
