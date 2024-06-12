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

datasets = ['davis', 'kiba']
for dataset in datasets:

    if dataset == 'davis':
        batch_size = 4
    elif dataset == 'kiba':
        batch_size = 1
    else:
        batch_size = 1

    processed_proteins_train = 'data/' + dataset + '_esm_train.csv'
    processed_proteins_test = 'data/' + dataset + '_esm_test.csv'
    if ((not os.path.isfile(processed_proteins_train)) or (not os.path.isfile(processed_proteins_test))):
        
        df = pd.read_csv('data/' + dataset + '_train.csv')
        train_prots = list(df['target_sequence'])
        
        # ESM protein representation
        labels = []
        for i in range(0, len(train_prots)):
            labels.append('protein' + str(i + 1))
        train_prots = list(zip(labels, train_prots))

        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        if torch.cuda.is_available():
            model = model.cuda()
        batch_converter = alphabet.get_batch_converter()
        embeddings = []

        for i in range(0, len(train_prots), batch_size):
            print('iteration ' + str(i // batch_size + 1) + ' of ' + str(len(train_prots) // batch_size + 1))
            batch_prots = train_prots[i : i + batch_size]
                
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_prots)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            with torch.no_grad():
                if torch.cuda.is_available():
                    results = model(batch_tokens.cuda(), repr_layers=[6])
                else:
                    results = model(batch_tokens, repr_layers = [6])
            token_representations = results["representations"][6]

            sequence_representations = []
            for j, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[j, 1 : tokens_len - 1].mean(0).cpu())

            for j in range(0, len(sequence_representations)):
                embeddings.append(sequence_representations[j])
        print('train embedding done')
        
        train_prots = np.asarray(embeddings)
        
        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_prots = list(df['target_sequence'])

        # ESM protein representation
        labels = []
        for i in range(0, len(test_prots)):
            labels.append('protein' + str(i + 1))
        test_prots = list(zip(labels, test_prots))

        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        if torch.cuda.is_available():
            model = model.cuda()
        batch_converter = alphabet.get_batch_converter()
        embeddings = []

        for i in range(0, len(test_prots), batch_size):
            print('iteration ' + str(i // batch_size + 1) + ' of ' + str(len(test_prots) // batch_size + 1))
            batch_prots = test_prots[i : i + batch_size]
                
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_prots)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            with torch.no_grad():
                if torch.cuda.is_available():
                    results = model(batch_tokens.cuda(), repr_layers=[6])
                else:
                    results = model(batch_tokens, repr_layers = [6])
            token_representations = results["representations"][6]

            sequence_representations = []
            for j, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[j, 1 : tokens_len - 1].mean(0).cpu())

            for j in range(0, len(sequence_representations)):
                embeddings.append(sequence_representations[j])
        print('test embedding done')
            
        test_prots = np.asarray(embeddings)

        df_train_prots = pd.DataFrame(train_prots)
        df_test_prots = pd.DataFrame(test_prots)

        df_train_prots.to_csv(processed_proteins_train, index = False)
        df_test_prots.to_csv(processed_proteins_test, index = False)

        print(processed_proteins_train, ' and ', processed_proteins_test, ' have been created')        
    else:
        print(processed_proteins_train, ' and ', processed_proteins_test, ' are already created')


######################## PYTORCH FILE FORMATTING #########################

datasets = ['davis', 'kiba']
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_esm_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_esm_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
       
        df = pd.read_csv('data/' + dataset + '_train.csv')
        train_drugs, train_Y = list(df['compound_iso_smiles']), list(df['affinity'])
        train_prots = pd.read_csv('data/' + dataset + '_esm_train.csv')
        train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
        
        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_drugs, test_Y = list(df['compound_iso_smiles']), list(df['affinity'])
        test_prots = pd.read_csv('data/' + dataset + '_esm_test.csv')
        test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)

        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_esm_train.pt in pytorch format!')
        train_data = ESM_TestbedDataset(root='data', dataset=dataset+'_esm_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph)
        print('preparing ', dataset + '_esm_test.pt in pytorch format!')
        test_data = ESM_TestbedDataset(root='data', dataset=dataset+'_esm_test', xd=test_drugs, xt=test_prots, y=test_Y,smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')        
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')        
