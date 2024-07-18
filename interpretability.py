import json
from collections import OrderedDict
import pandas as pd
import numpy as np
# TO-DO:
# extract ProtParam - DONE
# extract GO terms
# extract embeddings (save to new file if datadir doesnt exist)
# cross-correlation emb/param
# cross-correlation emb/go -> pay attention to mutations!

datasets = ['davis']#, 'kiba']
for dataset in datasets:

    if dataset == 'davis':
        proteins_dir = 'data/' + dataset + '/new_proteins.json'
    else:
        proteins_dir = 'data/' + dataset + '/proteins.txt'

    # goterms_dir = 'data/' + dataset + '/go_terms.csv'
    protparams_dir = 'data/' + dataset + '/interpretability/proteins_ProtParam.csv'

    # json parser
    proteins = json.load(open(proteins_dir), object_pairs_hook=OrderedDict)

    protparams_df = pd.read_csv(protparams_dir)
    protparams = protparams_df.iloc[:, [2,3,4,5,6,7,8]].to_numpy()




    