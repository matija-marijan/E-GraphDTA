import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import json
from deepfrier.Predictor import Predictor
import pandas as pd
import numpy as np

path = '/home/matijamarijan/projects/GraphDTA/'
model_config = '/home/matijamarijan/projects/GraphDTA/trained_models/model_config.json'
ont = 'cc'
emb_layer = 'global_max_pooling1d'

with open(model_config) as json_file:
    params = json.load(json_file)

params = params['cnn']
gcn = params['gcn']
models = params['models']
predictor = Predictor(models[ont], gcn = gcn)

datasets = ['davis', 'kiba']
for dataset in datasets:
    processed_proteins_train = path + 'data/' + dataset + '_deepfri_train.csv'
    processed_proteins_test = path + 'data/' + dataset + '_deepfri_test.csv'
    if ((not os.path.isfile(processed_proteins_train)) or (not os.path.isfile(processed_proteins_test))):

        predictor = Predictor(models[ont], gcn=gcn)
        
        df = pd.read_csv(path + 'data/' + dataset + '_train.csv')
        train_prots = list(df['target_sequence'])
        embeddings = []
        # DeepFRI protein representation
        for i in range(0, len(train_prots)):
            prot = train_prots[i]
            print('iteration ' + str(i + 1) + ' of ' + str(len(train_prots)))
            emb = predictor.predict_embeddings(prot, layer_name = emb_layer)
            embeddings.append(emb)
        print('train embeddings done')
        train_prots = np.asarray(embeddings)

        df = pd.read_csv(path + 'data/' + dataset + '_test.csv')
        test_prots = list(df['target_sequence'])
        embeddings = []
        # DeepFRI protein representation
        for i in range(0, len(test_prots)):
            prot = test_prots[i]
            print('iteration ' + str(i + 1) + ' of ' + str(len(test_prots)))
            emb = predictor.predict_embeddings(prot, layer_name = emb_layer)
            embeddings.append(emb)
        print('test embeddings done')      
        test_prots = np.asarray(embeddings)

        print(np.shape(train_prots))
        print(np.shape(test_prots))
        df_train_prots = pd.DataFrame(train_prots)
        df_test_prots = pd.DataFrame(test_prots)

        df_train_prots.to_csv(processed_proteins_train, index = False)
        df_test_prots.to_csv(processed_proteins_test, index = False)
        
        print(processed_proteins_train, ' and ', processed_proteins_test, ' have been created')        
    else:
        print(processed_proteins_train, ' and ', processed_proteins_test, ' are already created')