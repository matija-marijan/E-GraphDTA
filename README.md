# P-GraphDTA: Enhanced protein representations improve drug–target binding affinity prediction
Implementation of five novel protein representation models on top of GraphDTA's drug graph neural network processing methods for predicting the binding affinity of drug-target pairs. The proposed methods revolve around three key concepts of protein sequence representation and interpretation:
1. Incorporating information about the drug into the target representation
2. Redefining protein embedding convolution layers
3. Predicting protein representations using Large Language Models, which include [Evolutionary Scale Modeling](https://github.com/facebookresearch/esm) and [Functional Residue Identification](https://github.com/flatironinstitute/DeepFRI).

<p align="center">
<img src="images/pipeline.png" width="800">
</p> 

# Installation
All environments and source codes were created and tested in a Linux environment.
## GraphDTA environment
```
conda create -n geometric python=3
conda activate geometric
conda install -y -c conda-forge rdkit
conda install pytorch torchvision -c pytorch
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cu118.html
pip install torch-geometric
conda install cudatoolkit -c conda-forge
pip install fair-esm
```
+ Additionally, this repository contains ```requirements.txt``` and ```environment.yml``` files for environment creation. These requirements can be installed using ```pip install requirements.txt``` or ```conda env create -f environment.yml```.
## DeepFRI environment
+ The preprocessing steps required to run and extract embeddings from DeepFRI models run in a different and incompatible environment to ```geometric```.
+ preprocessing/FRI/ contains ```fri_requirements.txt``` and ```fri_environment.yml``` files for DeepFRI environment creation. These requirements can be installed using ```pip install fri_requirements.txt``` or ```conda env create -f fri_environment.yml```, from ```cd preprocessing/FRI```. Additionally, the DeepFRI environment can be installed using ```pip install .``` from ```cd preprocessing/FRI```.

# Resources:

## Datasets
+ data/davis/\*, data/kiba/\* contain the Davis and KIBA drug-target interaction datasets. These file were downloaded from [DeepDTA](https://github.com/hkmztrk/DeepDTA/tree/master/data).
+ data/davis/new_proteins.json contains the updated version of the proteins from the Davis dataset, which have been accounted for mutations. This file was created with data downloaded from [DTITR](https://github.com/larngroup/DTITR/blob/main/data/davis/dataset/davis_dataset_processed.csv).

## Proposed Models:
All the proposed models, along with GraphDTA models can be found in the models/ folder.
+  

## Large Language Models
+ The ESM model for extracting protein embeddings can be downloaded from [download link](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt), and should be moved to preprocessing/ESM/.
+ The DeepFRI models can be downloaded from [download link](https://users.flatironinstitute.org/~renfrew/DeepFRI_data/trained_models.tar.gz). The ```tar.gz``` file can be uncompressed into the preprocessing/FRI/ directory by using ```tar xvzf trained_models.tar.gz -C /path/to/GraphDTA/preprocessing/FRI```. The preprocessing/FRI/deepfrier folder contains the ```deepfrier``` modified source code from [DeepFRI](https://github.com/flatironinstitute/DeepFRI/tree/master/deepfrier), which is used for extracting embeddings during inference to create new datasets for training the updated GraphDTA models.

# Usage

## 1. Preprocessing ! SREDITI -x mutation flag! i cuvanje podataka!
+ ```python create_data.py``` - Create original data in pytorch format
+ python -m preprocessing.ESM.esm_preprocessing.py
+ python -m preprocessing.FRI.fri

## 2. Training
A prediction model can be trained using ```python training.py``` with the following arguments:
1. --model/-m:
2. --dataset/-d:
3. --cuda/-c:
4. --seed/-s:
5. --mutation/-x:
The training script saves the best overall model checkpoint with the lowest testing MSE, and continually outputs MSE, RMSE, Spearman correlation, and Pearson correlation values. It also calculates the Concordance Index for the best overall model at the end of training.
Example use:
```
python training.py -d davis -m PDD_Vnoc_GINConvNet -s 0 -x 1
```


### Source codes:
+ create_data.py: create data in pytorch format
+ utils.py: include TestbedDataset used by create_data.py to create data, and performance measures.
+ training.py: train a GraphDTA model.
+ models/ginconv.py, gat.py, gat_gcn.py, and gcn.py: proposed models GINConvNet, GATNet, GAT_GCN, and GCNNet receiving graphs as input for drugs.

### Preprocessing:

### Post-hoc analysis:

# Step-by-step running:

## 1. Create data in pytorch format
Running
```sh
conda activate geometric
python create_data.py
```
This returns kiba_train.csv, kiba_test.csv, davis_train.csv, and davis_test.csv, saved in data/ folder. These files are in turn input to create data in pytorch format,
stored at data/processed/, consisting of  kiba_train.pt, kiba_test.pt, davis_train.pt, and davis_test.pt.

## 2. Train a prediction model
To train a model using training data. The model is chosen if it gains the best MSE for testing data.  
Running 

```sh
conda activate geometric
python training.py 0 0 0
```

where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively;
 the second argument is for the index of the models, 0/1/2/3 for GINConvNet, GATNet, GAT_GCN, or GCNNet, respectively;
 and the third argument is for the index of the cuda, 0/1 for 'cuda:0' or 'cuda:1', respectively. 
 Note that your actual CUDA name may vary from these, so please change the following code accordingly:
```sh
cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
```

This returns the model and result files for the modelling achieving the best MSE for testing data throughout the training.
For example, it returns two files model_GATNet_davis.model and result_GATNet_davis.csv when running GATNet on Davis data.

## 3. Train a prediction model with validation 

In "3. Train a prediction model", a model is trained on training data and chosen when it gains the best MSE for testing data.
This follows how a model was chosen in https://github.com/hkmztrk/DeepDTA. The result by two ways of training is comparable though.

In this section, a model is trained on 80% of training data and chosen if it gains the best MSE for validation data, 
which is 20% of training data. Then the model is used to predict affinity for testing data.

Same arguments as in "3. Train a prediction model" are used. E.g., running 

```sh
python training_validation.py 0 0 0
```

This returns the model achieving the best MSE for validation data throughout the training and performance results of the model on testing data.
For example, it returns two files model_GATNet_davis.model and result_GATNet_davis.csv when running GATNet on Davis data.
