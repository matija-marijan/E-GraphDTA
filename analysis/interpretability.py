import argparse
import os
import pickle

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
    'GINConvNet',
    'GATNet',
    'GAT_GCN',
    'GCNNet',
    'PDC_GINConvNet', 
    'Vnoc_GINConvNet', 
    'ESM_GINConvNet', 
    'FRI_GINConvNet', 
    'PDC_Vnoc_GINConvNet'
}

parser = argparse.ArgumentParser(description="Run a specific model on a specific dataset.")

parser.add_argument('-d', '--dataset', type=str, choices=datasets, required=True, 
                    help="Dataset name: 'davis' or 'kiba'.")
parser.add_argument('-m', '--model', type=str, choices=all_models, required=True, 
                    help="Model name. Choose from: " + ", ".join(all_models) + ".")
parser.add_argument('-x', '--mutation', action='store_true', default=False,
                    help="Flag for including protein sequence mutations for the Davis dataset (default: False).")
parser.add_argument('-a', '--average', action='store_true', default=False,
                    help="Flag to average the embeddings (default: False).")
parser.add_argument('-p', '--parameter', type=str, default=None,
                    help="Protein parameter to analyze (optional). If not provided, no correlation analysis will be performed.")
parser.add_argument('-e', '--embedding', type=str, default=None,
                    help="Specific embedding to analyze (optional). If not provided, the embedding with the highest correlation with the chosen protein parameter will be used.")
parser.add_argument('-s', '--save', action='store_true', default=False,
                    help="Flag to save the plots (default: False).")
parser.add_argument('-t', '--text', action='store_true', default=False,
                    help="Flag to include text annotations in the plots (default: False).")

args = parser.parse_args()
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

model_st = args.model

dataset = args.dataset
mutation = ''
if dataset == 'davis' and args.mutation:
    mutation = '_mutation'

if __name__ == "__main__":

    data_path = 'data/' + dataset + '/'
    
    param_path = f'analysis/interpretability/protein_parameters/{dataset}{mutation}_proteins_ProtParam.csv'
    emb_path = f'analysis/interpretability/protein_embeddings/{dataset}{mutation}_{model_st}_embeddings.csv'
    test_csv_path = f'data/{dataset}{mutation}_test.csv'

    params_df = pd.read_csv(param_path)
    params_df = params_df.drop(params_df.columns[[30, 31]], axis=1)
    params_df = params_df.drop_duplicates(subset=['Sequence'])
    print("\nParams shape:", params_df.shape)

    embeddings_df = pd.read_csv(emb_path, header=None)
    print("Embeddings shape:", embeddings_df.shape)

    test_df = pd.read_csv(test_csv_path)
    sequences = test_df['target_sequence']

    # Combine target_sequence and embeddings into a single DataFrame
    embeddings_with_seq = pd.concat([sequences.reset_index(drop=True), embeddings_df.reset_index(drop=True)], axis=1)

    # Set column names: first column is 'target_sequence', rest are 'LV-i'
    embeddings_with_seq.columns = ['target_sequence'] + [f'LV-{i}' for i in range(embeddings_df.shape[1])]

    if args.average:
        # Group by target_sequence and calculate mean embedding for each unique sequence
        mean_embeddings = embeddings_with_seq.groupby('target_sequence').mean().reset_index()

        print(f"\nMean embeddings shape: {mean_embeddings.shape}")

        merged_df = pd.merge(params_df, mean_embeddings, left_on='Sequence', right_on='target_sequence', how='inner')
        print(f"Merged DataFrame shape: {merged_df.shape}")
        # print(merged_df.head())

    else:
        merged_df = pd.merge(params_df, embeddings_with_seq, left_on='Sequence', right_on = 'target_sequence', how='inner')
        print(f"\nMerged DataFrame shape: {merged_df.shape}")
        # print(merged_df.head())

# Split merged_df into 442x30 (parameter columns) and 442x128 (embedding columns), excluding sequence columns
param_cols = [col for col in merged_df.columns if col not in ['Sequence', 'GeneID'] and not col.startswith('LV-') and col != 'target_sequence']
embedding_cols = [f'LV-{i}' for i in range(embeddings_df.shape[1])]

params = merged_df[param_cols].to_numpy()
embeddings = merged_df[embedding_cols].to_numpy()   

print(f"\nParameters array shape: {params.shape}")
print(f"Embeddings array shape: {embeddings.shape}")      

# CCA analysis
n_components = 2
cca = CCA(n_components=n_components)
X_c, Y_c = cca.fit(embeddings, params).transform(embeddings, params)

cca_df = pd.DataFrame({
    'CCA1_X': X_c[:, 0],
    'CCA2_X': X_c[:, 1],
    'CCA1_Y': Y_c[:, 0],
    'CCA2_Y': Y_c[:, 1]
})

emb_proj_x = cca_df['CCA1_X']
emb_proj_y = cca_df['CCA2_X']
param_proj_x = cca_df['CCA1_Y']
param_proj_y = cca_df['CCA2_Y']

# left singular vectors
emb_vectors = cca.x_weights_
# right singular vectors
param_vectors = cca.y_weights_ 

print("\nemb_proj_x shape:", emb_proj_x.shape)
print("emb_proj_y shape:", emb_proj_y.shape)
print("param_proj_x shape:", param_proj_x.shape)
print("param_proj_y shape:", param_proj_y.shape)
print("cca.x_weights_ shape:", cca.x_weights_.shape)
print("cca.y_weights_ shape:", cca.y_weights_.shape)

plt.figure(figsize=(10, 10))
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
    if args.text:
        # Add text annotations for the farthest points
        text = plt.text(emb_vectors[idx, 0] + 0.025, emb_vectors[idx, 1] + 0.025, str(f'LV-{idx}'), color=color2, fontsize=10, fontweight='bold', zorder=5)
        # text.set_path_effects([path_effects.Stroke(linewidth=1, foreground=color3), path_effects.Normal()])

# Add arrows for protein descriptors
for i in range(params.shape[1]):
    arrows = plt.arrow(0, 0, param_vectors[i, 0], param_vectors[i, 1], color=color3, alpha=1, linewidth=1.5, head_width=0.015, head_length=0.0125, zorder = 4)
    # arrows.set_path_effects([path_effects.Stroke(linewidth=2, foreground=color3), path_effects.Normal()])

# Calculate the Euclidean distance from (0, 0) for each arrow endpoint
distances = np.sqrt(param_vectors[:, 0]**2 + param_vectors[:, 1]**2)

# Find the indices of the three/four farthest arrows
farthest_indices = np.argsort(distances)[-3:]

# Circle the two farthest arrows and add text with their indices
for i in farthest_indices:
    plt.scatter(param_vectors[i, 0], param_vectors[i, 1], s=200, facecolors='none', edgecolors=color2, linewidths=2, zorder=4)
    if args.text:
    # Add text annotations for the farthest arrows
        text = plt.text(param_vectors[i, 0] + 0.025, param_vectors[i, 1] + 0.025, params_df.columns[i + 2], color=color3, fontsize=10, fontweight='bold', zorder=5)
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

if args.save:
    os.makedirs('images/interpretability/', exist_ok=True)
    # if args.text:
    #     plt.savefig(f'images/interpretability/{model_st}_{dataset}{mutation}_CCA_annot.png', dpi=100)
    # else:
    #     plt.savefig(f'images/interpretability/{model_st}_{dataset}{mutation}_CCA.png', dpi=300, bbox_inches='tight')
plt.show()

if args.parameter:
    parameter = args.parameter

    if args.embedding:
        embedding = args.embedding

    else:
        # Get all embedding columns
        embedding_cols = [col for col in merged_df.columns if col.startswith('LV-')]

        # Compute Pearson correlations
        correlations = merged_df[embedding_cols].corrwith(merged_df[parameter], method='pearson')

        # Find the embedding with the highest absolute correlation
        best_embedding = correlations.abs().idxmax()
        best_corr = correlations[best_embedding]

        embedding = best_embedding
        print(f"Embedding with highest correlation to {parameter}: {best_embedding} (correlation = {best_corr:.3f})")

    x_axis = merged_df[parameter]
    y_axis = merged_df[embedding]
    if not args.average:
        x_axis = x_axis[::10]
        y_axis = y_axis[::10]

    plt.figure(figsize=(6, 4))
    plt.plot(y_axis, x_axis, '.', alpha=0.5)
    plt.xlabel(f'{embedding} activation')
    plt.ylabel(parameter)
    plt.title(f'{parameter} vs {embedding} activation')
    plt.grid(True)
    plt.tight_layout()
    # Calculate and print correlation coefficient
    corr = np.corrcoef(x_axis, y_axis)[0, 1]
    plt.text(0.95, 0.95, f'Pearson correlation: {corr:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))
    # plt.text(0.05, 0.95, f'Pearson correlation: {corr:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7))

    if args.save:
        os.makedirs('images/interpretability/activations/', exist_ok=True)
        plt.savefig(f'images/interpretability/activations/{model_st}_{dataset}{mutation}_{parameter}_{embedding}_activation.png', dpi=300, bbox_inches='tight')
    plt.show()