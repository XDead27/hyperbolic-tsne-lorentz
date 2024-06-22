"""
This script creates a table with the one nearest neighbor erroras obtained in embeddings with exact hyperbolic t-SNE vs.
embeddings that use the polar quad tree as acceleration data structure. The latter values are averaged over different
values of theta. Here, the one-nearest neighbor error is defined to be:
    1 - (number of points whose nearest embedding neighbor has the same label as them / number of all points).
"""

###########
# IMPORTS #
###########

from pathlib import Path
import numpy as np
from hyperbolicTSNE.data_loaders import Datasets, load_data
from hyperbolicTSNE.hd_mat_ import _distance_matrix
from hyperbolicTSNE.hyperbolic_barnes_hut.tsne import distance_py

from configs import setup_experiment
import os

#################################
# GENERAL EXPERIMENT PARAMETERS #
#################################

ids = [
    1010,
    1100
]
_, cfgs, paths = setup_experiment(ids)

DATASETS_DIR = paths["datasets_path"]
results_path = Path(paths["results_path"])
results_path = results_path.joinpath("samples_per_data_set")
K_HYPERBOLIC_NEIGHBOR_APPROXIMATION = 15  # Number of nearest neighbors to consider in a Euclidean approximation to find
# the actual nearest hyperbolic neighbor
datasets = [
    # Datasets.LUKK,
    Datasets.MYELOID8000,
    Datasets.PLANARIA,
    Datasets.MNIST,
    Datasets.C_ELEGANS,
    # Datasets.WORDNET
]


##################
# Helper Methods #
##################
def sort_dataframe(frame_to_be_sorted):
    # frame_to_be_sorted.loc[frame_to_be_sorted.dataset == "LUKK", "order"] = 1
    frame_to_be_sorted.loc[frame_to_be_sorted.dataset == "MYELOID8000", "order"] = 2
    frame_to_be_sorted.loc[frame_to_be_sorted.dataset == "PLANARIA", "order"] = 3
    frame_to_be_sorted.loc[frame_to_be_sorted.dataset == "MNIST", "order"] = 4
    # frame_to_be_sorted.loc[frame_to_be_sorted.dataset == "WORDNET", "order"] = 5
    frame_to_be_sorted.loc[frame_to_be_sorted.dataset == "C_ELEGANS", "order"] = 6
    frame_to_be_sorted = frame_to_be_sorted.sort_values(by="order", ascending=True)
    return frame_to_be_sorted


def one_nearest_neighbor_error(embedding_path):
    # Load the embedding and compute an approximation of its distance matrix
    Y = np.load(embedding_path.joinpath("final_embedding.npy"))
    D_Y = _distance_matrix(Y, method="sklearn", n_neighbors=K_HYPERBOLIC_NEIGHBOR_APPROXIMATION)

    num_points = D_Y.shape[0]
    num_available_nn_ld = D_Y[0, :].nnz

    # Computation of ordered neighbourhoods
    nz_D_Y = D_Y.nonzero()
    nz_rows_D_Y = nz_D_Y[0].reshape(-1, num_available_nn_ld)  # row coordinate of nz elements from D_Y
    nz_cols_D_Y = nz_D_Y[1].reshape(-1, num_available_nn_ld)  # col coordinate of nz elements from D_Y
    nz_dists_D_Y = np.asarray(D_Y[nz_rows_D_Y, nz_cols_D_Y].todense())
    sorting_ids_nz_dists_D_Y = np.argsort(nz_dists_D_Y, axis=1)
    sorted_nz_cols_D_Y = nz_cols_D_Y[nz_rows_D_Y, sorting_ids_nz_dists_D_Y]  # sorted cols of nz_D_Y
    sorted_nz_cols_D_Y = sorted_nz_cols_D_Y[:, 0:K_HYPERBOLIC_NEIGHBOR_APPROXIMATION]  # only get NNs that will be used

    # The above approximates the neighborhood by Euclidean distances, replace these by hyperbolic ones and resort
    arr = np.zeros(sorted_nz_cols_D_Y.shape)
    for (i, j), v in np.ndenumerate(sorted_nz_cols_D_Y):
        arr[i, j] = distance_py(Y[i], Y[v])
    sorting_ids_arr = np.argsort(arr, axis=1)
    sorted_nz_cols_D_Y = sorted_nz_cols_D_Y[nz_rows_D_Y, sorting_ids_arr]
    sorted_nz_cols_D_Y = sorted_nz_cols_D_Y[:, 0:1]  # only get NNs that will be used

    # Compute the nearest neighbor error based on the labels of the points
    num_correct_one_nearest = 0
    for point_id in range(num_points):
        # If the nearest point to a point carries the same label, increase the number of correct points
        if labels[point_id] == labels[sorted_nz_cols_D_Y[point_id]]:
            num_correct_one_nearest += 1

    return 1 - (num_correct_one_nearest / num_points)


####################
# READING THE DATA #
####################
data = []

# Iterate over the datasets
for dataset in datasets:
    for cfg in cfgs:
        print(f"[Table One Nearest Neighbor Error] Processing data {dataset.name}, nearest neighbor errors:")
        dataX, labels = load_data(dataset, to_return='X_labels', data_home=DATASETS_DIR,)

        # For each dataset, find the embedding quality of the exact embedding
        config_dir = results_path.joinpath(dataset.name, f"size_{dataX.shape[0]}", f"configuration_{cfg['config_id']}")

        errs = []
        for run_dir in config_dir.glob("*"):
            if run_dir.is_dir() and run_dir.joinpath("final_embedding.npy").exists():
                onne = one_nearest_neighbor_error(run_dir)
                errs.append(onne)

        print(f"{cfg['name']} error: {np.mean(errs)}")
