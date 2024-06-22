"""
This experiment runs the hyperbolic tsne code changing several parameters:
- dataset: [LUKK, MYELOID8000, PLANARIA, MNIST, C_ELEGANS, WORDNET]
- tsne_type: [accelerated, exact]
For each configuration combination, it saves the embedding coordinates, a plot of the embedding, and
timing data for the iterations.
If a run does not finish, the results are not saved.
The code only computes the runs that do not have a folder.
"""
###########
# IMPORTS #
###########

from pathlib import Path
import numpy as np
import json
from scipy.sparse import load_npz
import matplotlib.pyplot as plt

from hyperbolicTSNE.quality_evaluation_ import hyperbolic_nearest_neighbor_preservation
from hyperbolicTSNE import Datasets, load_data

from configs import setup_experiment

#################################
# GENERAL EXPERIMENT PARAMETERS #
#################################

_, _, paths = setup_experiment([])

BASE_DIR = Path(paths["results_path"])
BASE_DIR = BASE_DIR.joinpath("full_size_one_run")
DATASETS_DIR = paths["datasets_path"]

# Constants
NEIGHBORHOOD_SIZE = 100  # Neighborhood size to grab the k_max many neighbors from
K_START = 1  # Lowest point in the precision/recall curve
K_MAX = 30  # Highest point in the precision/recall curve
PERP = 30  # perplexity value to be used throughout the experiments
hd_params = {"perplexity": PERP}

c_fig, c_axs = plt.subplots(1, 4, figsize=(25, 5))
i = 0

# Iterate over all results,
for dataset_dir in BASE_DIR.glob("*"):
    if dataset_dir.is_dir():
        dataset_name = dataset_dir.stem
        dataX = load_data(Datasets[dataset_name], data_home=DATASETS_DIR, to_return="X", hd_params=hd_params)

        fig, ax = plt.subplots()
        ax.set_title(dataset_name)

        max_x, max_y = 0, 0

        for size_dir in dataset_dir.glob("*"):
            if size_dir.is_dir():
                for config_dir in size_dir.glob("*"):
                    if config_dir.is_dir():
                        for run_dir in config_dir.glob("*"):
                            if run_dir.is_dir():
                                print(f"[NNP plot] Processing {run_dir}.")
                                subset_idx = np.load(run_dir.joinpath("subset_idx.npy"), allow_pickle=True)
                                dataX_sample = dataX[subset_idx]

                                D_path = list(run_dir.glob("D.np*"))[0]
                                sparse_D = D_path.suffix == ".npz"
                                if sparse_D:
                                    D_X = load_npz(D_path)
                                else:
                                    D_X = np.load(D_path, allow_pickle=True)[()]

                                dataY = np.load(run_dir.joinpath("final_embedding.npy"), allow_pickle=True)

                                params = json.load(open(run_dir.joinpath("params.json")))

                                try:
                                    thresholds = np.load(run_dir.joinpath("thresholds.npy"), allow_pickle=True)
                                    precisions = np.load(run_dir.joinpath("precisions.npy"), allow_pickle=True)
                                    recalls = np.load(run_dir.joinpath("recalls.npy"), allow_pickle=True)
                                    true_positives = np.load(run_dir.joinpath("true_positives.npy"), allow_pickle=True)
                                    print("[NNP plot] Using saved values...")
                                except:
                                    thresholds, precisions, recalls, true_positives = \
                                        hyperbolic_nearest_neighbor_preservation(
                                            dataX_sample,
                                            dataY,
                                            K_START,
                                            K_MAX,
                                            D_X,
                                            False,
                                            False,
                                            False,
                                            "full",
                                            NEIGHBORHOOD_SIZE
                                        )

                                    # save results inside folder
                                    np.save(run_dir.joinpath("thresholds.npy"), thresholds)
                                    np.save(run_dir.joinpath("precisions.npy"), precisions)
                                    np.save(run_dir.joinpath("recalls.npy"), recalls)
                                    np.save(run_dir.joinpath("true_positives.npy"), true_positives)

                                # Add points to plot
                                ax.plot(precisions, recalls, label=params['name'])
                                c_axs[i].plot(precisions, recalls, label=params['name'])

                                max_x = max(max_x, np.max(precisions))
                                max_y = max(max_y, np.max(recalls))

                ax.set_xlabel("Precision")
                ax.set_ylabel("Recall")
                ax.legend()
                fig.savefig(size_dir.joinpath(f"{dataset_name}_prec-vs-rec_{subset_idx.shape[0]}.png"))
                plt.close(fig)

        c_axs[i].set_title(dataset_name)
        c_axs[i].legend()
        # c_axs[i].set_xlim([0, max_x])  # Recompute the limits of the data
        # c_axs[i].set_ylim([0, max_y])  # Recompute the limits of the data
        i += 1

c_fig.supxlabel("Precision")
c_fig.supylabel("Recall")

plt.tight_layout()
c_fig.savefig(BASE_DIR.joinpath(f"prec-vs-rec-all.png"))
plt.close(c_fig)

