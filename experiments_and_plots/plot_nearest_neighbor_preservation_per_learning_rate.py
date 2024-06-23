"""
This script goes through the nearest neighbor preservation data generated by the
`data_generation_learning_rate.py`
script and creates a plot with one precision/recall curve per learning rate value.
"""
###########
# IMPORTS #
###########

from pathlib import Path
import numpy as np
import json
from scipy.sparse import load_npz
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from hyperbolicTSNE import Datasets, load_data, initialization
import pandas as pd

from hyperbolicTSNE.quality_evaluation_ import hyperbolic_nearest_neighbor_preservation
from configs import load_vars_env

import os

#################################
# GENERAL EXPERIMENT PARAMETERS #
#################################

paths = load_vars_env()

BASE_DIR = os.path.join(paths["results_path"], "runs_learning_rate")
DATASETS_DIR = paths["datasets_path"]

# Constants
SEED = 42  # seed to initialize random processes
PERP = 30  # perplexity value to be used throughout the experiments
VANILLA = False  # whether to use momentum or not
EXAG = 12  # the factor by which the attractive forces are amplified during early exaggeration
hd_params = {"perplexity": PERP}
# Constants
dataset = Datasets.MNIST  # The dataset to run the experiment on

cmap = cm.get_cmap('viridis', 7)

###################
# EXPERIMENT LOOP #
###################

dataX = load_data(  # Load the data
    dataset,
    data_home=DATASETS_DIR,
    to_return="X",
    hd_params=hd_params,
)

X_embedded = initialization(
    n_samples=dataX.shape[0],
    n_components=2,
    X=dataX,
    random_state=SEED,
    method="pca"
)

# Save final embedding
fig, ax = plt.subplots()
ax.set_title(dataset.name)
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')

DATASET_SUBDIR = os.path.join(BASE_DIR, dataset.name)

plot_list = []
print(f"[nnp_per_lrn] Searching only in {DATASET_SUBDIR}")
for subdir, dirs, files in os.walk(DATASET_SUBDIR):
    for file in files:
        if str(os.path.basename(os.path.join(subdir, file))) == "final_embedding.npy":
            lrn_coef = float(str(subdir).split('/')[-1].split('_')[-1])

            print(f"[nnp_per_lrn] Processing {dataset}, LRN: {lrn_coef}")
            
            run_dir = Path(os.path.join(BASE_DIR, dataset.name, f"lrn_{lrn_coef}"))

            if run_dir.joinpath("timings.csv").exists():
                timing_df = pd.read_csv(run_dir.joinpath("timings.csv"))
                n_its = len(timing_df.it_n.unique()) - 250
            else:
                n_its = 28

            D_path = list(run_dir.glob("D.np*"))[0]
            sparse_D = D_path.suffix == ".npz"
            if sparse_D:
                D_X = load_npz(D_path)
            else:
                D_X = np.load(D_path, allow_pickle=True)[()]

            dataY = np.load(run_dir.joinpath("final_embedding.npy"), allow_pickle=True)

            params = json.load(open(run_dir.joinpath("params.json")))


            try:
                precisions = np.load(run_dir.joinpath(f"precisions.npy"))
                recalls = np.load(run_dir.joinpath(f"recalls.npy"))
                print("[nnp_per_lrn] Using saved values...")
            except:
                # Compute Precision and Recall values for the embedding
                _, precisions, recalls, _ = hyperbolic_nearest_neighbor_preservation(
                    dataX,
                    dataY,
                    k_start=1,
                    k_max=30,
                    D_X=D_X,
                    exact_nn=True,
                    consider_order=False,
                    strict=False,
                    to_return="full"
                )
                np.save(run_dir.joinpath(f"precisions.npy"), precisions)
                np.save(run_dir.joinpath(f"recalls.npy"), recalls)

            label = '$\\frac{{1}}{{{:.2f}}}$'.format(float(lrn_coef)) if float(lrn_coef) < 100 and float(lrn_coef) != 1. else f'$\\frac{{1}}{{{int(lrn_coef)}}}$'
            label += f" ({n_its} its)"

            plot_list.append((lrn_coef, label, precisions, recalls, cmap(int(np.log(float(lrn_coef))))))

plot_list = sorted(plot_list, key=lambda item: item[0])

for _, label, precisions, recalls, color in plot_list:
    ax.plot(precisions, recalls, label=label, color=color)

ax.legend(title="Learning Rate Coef.")
fig.savefig(Path(BASE_DIR).joinpath(f"{dataset.name}_prec-vs-rec_per_lrn.png"))