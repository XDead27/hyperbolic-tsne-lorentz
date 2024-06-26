"""
This script goes through the nearest neighbor preservation data generated by the
`data_generation_nearest_neighbor_preservation_per_theta_value_MNIST.py`
script and creates a plot with one precision/recall curve per theta value.
"""
###########
# IMPORTS #
###########

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from hyperbolicTSNE import Datasets

from configs import load_vars_env

#################################
# GENERAL EXPERIMENT PARAMETERS #
#################################

paths = load_vars_env()

BASE_DIR = paths["results_path"] + "/nnp_per_theta"

# Constants
dataset = Datasets.MNIST  # The dataset to run the experiment on
cmap = cm.get_cmap('viridis', 11)

###################
# EXPERIMENT LOOP #
###################

# Save final embedding
fig, ax = plt.subplots()
ax.set_title("MNIST")
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')

for theta in [x / 10 for x in range(0, 11, 1)]:  # Iterate over the different values for theta

    print(f"[nnp_per_theta] Processing {dataset}, Theta: {theta}")

    run_dir = Path(f"{BASE_DIR}/theta_{theta}/")

    precisions = np.load(run_dir.joinpath(f"precisions_theta_{theta}.npy"))
    recalls = np.load(run_dir.joinpath(f"recalls_theta_{theta}.npy"))

    ax.plot(precisions, recalls, label=f"{theta}", color=cmap(int(theta * 10)))

ax.legend()
fig.savefig(Path(BASE_DIR).joinpath(f"{dataset}_prec-vs-rec_per_theta.png"))
