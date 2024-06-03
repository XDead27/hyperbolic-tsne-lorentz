"""
This experiment runs the hyperbolic tsne code changing several parameters:
- dataset: [LUKK, MYELOID8000, PLANARIA, MNIST, C_ELEGANS, WORDNET] 
- tsne_type: [accelerated, exact]
- splitting strategy of the polar quad tree: [equal_area, equal_length]
For each data set and each configuration combination, it saves the embedding coordinates, a plot of the embedding, and
timing data for the iterations.
If a run does not finish, the results are not saved.
The code only computes the runs that do not have a folder.
"""

###########
# IMPORTS #
###########

import csv
import json
import traceback
from itertools import product
from pathlib import Path

import numpy as np
from scipy.sparse import issparse, save_npz
from matplotlib import pyplot as plt

from hyperbolicTSNE import Datasets, load_data, initialization, hd_matrix, SequentialOptimizer, HyperbolicTSNE
from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare
from configs import TSNEConfigs

#################################
# GENERAL EXPERIMENT PARAMETERS #
#################################

BASE_DIR = "./results/full_size_one_run"  # directory where results will be saved
DATASETS_DIR = "./datasets"  # directory to read the data from

# Constants
SEED = 42  # seed to initialize random processes
PERP = 30  # perplexity value to be used throughout the experiments
KNN_METHOD = "hnswlib"  # use hnswlib for determining nearest neighbors in high-dimensional space; note that this is
# an approximation, switch to "sklearn" for an exact method
VANILLA = False  # if vanilla is set to true, regular gradient descent without any modifications is performed; for
# vanilla set to false, the optimization makes use of momentum and gains
EXAG = 12  # the factor by which the attractive forces are amplified during early exaggeration
hd_params = {"perplexity": PERP}

# Variables
datasets = [
    # Datasets.LUKK,
    Datasets.MYELOID8000,
    Datasets.PLANARIA,
    Datasets.MNIST,
    Datasets.C_ELEGANS,
    # Datasets.WORDNET,
]
num_points = 100000
configs_instance = TSNEConfigs()
run_configs = [configs_instance.config_accelerated_poincare,
               configs_instance.config_accelerated_lorentz,
               configs_instance.config_exact_poincare,
               configs_instance.config_exact_lorentz]

###################
# EXPERIMENT LOOP #
###################

overview_created = False
for dataset in datasets:  # Iterate over the data sets

    rng = np.random.default_rng(seed=SEED)  # random number generator

    dataX, dataLabels = load_data(  # Load the data
        dataset,
        data_home=DATASETS_DIR,
        to_return="X_labels",  # Return the high-dimensional data and its labels
        hd_params=hd_params,
        knn_method=KNN_METHOD
    )

    n_samples = dataX.shape[0]
    sample_sizes = np.array([min(num_points, n_samples), ]).astype(int)  # only run the full size sample, don't create sub-samples

    X_embedded = initialization(  # create an initial embedding of the data into 2-dimensional space via PCA
        n_samples=n_samples,
        n_components=2,
        X=dataX,
        random_state=rng.integers(0, 1000000),
        method="pca"
    )

    for id, config in enumerate(product(sample_sizes, run_configs)):
        for run_n in [0, ]:  # do not repeat the run

            sample_size, tsne_config = config
            omit = False

            print(f"[experiment_grid] Processing {dataset}, run_id {run_n}, config_id ({tsne_config['config_id']}): {config}")

            # Generate random sample
            idx = rng.choice(np.arange(n_samples), sample_size, replace=False)
            idx = np.sort(idx)

            dataX_sample = dataX[idx]
            dataLabels_sample = dataLabels[idx]
            X_embedded_sample = X_embedded[idx]

            D, V = hd_matrix(X=dataX_sample, hd_params=hd_params, knn_method=KNN_METHOD)  # Compute the NN matrix

            opt_params = tsne_config["get_opt_params"](configs_instance, dataX_sample.shape[0])

            run_dir = Path(f"{BASE_DIR}/{dataset.name}/size_{sample_size}/configuration_{tsne_config['config_id']}/run_{run_n}/")

            if run_dir.exists():
                # Skip already computed embeddings
                print(f"[experiment_grid] - Exists so not computing it: {run_dir}")
            else:
                run_dir.mkdir(parents=True, exist_ok=True)

                params = {
                    "name": tsne_config["name"],
                    "perplexity": PERP,
                    "seed": SEED,
                    "sample_size": int(sample_size),
                    "opt_params": tsne_config["opt_params"],
                }

                print(f"[experiment_grid] - Starting configuration {tsne_config['config_id']} with dataset {dataset.name}: {params}")

                opt_params["logging_dict"] = {
                    "log_path": str(run_dir.joinpath("embeddings"))
                }

                # Save the high-dimensional neighborhood matrices for later use
                json.dump(params, open(run_dir.joinpath("params.json"), "w"))
                np.save(run_dir.joinpath("subset_idx.npy"), idx)
                if issparse(D):
                    save_npz(run_dir.joinpath("D.npz"), D)
                else:
                    np.save(run_dir.joinpath("D.npy"), D)
                if issparse(V):
                    save_npz(run_dir.joinpath("P.npz"), V)
                else:
                    np.save(run_dir.joinpath("P.npy"), V)

                hdeo_hyper = HyperbolicTSNE(  # Initialize an embedding object
                    init=X_embedded_sample,
                    n_components=tsne_config["data_num_components"],
                    metric="precomputed",
                    verbose=2,
                    opt_method=SequentialOptimizer,
                    opt_params=opt_params
                )

                error_title = ""
                try:
                    res_hdeo_hyper = hdeo_hyper.fit_transform((D, V))  # Compute the hyperbolic embedding
                except ValueError:

                    error_title = "_error"
                    res_hdeo_hyper = find_last_embedding(opt_params["logging_dict"]["log_path"])
                    traceback.print_exc(file=open(str(run_dir) + "traceback.txt", "w"))

                    print("[experiment_grid] - Run failed ...")

                else:  # we save the data if there were no errors

                    print("[experiment_grid] - Finished running, saving run data directory ...")

                    # Save the final embedding coordinates
                    np.save(run_dir.joinpath("final_embedding.npy"), res_hdeo_hyper)

                    # Save a plot of the final embedding
                    fig = plot_poincare(res_hdeo_hyper, labels=dataLabels_sample)
                    fig.savefig(run_dir.joinpath(f"final_embedding{error_title}.png"))
                    plt.close(fig)

                    np.save(run_dir.joinpath("logging_dict.npy"), opt_params["logging_dict"])

                    # Write out timings csv
                    timings = np.array(hdeo_hyper.optimizer.cf.results)
                    with open(run_dir.joinpath("timings.csv"), "w", newline="") as timings_file:
                        timings_writer = csv.writer(timings_file)
                        timings_writer.writerow(["it_n", "time_type", "total_time"])

                        for n, row in enumerate(timings):
                            timings_writer.writerow([n, "tree_building", row[0]])
                            timings_writer.writerow([n, "tot_gradient", row[1]])
                            timings_writer.writerow([n, "neg_force", row[2]])
                            timings_writer.writerow([n, "pos_force", row[3]])

                    # Create or append to overview csv file after every run
                    with open(run_dir.joinpath(f"overview_part.csv"), "w", newline="") as overview_file:
                        overview_writer = csv.writer(overview_file)
                        overview_writer.writerow(["dataset", *params, "run", "run_directory", "error"])
                        overview_writer.writerow(
                            [dataset.name, *params.values(), run_n, str(run_dir).replace(str(BASE_DIR), "."),
                             error_title != ""])

                    print()
