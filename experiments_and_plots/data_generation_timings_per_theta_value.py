"""
This experiment runs the hyperbolic tsne code changing several parameters:
- dataset: [LUKK, MYELOID8000, PLANARIA, MNIST, C_ELEGANS, WORDNET]
- Value of Theta in the approximation: [0.0, 0.1, ..., 1.0]
The code only computes the runs that do not have a folder.
"""

###########
# IMPORTS #
###########

from pathlib import Path

import csv
import json
import traceback

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import issparse, save_npz

from configs import setup_experiment

from hyperbolicTSNE import Datasets, load_data, initialization, SequentialOptimizer, HyperbolicTSNE
from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare

#################################
# GENERAL EXPERIMENT PARAMETERS #
#################################
ids = [
    1100,
]

ci, cfgs, paths = setup_experiment(ids)
cfg = cfgs[0]

BASE_DIR = paths["results_path"] + "/timings_per_theta"
DATASETS_DIR = paths["datasets_path"]

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
    # Datasets.WORDNET
]
thetas = [n / 20 for n in range(20, -1, -1)]  # The different theta values to be used in the acceleration experiment

###################
# EXPERIMENT LOOP #
###################

overview_created = False
for dataset in datasets:  # Iterate over the data sets

    rng = np.random.default_rng(seed=SEED)  # random number generator

    dataX, dataLabels, D, V = load_data(
        dataset,
        data_home=DATASETS_DIR,
        to_return="X_labels_D_V",  # Return the high-dimensional data, its labels, the NN-graph, the probabilty matrix
        hd_params=hd_params,
        knn_method=KNN_METHOD
    )

    n_samples = dataX.shape[0]

    X_embedded = initialization(  # create an initial embedding of the data into 2-dimensional space via PCA
        n_samples=n_samples,
        n_components=2,
        X=dataX,
        random_state=rng.integers(0, 1000000),
        method="pca"
    )

    for config_id, theta in enumerate(thetas):

        print(f"[theta_timings] Processing {dataset}, config_id ({config_id}) with Theta: {theta}")

        cfg["opt_params"]["angle"] = theta
        opt_params = cfg["get_opt_params"](ci, dataX.shape[0])
        run_dir = Path(f"{BASE_DIR}/{dataset.name}/theta_{theta}/")

        if run_dir.exists():
            # Skip already computed embeddings
            print(f"[theta_timings] - Exists so not computing it: {run_dir}")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)

            params = {
                "name": cfg["name"],
                "perplexity": PERP,
                "seed": SEED,
                "sample_size": int(n_samples),
                "theta": theta,
                "opt_params": cfg["opt_params"],
            }

            print(f"[theta_timings] - Starting configuration {config_id} with dataset {dataset.name}: {params}")

            # opt_params["logging_dict"] = {
            #     "log_path": str(run_dir.joinpath("embeddings"))
            # }

            # Save the high-dimensional neighborhood matrices for later use
            json.dump(params, open(run_dir.joinpath("params.json"), "w"))
            if issparse(D):
                save_npz(run_dir.joinpath("D.npz"), D)
            else:
                np.save(run_dir.joinpath("D.npy"), D)
            if issparse(V):
                save_npz(run_dir.joinpath("P.npz"), V)
            else:
                np.save(run_dir.joinpath("P.npy"), V)

            hdeo_hyper = HyperbolicTSNE(
                init=X_embedded,
                n_components=cfg["data_num_components"],
                metric="precomputed",
                verbose=2,
                opt_method=SequentialOptimizer,
                opt_params=opt_params
            )

            error_title = ""
            try:
                res_hdeo_hyper = hdeo_hyper.fit_transform((D, V))
            except ValueError:

                error_title = "_error"
                # res_hdeo_hyper = find_last_embedding(opt_params["logging_dict"]["log_path"])
                res_hdeo_hyper = None
                traceback.print_exc(file=open(str(run_dir) + "traceback.txt", "w"))

                print("[theta_timings] - Run failed ...")

            else:  # we save the data if there were no errors

                print("[theta_timings] - Finished running, saving run data directory ...")

                # Save the final embedding coordinates
                np.save(run_dir.joinpath("final_embedding.npy"), res_hdeo_hyper)

                # Save a plot of the final embedding
                fig = plot_poincare(res_hdeo_hyper, labels=dataLabels)
                fig.savefig(run_dir.joinpath(f"final_embedding{error_title}.png"))
                plt.close(fig)

                # np.save(run_dir.joinpath("logging_dict.npy"), opt_params["logging_dict"])

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
                            [dataset.name, *params.values(), str(run_dir).replace(str(BASE_DIR), "."),
                             error_title != ""])

                    print()
