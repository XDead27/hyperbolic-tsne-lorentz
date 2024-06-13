import os
import sys
import traceback

from experiments_and_plots.configs import setup_experiment

from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare, plot_lorentz, animate
from hyperbolicTSNE import load_data, Datasets, SequentialOptimizer, initialization, HyperbolicTSNE
from hyperbolicTSNE.initializations_ import to_lorentz, from_lorentz


config_id = [1120]

ci, cfg, paths = setup_experiment(config_id)
cfg = cfg[0]

data_home = paths["datasets_path"]
log_path = paths["logging_path"]
results_path = os.path.join(paths["results_path"], "basic_script")

only_animate = False
seed = 42
dataset = Datasets.MNIST  # the Datasets handler provides access to several data sets used throughout the repository
num_points = 10000  # we use a subset for demonstration purposes, full MNIST has N=70000
perp = 30  # we use a perplexity of 30 in this example

dataX, dataLabels, D, V, _ = load_data(
    dataset, 
    data_home=data_home, 
    random_state=seed, 
    to_return="X_labels_D_V",
    hd_params={"perplexity": perp}, 
    sample=num_points, 
    knn_method="hnswlib"  # we use an approximation of high-dimensional neighbors to speed up computations
)

model = cfg["opt_params"]["hyperbolic_model"]
n_components = cfg["data_num_components"]
opt_params = cfg["get_opt_params"](ci, num_points)

# Start: configure logging
logging_dict = {
    "log_path": log_path
}
opt_params["logging_dict"] = logging_dict

log_path = opt_params["logging_dict"]["log_path"]
# Delete old log path
if os.path.exists(log_path) and not only_animate:
    import shutil
    shutil.rmtree(log_path)
# End: logging

print(f"config: {cfg['opt_params']}")

# Compute an initial embedding of the data via PCA
X_embedded = initialization(
    n_samples=dataX.shape[0],
    n_components=2,
    X=dataX,
    random_state=seed,
    method="pca"
)

# if model == "lorentz":
#     X_embedded = to_lorentz(X_embedded)


# Initialize the embedder
htsne = HyperbolicTSNE(
    init=X_embedded,
    n_components=n_components,
    metric="precomputed",
    verbose=True,
    opt_method=SequentialOptimizer,
    opt_params=opt_params
)

# Compute the embedding
try:
    hyperbolicEmbedding = htsne.fit_transform((D, V))
except ValueError:
    hyperbolicEmbedding = find_last_embedding(log_path)
    traceback.print_exc()

# Create a rendering of the embedding and save it to a file
if not os.path.exists(results_path):
    os.mkdir(results_path)
fig_p = plot_poincare(hyperbolicEmbedding, dataLabels)
fig_p.savefig(f"{results_path}/{dataset.name}_{model}_p.png" if len(sys.argv) <= 1 else f"{results_path}/{sys.argv[1]}_p.png")

if model == "lorentz":
    fig_l = plot_lorentz(find_last_embedding(log_path), dataLabels)
    fig_l.savefig(f"{results_path}/{dataset.name}_{model}_l.png" if len(sys.argv) <= 1 else f"{results_path}/{sys.argv[1]}_l.png")

# This renders a GIF animation of the embedding process. If FFMPEG is installed, the command also supports .mp4 as file ending 
animate(logging_dict, dataLabels, f"{results_path}/{dataset.name}_{model}_ani.gif" if len(sys.argv) <= 1 else f"{results_path}/{sys.argv[1]}_ani.gif", fast=True, plot_ee=True)

if model == "lorentz":
    animate(logging_dict, dataLabels, f"{results_path}/{dataset.name}_{model}_hyperb_ani.gif" if len(sys.argv) <= 1 else f"{results_path}/{sys.argv[1]}_hyperb_ani.gif", fast=True, plot_ee=True, n_dims=n_components)

