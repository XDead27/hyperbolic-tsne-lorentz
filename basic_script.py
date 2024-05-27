import os
import sys
import traceback

from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare, animate
from hyperbolicTSNE import load_data, Datasets, SequentialOptimizer, initialization, HyperbolicTSNE
from hyperbolicTSNE.initializations_ import to_lorentz, from_lorentz

data_home = "datasets"
log_path = "temp/poincare/"  # path for saving embedding snapshots

# model = "poincare"
model = "lorentz"
n_components = 2 if model == "poincare" else 3
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

exaggeration_factor = 12  # Just like regular t-SNE, we use early exaggeration with a factor of 12
# learning_rate = (dataX.shape[0] * 1) / (exaggeration_factor * 1000)  # We adjust the learning rate to the hyperbolic setting
learning_rate = (dataX.shape[0] * 1) / (exaggeration_factor * 5)
ex_iterations = 250  # The embedder is to execute 250 iterations of early exaggeration, ...
main_iterations = 750  # ... followed by 750 iterations of non-exaggerated gradient descent.

opt_config = dict(
    learning_rate_ex=learning_rate / 12,  # learning rate during exaggeration
    learning_rate_main=learning_rate,  # learning rate main optimization 
    exaggeration=exaggeration_factor, 
    exaggeration_its=ex_iterations, 
    gradientDescent_its=main_iterations, 
    vanilla=False,  # if vanilla is set to true, regular gradient descent without any modifications is performed; for  vanilla set to false, the optimization makes use of momentum and gains
    momentum_ex=0.35,  # Set momentum during early exaggeration to 0.5
    momentum=0.6,  # Set momentum during non-exaggerated gradient descent to 0.8
    exact=False,  # To use the quad tree for acceleration (like Barnes-Hut in the Euclidean setting) or to evaluate the gradient exactly
    area_split=False,  # To build or not build the polar quad tree based on equal area splitting or - alternatively - on equal length splitting
    n_iter_check=10,  # Needed for early stopping criterion
    size_tol=0.97,  # Size of the embedding to be used as early stopping criterion
    hyperbolic_model=model,
)

opt_params = SequentialOptimizer.sequence_poincare(**opt_config)

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

print(f"config: {opt_config}")

# Compute an initial embedding of the data via PCA
X_embedded = initialization(
    n_samples=dataX.shape[0],
    n_components=2,
    X=dataX,
    random_state=seed,
    method="pca"
)

if model == "lorentz":
    X_embedded, lorentz_tr_err = to_lorentz(X_embedded)
    print(f'Max Lorentz Translation Error: {lorentz_tr_err}')


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
    poincare_embedding = from_lorentz(hyperbolicEmbedding) if model == "lorentz" else hyperbolicEmbedding
except ValueError:
    poincare_embedding = find_last_embedding(log_path)
    traceback.print_exc()

# Create a rendering of the embedding and save it to a file
if not os.path.exists("results"):
    os.mkdir("results")
fig = plot_poincare(poincare_embedding, dataLabels)
fig.savefig(f"results/{dataset.name}_{model}.png" if len(sys.argv) <= 1 else f"results/{sys.argv[1]}.png")

# This renders a GIF animation of the embedding process. If FFMPEG is installed, the command also supports .mp4 as file ending 
animate(logging_dict, dataLabels, f"results/{dataset.name}_{model}_ani.gif" if len(sys.argv) <= 1 else f"results/{sys.argv[1]}_ani.gif", fast=True, plot_ee=True)

