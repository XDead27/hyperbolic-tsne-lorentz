import os
import traceback

from hyperbolicTSNE import load_data, Datasets, SequentialOptimizer, initialization, HyperbolicTSNE, HyperbolicKL 

data_home = "datasets"
log_path = "temp/poincare/"  # path for saving embedding snapshots

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
    knn_method="hnswlib"
)

# Compute an initial embedding of the data via PCA
X_embedded = initialization(
    n_samples=dataX.shape[0],
    n_components=2,
    X=dataX,
    random_state=seed,
    method="pca"
)

hkl_params = HyperbolicKL.bh_tsne()
hkl_params["params"]["calc_both"] = False
hkl_params["params"]["area_split"] = False
hkl_params["params"]["grad_fix"] = False 

cf = HyperbolicKL(n_components=2, other_params=hkl_params)
cf.grad(model="lorentz", Y=X_embedded, V=V)

timings = cf.results[0]
print("Build time Lorentz: " + str(timings[0]) + "s")
print("Queary time Lorentz: " + str(timings[1]) + "s")

cf = HyperbolicKL(n_components=2, other_params=hkl_params)
cf.grad(model="poincare", Y=X_embedded, V=V)

timings = cf.results[0]
print("Build time Poincare: " + str(timings[0]) + "s")
print("Queary time Poincare: " + str(timings[1]) + "s")


