import sys
import numpy as np
from hyperbolicTSNE import load_data, Datasets, initialization, HyperbolicKL
from hyperbolicTSNE.solver_ import gradient_descent

num_its = int(sys.argv[1])

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
X_embedded_cpy = X_embedded.copy()

hkl_params = HyperbolicKL.bh_tsne()
hkl_params["params"]["calc_both"] = False
hkl_params["params"]["area_split"] = False
hkl_params["params"]["grad_fix"] = False 

cf_params = {
    "V": V,
}

print("############# POINCARE ##############")
cfp = HyperbolicKL(n_components=2, other_params=hkl_params)
gradient_descent(X_embedded, cfp, cf_params, n_iter=num_its, learning_rate=0.1, hyperbolic_model="poincare")

# print(cfp.results)
timings = np.mean(cfp.results, axis=0)
print("--> Build time: " + str(timings[0]) + "s")
print("--> Query time: " + str(timings[1]) + "s")
print(" |--> neg: " + str(timings[2]) + "s")
print(" |--> pos: " + str(timings[3]) + "s")

print("############# LORENTZ ##############")
cf = HyperbolicKL(n_components=2, other_params=hkl_params)
gradient_descent(X_embedded_cpy, cf, cf_params, n_iter=num_its, learning_rate=0.1, hyperbolic_model="lorentz")

# print(cf.results)
timings = np.mean(cf.results, axis=0)
print("==> Build time: " + str(timings[0]) + "s")
print("==> Query time: " + str(timings[1]) + "s")
print(" |--> neg: " + str(timings[2]) + "s")
print(" |--> pos: " + str(timings[3]) + "s")

