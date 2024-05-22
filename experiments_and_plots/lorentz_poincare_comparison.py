import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from hyperbolicTSNE import load_data, Datasets, initialization, HyperbolicKL
from hyperbolicTSNE.solver_ import gradient_descent

num_its = int(sys.argv[1])

data_home = "datasets"
log_path = "temp/poincare/"  # path for saving embedding snapshots

max_num_points = 70000
seed = 42
dataset = Datasets.MNIST  # the Datasets handler provides access to several data sets used throughout the repository
num_points = np.linspace(10, max_num_points, 19, dtype=int)
perp = 30  # we use a perplexity of 30 in this example

timings_p = np.zeros([num_points.size, 4])
timings_l = np.zeros([num_points.size, 4])

for i in range(num_points.size):
    num_p = num_points[i]
    dataX, dataLabels, D, V, _ = load_data(
        dataset, 
        data_home=data_home, 
        random_state=seed, 
        to_return="X_labels_D_V",
        hd_params={"perplexity": perp}, 
        sample=num_p, 
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

    print("\n\n(For num_points: " + str(num_p) + ")\n############# POINCARE ##############")
    cfp = HyperbolicKL(n_components=2, other_params=hkl_params)
    gradient_descent(X_embedded, cfp, cf_params, n_iter=num_its, learning_rate=0.1, hyperbolic_model="poincare")

    # print(cfp.results)
    timings_p[i] = np.mean(cfp.results, axis=0)
    print("--> Build time: " + str(timings_p[i][0]) + "s")
    print("--> Query time: " + str(timings_p[i][1]) + "s")
    print(" |--> neg: " + str(timings_p[i][2]) + "s")
    print(" |--> pos: " + str(timings_p[i][3]) + "s")

    print("\n\n############# LORENTZ ##############")
    cf = HyperbolicKL(n_components=2, other_params=hkl_params)
    gradient_descent(X_embedded_cpy, cf, cf_params, n_iter=num_its, learning_rate=0.1, hyperbolic_model="lorentz")

    # print(cf.results)
    timings_l[i] = np.mean(cf.results, axis=0)
    print("==> Build time: " + str(timings_l[i][0]) + "s")
    print("==> Query time: " + str(timings_l[i][1]) + "s")
    print(" |--> neg: " + str(timings_l[i][2]) + "s")
    print(" |--> pos: " + str(timings_l[i][3]) + "s")

fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(num_points, timings_p[:, 0], c='r')
ax1.plot(num_points, timings_l[:, 0], c='b')
ax1.legend(['Poincare polar quadtree-accelerated t-SNE', 'Our version'])
ax1.set_title("Tree build time")
ax1.set_ylabel("time (s)")

ax2.plot(num_points, timings_p[:, 1], c='r')
ax2.plot(num_points, timings_l[:, 1], c='b')
ax2.legend(['Poincare polar quadtree-accelerated t-SNE', 'Our version'])
ax2.set_title("Tree query time")
ax2.set_ylabel("time (s)")
ax2.set_xlabel("# points in dataset")

fig.suptitle("Mean times of tree operations")

plt.tight_layout()

plt.savefig("results/Lorentz_Poincare_Comparison.png")
plt.show()
