import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from hyperbolicTSNE import load_data, Datasets, initialization, HyperbolicTSNE, SequentialOptimizer

from configs import setup_experiment

num_its = int(sys.argv[1])

config_ids = [1000, 1100, 1120]

ci, cfgs, paths = setup_experiment(config_ids)

data_home = paths["datasets_path"]
log_path = "temp/poincare/"  # path for saving embedding snapshots

max_num_points = 10000
seed = 42
dataset = Datasets.MNIST  # the Datasets handler provides access to several data sets used throughout the repository
num_points = np.linspace(1000, max_num_points, 3, dtype=int)
perp = 30  # we use a perplexity of 30 in this example

timings_p = np.zeros([num_points.size, 8])
timings_l = np.zeros([num_points.size, 8])

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

    print("\n\n(For num_points: " + str(num_p))
    for cfg in cfgs:
        print(f"############# {cfg['name']} ##############")

        cfg["opt_params"]["exaggeration_its"] = num_its
        cfg["opt_params"]["gradientDescent_its"] = num_its
        opt_params = cfg["get_opt_params"](ci, dataX.shape[0])

        hdeo_hyper = HyperbolicTSNE(
            init=X_embedded,
            n_components=cfg["data_num_components"],
            metric="precomputed",
            verbose=0,
            opt_method=SequentialOptimizer,
            opt_params=opt_params
        )

        res_hdeo_hyper = hdeo_hyper.fit_transform((D, V))

        # print(cfp.results)
        timings_p[i] = np.mean(hdeo_hyper.optimizer.cf.results, axis=0)
        print(f"--> Build time: {str(timings_p[i][0])}s")
        print(f"--> Query time: {str(timings_p[i][1])}s")
        print(f" |--> neg: {str(timings_p[i][2])}s")
        print(f"   |--> distance grad: {timings_p[i][4]} / {timings_p[i][5]}")
        print(f"   |--> distance: {timings_p[i][6]} / {timings_p[i][7]}")
        print(f" |--> pos: {str(timings_p[i][3])}s\n")

# fig, (ax1, ax2) = plt.subplots(2)
#
# ax1.plot(num_points, timings_p[:, 0], c='r')
# ax1.plot(num_points, timings_l[:, 0], c='b')
# ax1.legend(['Poincare polar quadtree-accelerated t-SNE', 'Our version'])
# ax1.set_title("Tree build time")
# ax1.set_ylabel("time (s)")
#
# ax2.plot(num_points, timings_p[:, 1], c='r')
# ax2.plot(num_points, timings_l[:, 1], c='b')
# ax2.legend(['Poincare polar quadtree-accelerated t-SNE', 'Our version'])
# ax2.set_title("Tree query time")
# ax2.set_ylabel("time (s)")
# ax2.set_xlabel("# points in dataset")
#
# fig.suptitle("Mean times of tree operations")
#
# plt.tight_layout()
#
# plt.savefig("results/Lorentz_Poincare_Comparison.png")
# plt.show()
