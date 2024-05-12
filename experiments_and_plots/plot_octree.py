"""
This scripts takes an embedding of the C.Elegans data set and plots a polar quad tree on top of it.
"""
###########
# IMPORTS #
###########
import ctypes
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import pandas as pd
import seaborn as sns
from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.hyperbolic_barnes_hut.lotsne import _OcTree
from hyperbolicTSNE.hyperbolic_barnes_hut.lotsne import distance_py
from hyperbolicTSNE import load_data, Datasets, initialization, SequentialOptimizer, HyperbolicTSNE

#############
# VARIABLES #
#############
model = "lorentz"
exaggeration_factor = 12
ex_iterations = 250
main_iterations = 750

##############
# PLOT SETUP #
##############
MACHINE_EPSILON = np.finfo(np.double).eps
np.random.seed(594507)
matplotlib.rcParams['figure.dpi'] = 300
c = '#0173B2'  # Color for the tree
s = '-'  # style of the tree lines
w = 0.5  # width of the tree lines

##################
# Helper Methods #
##################
def get_random_point():
    length = np.sqrt(np.random.uniform(0, 0.6))
    angle = np.pi * np.random.uniform(0, 2)
    return np.array([length, angle])

def poincare_to_lorentz(y):
    term = 1 - y[0] * y[0] - y[1] * y[1]
    return [2 * y[0] / term,
            2 * y[1] / term,
            2 / term - 1]

def plot_wire_cube(p1, p2, color, ax):
    # Define the vertices of the cube
    vertices = np.array([
        p1,
        [p2[0], p1[1], p1[2]],
        [p2[0], p2[1], p1[2]],
        [p1[0], p2[1], p1[2]],
        [p1[0], p1[1], p2[2]],
        [p2[0], p1[1], p2[2]],
        p2,
        [p1[0], p2[1], p2[2]]
    ])

    # Define the edges of the cube
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        ax.plot3D(*zip(vertices[edge[0]], vertices[edge[1]]), color=color)

def compute_embedding(dataX):
    learning_rate = (dataX.shape[0] * 1) / (exaggeration_factor * 1000)

    opt_config = dict(
        learning_rate_ex=learning_rate,
        learning_rate_main=learning_rate,
        exaggeration=exaggeration_factor, 
        exaggeration_its=ex_iterations, 
        gradientDescent_its=main_iterations, 
        vanilla=False,  # if vanilla is set to true, regular gradient descent without any modifications is performed; for  vanilla set to false, the optimization makes use of momentum and gains
        momentum_ex=0.5,  # Set momentum during early exaggeration to 0.5
        momentum=0.8,  # Set momentum during non-exaggerated gradient descent to 0.8
        exact=False,  # To use the quad tree for acceleration (like Barnes-Hut in the Euclidean setting) or to evaluate the gradient exactly
        area_split=False,  # To build or not build the polar quad tree based on equal area splitting or - alternatively - on equal length splitting
        n_iter_check=10,  # Needed for early stopping criterion
        size_tol=0.999,  # Size of the embedding to be used as early stopping criterion
        hyperbolic_model=model
    )

    opt_params = SequentialOptimizer.sequence_poincare(**opt_config)

    print(f"config: {opt_config}")

    # Compute an initial embedding of the data via PCA
    X_embedded = initialization(
        n_samples=dataX.shape[0],
        n_components=2,
        X=dataX,
        random_state=seed,
        method="pca"
    )


    # Initialize the embedder
    htsne = HyperbolicTSNE(
        init=X_embedded, 
        n_components=2, 
        metric="precomputed", 
        verbose=True, 
        opt_method=SequentialOptimizer, 
        opt_params=opt_params
    )

    # Compute the embedding
    try:
        hyperbolicEmbedding = htsne.fit_transform((D, V))
    except ValueError:
        traceback.print_exc()

    return hyperbolicEmbedding

def plot_octree(points, qp_idx, ax):
    cart_points = np.array(points, dtype=np.float64)

    pqt = _OcTree(3, verbose=0)
    pqt.build_tree(cart_points)

    theta = 0.5

    idx, summary = pqt._py_summarize(cart_points[qp_idx], cart_points, angle=theta)

    sizes = []
    for j in range(idx // 4):
        size = summary[j * 4 + 2 + 1]
        sizes.append(int(size))

    summarized = set()
    display = set()
    mib = 0
    mab = 0

    cnt = 0
    for c_id, cell in enumerate(pqt.__getstate__()['cells']):
        # if cnt <= 3:
        #     display.add(c_id)
        # cnt += 1
        #
        # if c_id != 0 and cell['parent'] not in display:
        #     continue
        if cell['parent'] in summarized:
            summarized.add(c_id)
            continue

        min_bound = cell['min_bounds']
        max_bound = cell['max_bounds']
        barycenter = cell['barycenter']
        max_width = cell['squared_max_width']

        # print("Barycenter for cell [" + str(c_id) + "]:")
        print(barycenter)

        h_dist = distance_py(
            np.array(cart_points[qp_idx], dtype=ctypes.c_double), np.array(barycenter, dtype=ctypes.c_double)
        ) ** 2

        # print("DIST " + str(h_dist))
        if h_dist < MACHINE_EPSILON:
            continue
        ratio = (max_width / h_dist)
        is_summ = ratio < (theta ** 2)

        # print("MAX_WIDTH " + str(max_width) + ", RATIO " + str(ratio))

        if is_summ:
            summarized.add(c_id)
        elif cnt != 0:
            continue

        # print("DRAW FFS")
        ax.scatter([barycenter[1]], [barycenter[0]], [barycenter[2]], linewidth=0.5, marker='.', c="#253494", zorder=1, s=5)
        dif = ((np.floor(max_bound - min_bound) * 173) % 255) / 255

        # Swap axes, but for why ??
        min_bound[0], min_bound[1] = min_bound[1], min_bound[0]
        max_bound[0], max_bound[1] = max_bound[1], max_bound[0]
        plot_wire_cube(min_bound, max_bound, (dif[0], dif[1], 1.0), ax)

        if cnt == 0:
            mib = min_bound
            mab = max_bound
            cnt = 1
    return mib, mab

def plot_lorentz_embedding(points, labels, ax):
    lorentz_points = np.array([poincare_to_lorentz(p) for p in points])

    X = lorentz_points[:, 0]
    Y = lorentz_points[:, 1]
    Z = lorentz_points[:, 2]

    df = pd.DataFrame({"x": X, "y": Y, "z": Z, "l": labels})
    random_idx = np.random.randint(lorentz_points.shape[0])

    colormap = np.zeros(lorentz_points.shape[0])
    colormap[random_idx] = 1

    for s in np.unique(labels):
        ax.scatter(df.x[df.l == s], df.y[df.l == s], df.z[df.l == s], linewidth=0.5, marker='.', zorder=-10)
    
    ax.scatter(lorentz_points[random_idx, 0], lorentz_points[random_idx, 1], lorentz_points[random_idx, 2], marker='x', c='#E31A1C', zorder=10)

    v = 15.0
    lim = [-v, v]
    # ax.set_xlim(lim)
    # ax.set_ylim(lim)
    # ax.set_zlim([0.0, 2*v])

    return random_idx

def plot_hyperboloid(min_bounds, max_bounds, ax):
    # Make data.
    X = np.linspace(min_bounds[0], max_bounds[0], 30)
    Y = np.linspace(min_bounds[1], max_bounds[1], 30)
    X, Y = np.meshgrid(X, Y)
    Z = np.sqrt(X**2 + Y**2 + 1)

    # Plot the surface.
    ax.plot_surface(X, Y, Z, edgecolor='darkgreen', lw=0.5, rstride=8, cstride=8, alpha=0.05)

if __name__ == '__main__':
    # fig = plt.figure(figsize=plt.figaspect(2.0))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    # ax2 = fig.add_subplot(2, 1, 2)

    data_home = "datasets"

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

    X_embedded = compute_embedding(dataX)
    qp_idx = plot_lorentz_embedding(X_embedded, dataLabels, ax1)
    mib, mab = plot_octree(X_embedded, qp_idx, ax1)
    plot_hyperboloid(mib, mab, ax1)
    # plot_embedding(points, labels, ax2)

    plt.tight_layout()

    plt.savefig("teaser_files/c_elegans_octree.png")
    plt.show()
