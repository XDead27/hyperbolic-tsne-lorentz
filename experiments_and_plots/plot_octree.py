"""
This scripts takes an embedding of the C.Elegans data set and plots a polar quad tree on top of it.
"""
###########
# IMPORTS #
###########
import ctypes
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import pandas as pd
import seaborn as sns
from hyperbolicTSNE.hyperbolic_barnes_hut.lotsne import _OcTree
from hyperbolicTSNE.hyperbolic_barnes_hut.lotsne import distance_py
from hyperbolicTSNE import load_data, Datasets, initialization 

##############
# PLOT SETUP #
##############
MACHINE_EPSILON = np.finfo(np.double).eps
np.random.seed(594507)
matplotlib.rcParams['figure.dpi'] = 300
c = '#0173B2'  # Color for the tree
s = '-'  # style of the tree lines
w = 0.5  # width of the tree lines

c_elegans_palette = {
    'ABarpaaa_lineage': '#91003f',  # embryonic lineage
    'Germline': '#7f2704',
    # Somatic gonad precursor cell
    'Z1_Z4': '#800026',

    # Two embryonic hypodermal cells that may provide a scaffold for the early organization of ventral bodywall muscles
    'XXX': '#fb8072',

    'Ciliated_amphid_neuron': '#c51b8a', 'Ciliated_non_amphid_neuron': '#fa9fb5',

    # immune
    'Coelomocyte': '#ffff33', 'T': '#54278f',

    # Exceratory
    'Excretory_cell': '#004529',
    'Excretory_cell_parent': '#006837',
    'Excretory_duct_and_pore': '#238443',
    'Parent_of_exc_duct_pore_DB_1_3': '#41ab5d',
    'Excretory_gland': '#78c679',
    'Parent_of_exc_gland_AVK': '#addd8e',
    'Rectal_cell': '#d9f0a3',
    'Rectal_gland': '#f7fcb9',
    'Intestine': '#7fcdbb',

    # esophagus, crop, gizzard (usually) and intestine
    'Pharyngeal_gland': '#fed976',
    'Pharyngeal_intestinal_valve': '#feb24c',
    'Pharyngeal_marginal_cell': '#fd8d3c',
    'Pharyngeal_muscle': '#fc4e2a',
    'Pharyngeal_neuron': '#e31a1c',

    # hypodermis (epithelial)
    'Parent_of_hyp1V_and_ant_arc_V': '#a8ddb5',
    'hyp1V_and_ant_arc_V': '#ccebc5',
    'Hypodermis': '#253494',
    'Seam_cell': '#225ea8',
    'Arcade_cell': '#1d91c0',

    # set of six cells that form a thin cylindrical sheet between pharynx and ring neuropile
    'GLR': '#1f78b4',

    # Glia, also called glial cells or neuroglia, are non-neuronal cells in the central nervous system
    'Glia': '#377eb8',

    # head mesodermal cell: the middle layer of cells or tissues of an embryo
    'Body_wall_muscle': '#9e9ac8',
    'hmc': '#54278f',
    'hmc_and_homolog': '#02818a',
    'hmc_homolog': '#bcbddc',
    'Intestinal_and_rectal_muscle': '#41b6c4',
    # Postembryonic mesoblast: the mesoderm of an embryo in its earliest stages.
    'M_cell': '#3f007d',

    # pharyngeal gland cel
    'G2_and_W_blasts': '#abdda4',

    'unannotated': '#969696',
    'not provided': '#969696'
}


##################
# Helper Methods #
##################
def get_random_point():
    length = np.sqrt(np.random.uniform(0, 0.6))
    angle = np.pi * np.random.uniform(0, 2)
    return np.array([length, angle])


def cart_to_polar(p):
    length = np.sqrt(p[0] ** 2 + p[1] ** 2)
    angle = np.arctan2(p[1], p[0])
    angle = angle if angle > 0 else angle + 2 * np.pi
    return np.array([length, angle])


def cart_to_polar_2(p):
    radius = np.sqrt(p[0] * p[0] + p[1] * p[1])
    # Calculating angle (theta) in radian
    theta = np.arctan(p[1] / p[0])
    # Converting theta from radian to degree
    theta = 180 * theta / np.pi
    return np.array([radius, theta])


def cart2pol(p):
    x, y = p
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def poincare_to_lorentz(y):
    term = 1 - y[0] * y[0] - y[1] * y[1]
    return [2 * y[0] / term,
            2 * y[1] / term,
            2 / term - 1]

# def plot_tree(points, ax):
#     rticks, thetagrids = [], []
#
#     polar_points = np.array([cart_to_polar(p) for p in points])
#     cart_points = points
#
#     pqt = _QuadTree(cart_points.shape[1], verbose=0)
#     pqt.build_tree(cart_points)
#
#     theta = 0.5
#     random_idx = np.random.randint(polar_points.shape[0])
#
#     idx, summary = pqt._py_summarize(cart_points[random_idx], cart_points, angle=theta)
#     colormap = np.zeros(polar_points.shape[0])
#     colormap[random_idx] = 1
#
#     sizes = []
#     for j in range(idx // 4):
#         size = summary[j * 4 + 2 + 1]
#         sizes.append(int(size))
#
#     ax.scatter(polar_points[:, 1], polar_points[:, 0], linewidth=0.5, marker='.', c='lightgray', zorder=-10, s=2)
#     ax.scatter(polar_points[random_idx, 1], polar_points[random_idx, 0], marker='x', c='#E31A1C', zorder=10)
#
#     summarized = set()
#
#     for c_id, cell in enumerate(pqt.__getstate__()['cells']):
#
#         if cell['parent'] in summarized:
#             summarized.add(c_id)
#             continue
#
#         range_min = cell['min_bounds'][0]
#         range_max = cell['max_bounds'][0]
#         angle_min = cell['min_bounds'][1]
#         angle_max = cell['max_bounds'][1]
#         barycenter = cell['barycenter']
#         max_width = cell['squared_max_width']
#         polar_barycenter = cart_to_polar(barycenter)
#
#         h_dist = distance_py(
#             np.array(cart_points[random_idx], dtype=ctypes.c_double), np.array(barycenter, dtype=ctypes.c_double)
#         ) ** 2
#
#         if h_dist < MACHINE_EPSILON:
#             continue
#         ratio = (max_width / h_dist)
#         is_summ = ratio < (theta ** 2)
#
#         if is_summ:
#             summarized.add(c_id)
#         else:
#             continue
#
#         ax.scatter([polar_barycenter[1]], [polar_barycenter[0]], linewidth=0.5, marker='.', c="#253494", zorder=1, s=5)
#         ax.plot(
#             np.linspace(angle_min, angle_max, 100),
#             np.ones(100) * range_min,
#             color=c,
#             linestyle=s,
#             linewidth=w,
#             antialiased=True,
#             zorder=-1
#         )
#         ax.plot(
#             np.linspace(angle_min, angle_max, 100),
#             np.ones(100) * range_max,
#             color=c,
#             linestyle=s,
#             linewidth=w,
#             antialiased=True,
#             zorder=-1
#         )
#         ax.plot(
#             np.ones(100) * angle_min,
#             np.linspace(range_min, range_max, 100),
#             color=c,
#             linestyle=s,
#             linewidth=w,
#             antialiased=True,
#             zorder=-1
#         )
#         ax.plot(
#             np.ones(100) * angle_max,
#             np.linspace(range_min, range_max, 100),
#             color=c,
#             linestyle=s,
#             linewidth=w,
#             antialiased=True,
#             zorder=-1
#         )
#
#     ax.set_rmax(1)
#     ax.set_rticks(rticks)  # Less radial ticksz
#     ax.set_thetagrids(thetagrids)
#     ax.grid(True)

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

def plot_octree(points, qp_idx, ax):
    rticks, thetagrids = [], []

    cart_points = np.array(points, dtype=np.float128)

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
        else:
            continue

        # print("DRAW FFS")
        ax.scatter([barycenter[0]], [barycenter[1]], [barycenter[2]], linewidth=0.5, marker='.', c="#253494", zorder=1, s=5)
        dif = ((np.floor(max_bound - min_bound) * 173) % 255) / 255
        plot_wire_cube(min_bound, max_bound, (dif[0], dif[1], 1.0), ax)

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

def plot_hyperboloid(ax):
    # Make data.
    X = np.arange(-1.40976908e+16, 8.42840804e+15, 10000000000000)
    Y = np.arange(-16339744986096034.000000, 11265913305324462.000000, 100000000000000)
    X, Y = np.meshgrid(X, Y)
    Z = np.sqrt(X**2 + Y**2 + 1)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.7)

def plot_embedding(points, labels, ax):

    df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1]})

    point_size = 2
    font_size = 5
    alpha = 1.0

    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue=labels,
        hue_order=np.unique(labels),
        palette=c_elegans_palette,
        alpha=alpha,
        edgecolor="none",
        ax=ax,
        s=point_size,
        legend=False
    )

    circle = plt.Circle((0, 0), radius=1, fc='none', color='black')
    ax.add_patch(circle)
    ax.plot(0, 0, '.', c=(0, 0, 0), ms=4)

    # fig.tight_layout()
    ax.axis('off')
    ax.axis('equal')

    ax.set_ylim([-0.94, 0.94])
    ax.set_xlim([-0.94, 0.94])


if __name__ == '__main__':
    # fig = plt.figure(figsize=plt.figaspect(2.0))
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
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

    # Compute an initial embedding of the data via PCA
    X_embedded = initialization(
        n_samples=dataX.shape[0],
        n_components=2,
        X=dataX,
        random_state=seed,
        method="pca"
    )

    # points_simple = np.array([[0, 0],
    #                    [0.5, 0.5],
    #                    [-0.5, 0.5],
    #                    [-0.5, -0.5],
    #                    [0.5, -0.5]])
    # labels = np.load("../teaser_files/c_elegans_labels.npy", allow_pickle=True)

    qp_idx = plot_lorentz_embedding(X_embedded, dataLabels, ax1)
    # plot_hyperboloid(ax1)
    plot_octree(dataX, 0, ax1)
    # plot_embedding(points, labels, ax2)

    plt.tight_layout()

    plt.savefig("teaser_files/c_elegans_octree.png")
    plt.show()
