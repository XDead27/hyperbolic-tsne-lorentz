# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
#
# This file implements hyperbolic t-SNE components efficiently using Cython.
# The implementation is based on the tSNE code from Christopher Moody and 
# and Nick Travers available at https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_barnes_hut_tsne.pyx
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libc.math cimport sqrt, log, acosh, cosh, cos, sin, M_PI, atan2, tanh, atanh, isnan, fabs, fmin, fmax, sinh
from libc.stdlib cimport malloc, free, realloc
from cython.parallel cimport prange, parallel
from libc.string cimport memcpy
from libc.stdint cimport SIZE_MAX
from libc.math cimport isnan

np.import_array()

cdef char* EMPTY_STRING = ""

# Smallest strictly positive value that can be represented by floating
# point numbers for different precision levels. This is useful to avoid
# taking the log of zero when computing the KL divergence.
cdef float FLOAT32_TINY = np.finfo(np.float32).tiny

# Useful to void division by zero or divergence to +inf.
cdef float FLOAT64_EPS = np.finfo(np.float64).eps
cdef float FLOAT128_EPS = np.finfo(np.float128).eps

cdef double EPSILON = 1e-5
cdef double MAX_TANH = 15.0
cdef double BOUNDARY = 1 - EPSILON
cdef int LORENTZ_T = 2
cdef int LORENTZ_X_1 = 0
cdef int LORENTZ_X_2 = 1
cdef double MACHINE_EPSILON = np.finfo(np.double).eps
cdef int TAKE_TIMING = 1
cdef int AREA_SPLIT = 0
cdef int GRAD_FIX = 1

cdef double clamp(double n, double lower, double upper) nogil:
    cdef double t = lower if n < lower else n
    return upper if t > upper else t

##################################################
# OcTree
##################################################
ctypedef np.npy_float64 DTYPE_t          # Type of X
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef struct Cell:
    # Base storage structure for cells in a OcTree object

    # Tree structure
    SIZE_t parent              # Parent cell of this cell
    SIZE_t[8] children         # Array pointing to children of this cell

    # Cell description
    SIZE_t cell_id             # Id of the cell in the cells array in the Tree
    SIZE_t point_index         # Index of the point at this cell (only defined
                               # in non empty leaf)
    bint is_leaf               # Does this cell have children?
    DTYPE_t squared_max_width  # Squared value of the maximum width w
    SIZE_t depth               # Depth of the cell in the tree
    SIZE_t cumulative_size     # Number of points included in the subtree with
                               # this cell as a root.

    # Internal constants
    DTYPE_t[3] center          # Store the center for quick split of cells
    DTYPE_t[3] barycenter      # Keep track of the center of mass of the cell
    DTYPE_t lorentz_factor_sum

    # Cell boundaries
    DTYPE_t[3] min_bounds      # Inferior boundaries of this cell (inclusive)
    DTYPE_t[3] max_bounds      # Superior boundaries of this cell (exclusive)

# Build the corresponding numpy dtype for Cell.
# This works by casting `dummy` to an array of Cell of length 1, which numpy
# can construct a `dtype`-object for. See https://stackoverflow.com/q/62448946
# for a more detailed explanation.
cdef Cell dummy;
CELL_DTYPE = np.asarray(<Cell[:1]>(&dummy)).dtype

assert CELL_DTYPE.itemsize == sizeof(Cell)

ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)
    (Cell*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        with gil:
            raise MemoryError("could not allocate (%d * %d) bytes"
                              % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        with gil:
            raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  # for convenience

# This is effectively an ifdef statement in Cython
# It allows us to write printf debugging lines
# and remove them at compile time
cdef enum:
    DEBUGFLAG = 0

cdef extern from "time.h":
    # Declare only what is necessary from `tm` structure.
    ctypedef long clock_t
    clock_t clock() nogil
    double CLOCKS_PER_SEC


from cpython cimport Py_INCREF, PyObject, PyTypeObject

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(np.ndarray arr, PyObject* obj)

cdef DTYPE_t sq_norm(DTYPE_t x, DTYPE_t y) nogil:
    return x * x + y * y

cdef void poincare_to_lorentz(DTYPE_t y1, DTYPE_t y2, DTYPE_t* result) nogil:
    cdef:
        DTYPE_t term = 1 - y1 * y1 - y2 * y2

    result[LORENTZ_T] = 2 / term - 1
    result[LORENTZ_X_1] = 2 * y1 / term
    result[LORENTZ_X_2] = 2 * y2 / term

cdef void lorentz_to_poincare(DTYPE_t* lp, DTYPE_t* result) nogil:
    result[0] = lp[LORENTZ_X_1] / (1 + lp[LORENTZ_T])
    result[1] = lp[LORENTZ_X_2] / (1 + lp[LORENTZ_T])

cdef void lorentz_to_klein(DTYPE_t* lp, DTYPE_t* result) nogil:
    result[0] = lp[LORENTZ_X_1] / lp[LORENTZ_T]
    result[1] = lp[LORENTZ_X_2] / lp[LORENTZ_T]

cdef void klein_to_lorentz(DTYPE_t* z, DTYPE_t* result) nogil:
    cdef DTYPE_t term = sqrt(1 - z[0] * z[0] - z[1] * z[1])

    result[LORENTZ_T] = 1 / term
    result[LORENTZ_X_1] = z[0] / term
    result[LORENTZ_X_2] = z[1] / term

cdef DTYPE_t lorentz_factor(DTYPE_t sq_n) nogil:
    cdef DTYPE_t x = sqrt(1 - sq_n)
    
    x = max(x, FLOAT64_EPS)
    return 1 / x

cdef DTYPE_t minkowski_bilinear(DTYPE_t[3] lp1, DTYPE_t[3] lp2) nogil:
    return lp1[LORENTZ_X_1] * lp2[LORENTZ_X_1] + lp1[LORENTZ_X_2] * lp2[LORENTZ_X_2] - lp1[LORENTZ_T] * lp2[LORENTZ_T]

# Checks if value lies in the upper polynomial and is in range 0, 1
cdef bint _check_dist_param_value(DTYPE_t val, DTYPE_t z0, DTYPE_t z_d) nogil:
    cdef DTYPE_t t = z0 + val * z_d

    return t >= 1. and val <= 1. and val >= 0

cdef void _get_point_param(DTYPE_t param, DTYPE_t[3] v_0, DTYPE_t[3] v_d, DTYPE_t* result) nogil:
    result[LORENTZ_X_1] = v_0[0] + v_d[0] * param
    result[LORENTZ_X_2] = v_0[1] + v_d[1] * param
    result[LORENTZ_T] = v_0[2] + v_d[2] * param

# Get intersection between line described by two points and the hyperboloid
cdef DTYPE_t _get_line_hyperboloid_intersection(DTYPE_t[3] la, DTYPE_t[3] lb, DTYPE_t* res1, DTYPE_t* res2, int* cnts) nogil:
    cdef:
        DTYPE_t[3] v_d
        DTYPE_t[3] v_0
        DTYPE_t c_a, c_b, c_c, delta, w0, w1, aux0, aux1, aux2
    
    # Setup variables
    cnts[0] = 0
    cnts[1] = 0
    v_0[0] = la[LORENTZ_X_1]
    v_0[1] = la[LORENTZ_X_2]
    v_0[2] = la[LORENTZ_T]
    v_d[0] = lb[LORENTZ_X_1] - v_0[0]
    v_d[1] = lb[LORENTZ_X_2] - v_0[1]
    v_d[2] = lb[LORENTZ_T] - v_0[2]
    c_a = v_d[2] * v_d[2] - v_d[0] * v_d[0] - v_d[1] * v_d[1]
    c_b = 2 * (v_d[2] * v_0[2] - v_d[0] * v_0[0] - v_d[1] * v_0[1])
    c_c = v_0[2] * v_0[2] - v_0[0] * v_0[0] - v_0[1] * v_0[1] - 1
    delta = c_b * c_b - 4 * c_a * c_c
    
    # Get intersection points with hyperboloid
    if delta < 0:
        return 0

    aux0 = 2 * c_a
    aux1 = c_b / aux0
    aux2 = sqrt(delta) / aux0
    w0 = aux2 - aux1
    w1 = -aux1 - aux2

    # Select only those which lie on the segment of the cube
    if _check_dist_param_value(w0, v_0[2], v_d[2]):
        _get_point_param(w0, v_0, v_d, res1)
        cnts[0] = 1

    if delta > 0 and _check_dist_param_value(w1, v_0[2], v_d[2]):
        _get_point_param(w1, v_0, v_d, res2)
        cnts[1] = 1

    return w0

# Copy point 
cdef void _copy_point(DTYPE_t[3] p, DTYPE_t* res) nogil:
    res[0] = p[0]
    res[1] = p[1]
    res[2] = p[2]

cdef DTYPE_t get_max_dist_hyperboloid_sect(DTYPE_t[3] la, DTYPE_t[3] lb) nogil:
    cdef DTYPE_t[8][3] points
    cdef DTYPE_t[24][3] intersect
    cdef DTYPE_t max_dist = 0.0, dist
    cdef int[24] cnts


    # Setup points (too lazy to think of algorithm)
    _copy_point(la, points[0]) # min bound
    _copy_point(la, points[1])
    _copy_point(la, points[3])
    _copy_point(la, points[4])
    _copy_point(lb, points[2])
    _copy_point(lb, points[5])
    _copy_point(lb, points[6]) # max bound
    _copy_point(lb, points[7])
    points[1][LORENTZ_X_1] = lb[LORENTZ_X_1]
    points[3][LORENTZ_X_2] = lb[LORENTZ_X_2]
    points[4][LORENTZ_T] = lb[LORENTZ_T]
    points[2][LORENTZ_T] = la[LORENTZ_T]
    points[5][LORENTZ_X_2] = la[LORENTZ_X_2]
    points[7][LORENTZ_X_1] = la[LORENTZ_X_1]

    # Get (max 24) intersection points with hyperboloid
    _get_line_hyperboloid_intersection(points[0], points[1], intersect[0], intersect[1], &cnts[0])
    _get_line_hyperboloid_intersection(points[0], points[3], intersect[2], intersect[3], &cnts[2])
    _get_line_hyperboloid_intersection(points[0], points[4], intersect[4], intersect[5], &cnts[4])
    _get_line_hyperboloid_intersection(points[6], points[5], intersect[6], intersect[7], &cnts[6])
    _get_line_hyperboloid_intersection(points[6], points[7], intersect[8], intersect[9], &cnts[8])
    _get_line_hyperboloid_intersection(points[6], points[2], intersect[10], intersect[11], &cnts[10])
    _get_line_hyperboloid_intersection(points[1], points[2], intersect[12], intersect[13], &cnts[12])
    _get_line_hyperboloid_intersection(points[1], points[5], intersect[14], intersect[15], &cnts[14])
    _get_line_hyperboloid_intersection(points[3], points[2], intersect[16], intersect[17], &cnts[16])
    _get_line_hyperboloid_intersection(points[3], points[7], intersect[18], intersect[19], &cnts[18])
    _get_line_hyperboloid_intersection(points[4], points[5], intersect[20], intersect[21], &cnts[20])
    _get_line_hyperboloid_intersection(points[4], points[7], intersect[22], intersect[23], &cnts[22])

    # Compute pairwise (lorentz) distance between the selected points
    for i in range(24):
        if cnts[i] == 0:
            continue
        for j in range(i, 24):
            if cnts[j] == 0:
                continue
            dist = distance_lorentz(intersect[i], intersect[j])
            if dist > max_dist:
                max_dist = dist

    return max_dist

cdef class _OcTree:
    """Array-based representation of a OcTree.

    This class is currently working for indexing 2D data (regular QuadTree) and
    for indexing 3D data (OcTree). It is planned to split the 2 implementations
    using `Cython.Tempita` to save some memory for QuadTree.

    Note that this code is currently internally used only by the Barnes-Hut
    method in `sklearn.manifold.TSNE`. It is planned to be refactored and
    generalized in the future to be compatible with nearest neighbors API of
    `sklearn.neighbors` with 2D and 3D data.
    """

    # Parameters of the tree
    cdef public int n_dimensions         # Number of dimensions in X
    cdef public int verbose              # Verbosity of the output
    cdef SIZE_t n_cells_per_cell         # Number of children per node. (2 ** n_dimension)

    # Tree inner structure
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t cell_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef public SIZE_t n_points          # Total number of points
    cdef Cell* cells                     # Array of nodes

    def __cinit__(self, int n_dimensions, int verbose):
        """Constructor."""
        # Parameters of the tree
        self.n_dimensions = n_dimensions
        self.verbose = verbose
        self.n_cells_per_cell = 2 ** self.n_dimensions

        # Inner structures
        self.max_depth = 0
        self.cell_count = 0
        self.capacity = 0
        self.n_points = 0
        self.cells = NULL

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.cells)

    property cumulative_size:
        def __get__(self):
            return self._get_cell_ndarray()['cumulative_size'][:self.cell_count]

    property leafs:
        def __get__(self):
            return self._get_cell_ndarray()['is_leaf'][:self.cell_count]

    def build_tree(self, X):
        """
        Build a tree from an array of points X.
        X is Lorentz coordinates (X_1, X_2, X_T)
        """
        cdef:
            int i
            DTYPE_t[3] pt
            DTYPE_t[3] min_bounds, max_bounds

        n_samples = X.shape[0]
        # LX = np.zeros((n_samples, 3))
        
        # # Change points from Poincare to Lorentz
        # for i in range(n_samples):
        #     poincare_to_lorentz(X[i, 0], X[i, 1], pt)
        #     LX[i, LORENTZ_T] = pt[LORENTZ_T]
        #     LX[i, LORENTZ_X_1] = pt[LORENTZ_X_1]
        #     LX[i, LORENTZ_X_2] = pt[LORENTZ_X_2]

        capacity = 100
        self._resize(capacity)
        m = np.min(X, axis=0)
        M = np.max(X, axis=0)
        # Scale the maximum to get all points strictly in the tree bounding box
        # The 3 bounds are for positive, negative and small values
        M = np.maximum(M * (1. + 1e-3 * np.sign(M)), M + 1e-3)
        for i in range(self.n_dimensions):
            min_bounds[i] = m[i]
            max_bounds[i] = M[i]

            if self.verbose > 10:
                printf("[OcTree] bounding box axis %i : [%f, %f]\n",
                       i, min_bounds[i], max_bounds[i])

        # Create the initial node with boundaries from the dataset
        self._init_root(min_bounds, max_bounds)

        # Insert all points
        for i in range(n_samples):
            for j in range(self.n_dimensions):
                pt[j] = X[i, j]

            self.insert_point(pt, i)

        # Shrink the cells array to reduce memory usage
        self._resize(capacity=self.cell_count)
    
    cdef int insert_point(self, DTYPE_t[3] point, SIZE_t point_index,
                      SIZE_t cell_id=0) except -1 nogil:
        """Insert a point in the QuadTree."""
        cdef int ax
        cdef SIZE_t selected_child
        cdef Cell* cell = &self.cells[cell_id]
        cdef SIZE_t n_point = cell.cumulative_size
        cdef DTYPE_t[2] klein_point
        cdef DTYPE_t[2] klein_barycenter
        cdef DTYPE_t temp_norm
        cdef DTYPE_t temp_lorentz

        if self.verbose > 10:
            printf("[OcTree] Inserting depth %li\n", cell.depth)

        # If the cell is an empty leaf, insert the point in it
        if cell.cumulative_size == 0:
            cell.cumulative_size = 1
            self.n_points += 1
            for i in range(self.n_dimensions):
                cell.barycenter[i] = point[i]
            cell.point_index = point_index

            lorentz_to_klein(point, klein_point)

            temp_norm = sq_norm(klein_point[0],
                                klein_point[1])
            temp_lorentz = lorentz_factor(temp_norm)
            cell.lorentz_factor_sum = temp_lorentz

            if self.verbose > 10:
                printf("[OcTree] inserted point %li in cell %li\n",
                       point_index, cell_id)
            return cell_id

        # If the cell is not a leaf, update cell internals and
        # recurse in selected child
        if not cell.is_leaf:
            # Recompute barycenter of cell
            lorentz_to_klein(point, klein_point)

            temp_norm = sq_norm(klein_point[0],
                                klein_point[1])

            if temp_norm >= 1.:
                printf("We kinda have an issue...\n")
            temp_lorentz = lorentz_factor(temp_norm)
            if isnan(temp_lorentz):
                printf("We have enother issue...\n")

            lorentz_to_klein(cell.barycenter, klein_barycenter)

            klein_barycenter[0] = (klein_barycenter[0] * cell.lorentz_factor_sum + temp_lorentz * klein_point[0]) / (cell.lorentz_factor_sum + temp_lorentz)
            klein_barycenter[1] = (klein_barycenter[1] * cell.lorentz_factor_sum + temp_lorentz * klein_point[1]) / (cell.lorentz_factor_sum + temp_lorentz)

            klein_to_lorentz(klein_barycenter, cell.barycenter)

            cell.lorentz_factor_sum += temp_lorentz

            # Increase the size of the subtree starting from this cell
            cell.cumulative_size += 1

            # Insert child in the correct subtree
            selected_child = self._select_child(point, cell)
            if self.verbose > 49:
                printf("[OcTree] selected child %li\n", selected_child)
            if selected_child == -1:
                self.n_points += 1
                return self._insert_point_in_new_child(point, cell, point_index)
            return self.insert_point(point, point_index, selected_child)

        # Finally, if the cell is a leaf with a point already inserted,
        # split the cell in n_cells_per_cell if the point is not a duplicate.
        # If it is a duplicate, increase the size of the leaf and return.
        if self._is_duplicate(point, cell.barycenter):
            if self.verbose > 10:
                printf("[OcTree] found a duplicate!\n")
            cell.cumulative_size += 1
            self.n_points += 1
            return cell_id

        if self.verbose > 49:
            printf("[OcTree] Inserting %li in leaf!\n", point_index)

        # In a leaf, the barycenter correspond to the only point included
        # in it.
        self._insert_point_in_new_child(cell.barycenter, cell, cell.point_index,
                                        cell.cumulative_size)
        return self.insert_point(point, point_index, cell_id)

    # XXX: This operation is not Thread safe
    cdef SIZE_t _insert_point_in_new_child(
        self, DTYPE_t[3] point, Cell* cell, SIZE_t point_index, SIZE_t size=1
    ) noexcept nogil:
        """Create a child of cell which will contain point."""

        # Local variable definition
        cdef:
            SIZE_t cell_id, cell_child_id, parent_id
            DTYPE_t[3] save_point
            DTYPE_t width
            Cell* child
            int i
            DTYPE_t[2] klein_barycenter
            DTYPE_t temp_norm

        # If the maximal capacity of the Tree have been reached, double the capacity
        # We need to save the current cell id and the current point to retrieve them
        # in case the reallocation
        if self.cell_count + 1 > self.capacity:
            parent_id = cell.cell_id
            for i in range(self.n_dimensions):
                save_point[i] = point[i]
            self._resize(SIZE_MAX)
            cell = &self.cells[parent_id]
            point = save_point

        # Get an empty cell and initialize it
        cell_id = self.cell_count
        self.cell_count += 1
        child = &self.cells[cell_id]

        self._init_cell(child, cell.cell_id, cell.depth + 1)
        child.cell_id = cell_id

        # Set the cell as an inner cell of the Tree
        cell.is_leaf = False
        cell.point_index = -1

        # Set the correct boundary for the cell, store the point in the cell
        # and compute its index in the children array.
        cell_child_id = 0
        for i in range(self.n_dimensions):
            cell_child_id *= 2
            if point[i] >= cell.center[i]:
                cell_child_id += 1
                child.min_bounds[i] = cell.center[i]
                child.max_bounds[i] = cell.max_bounds[i]
            else:
                child.min_bounds[i] = cell.min_bounds[i]
                child.max_bounds[i] = cell.center[i]
            child.center[i] = (child.min_bounds[i] + child.max_bounds[i]) / 2.

            child.barycenter[i] = point[i]

        # Compute lorentz factor sum (TODO get rid of it)
        lorentz_to_klein(child.barycenter, klein_barycenter)
        temp_norm = sq_norm(klein_barycenter[0], klein_barycenter[1])
        child.lorentz_factor_sum = lorentz_factor(temp_norm)

        # Compute the maximum squared distance by intersecting with hyperboloid
        width = get_max_dist_hyperboloid_sect(child.min_bounds, child.max_bounds)
        child.squared_max_width = width * width

        # Store the point info and the size to account for duplicated points
        child.point_index = point_index
        child.cumulative_size = size

        # Store the child cell in the correct place in children
        cell.children[cell_child_id] = child.cell_id

        if self.verbose > 10:
            printf("[OcTree] inserted point %li in new child %li\n",
                   point_index, cell_id)

        return cell_id

    cdef bint _is_duplicate(self, DTYPE_t[3] point1, DTYPE_t[3] point2) noexcept nogil:
        """Check if the two given points are equals."""
        cdef int i
        cdef bint res = True
        for i in range(self.n_dimensions):
            # Use EPSILON to avoid numerical error that would overgrow the tree
            res &= fabs(point1[i] - point2[i]) <= EPSILON
        return res

    cdef SIZE_t _select_child(self, DTYPE_t[3] point, Cell* cell) noexcept nogil:
        """Select the child of cell which contains the given query point."""
        cdef:
            int i
            SIZE_t selected_child = 0

        for i in range(self.n_dimensions):
            # Select the correct child cell to insert the point by comparing
            # it to the borders of the cells using precomputed center.
            selected_child *= 2
            if point[i] >= cell.center[i]:
                selected_child += 1
        return cell.children[selected_child]

    cdef void _init_cell(self, Cell* cell, SIZE_t parent, SIZE_t depth) nogil:
        """Initialize a cell structure with some constants."""
        cell.parent = parent
        cell.is_leaf = True
        cell.depth = depth
        cell.squared_max_width = 0
        cell.cumulative_size = 0
        for i in range(self.n_cells_per_cell):
            cell.children[i] = SIZE_MAX

    cdef void _init_root(self, DTYPE_t[3] min_bounds, DTYPE_t[3] max_bounds
                         ) noexcept nogil:
        """Initialize the root node with the given space boundaries"""
        cdef:
            int i
            DTYPE_t width
            Cell* root = &self.cells[0]

        self._init_cell(root, -1, 0)
        for i in range(self.n_dimensions):
            root.min_bounds[i] = min_bounds[i]
            root.max_bounds[i] = max_bounds[i]
            root.center[i] = (max_bounds[i] + min_bounds[i]) / 2.

        # Compute maximum squared distance of the root by intersecting with the hyperboloid
        width = get_max_dist_hyperboloid_sect(root.min_bounds, root.max_bounds)
        root.squared_max_width = max(root.squared_max_width, width*width)
        # printf("[OcTree] Root max width is: %f\n", width)

        root.cell_id = 0

        self.cell_count += 1

    cdef long summarize(self, DTYPE_t[3] point, DTYPE_t* results,
                        float squared_theta=.5, SIZE_t cell_id=0, long idx=0
                        ) noexcept nogil:
        """Summarize the tree compared to a query point.

        Input arguments
        ---------------
        point : array (n_dimensions)
             query point to construct the summary.
        cell_id : integer, optional (default: 0)
            current cell of the tree summarized. This should be set to 0 for
            external calls.
        idx : integer, optional (default: 0)
            current index in the result array. This should be set to 0 for
            external calls
        squared_theta: float, optional (default: .5)
            threshold to decide whether the node is sufficiently far
            from the query point to be a good summary. The formula is such that
            the node is a summary if
                node_width^2 / dist_node_point^2 < squared_theta.
            Note that the argument should be passed as theta^2 to avoid
            computing square roots of the distances.

        Output arguments
        ----------------
        results : array (n_samples * (n_dimensions+2))
            result will contain a summary of the tree information compared to
            the query point:
            - results[idx:idx+n_dimensions] contains the coordinate-wise
                difference between the query point and the summary cell idx.
                This is useful in t-SNE to compute the negative forces.
            - result[idx+n_dimensions+1] contains the squared euclidean
                distance to the summary cell idx.
            - result[idx+n_dimensions+2] contains the number of point of the
                tree contained in the summary cell idx.

        Return
        ------
        idx : integer
            number of elements in the results array.
        """
        cdef:
            int i, idx_d = idx + self.n_dimensions 
            bint duplicate = True
            DTYPE_t dist
            Cell* cell = &self.cells[cell_id]
            DTYPE_t[2] poincare_barycenter

        # TODO: Either
        # (1) Change signature to receive Poincare point, then, for 
        #     each point in the tree, translate to Poincare model and
        #     use distance_grad() (easy, not clean)
        # (2) Preserve signature, change both the argument point and all
        #     the ones contained in the tree to Poincare model and use
        #     distance_grad() (cleaner, but has to respect other constraints)
        # (3) (NON-PRIORITY) Preserve signature and use hyperboloid 
        #     gradient and distance methods to summarize (cleaneast, hardest) (CHOSEN)
        
        # lorentz_to_poincare(cell.barycenter, poincare_barycenter)

        results[idx_d] = 0.
        distance_grad_q(point, cell.barycenter, &results[idx])
        for i in range(self.n_dimensions):
            duplicate &= fabs(results[idx + i]) <= EPSILON

        dist = distance_q(point, cell.barycenter)
        results[idx_d] = dist * dist

        # Do not compute self interactions
        if duplicate and cell.is_leaf:
            return idx

        # Check whether we can use this node as a summary
        # It's a summary node if the angular size as measured from the point
        # is relatively small (w.r.t. theta) or if it is a leaf node.
        # If it can be summarized, we use the cell center of mass
        # Otherwise, we go a higher level of resolution and into the leaves.
        if cell.is_leaf or (
                (cell.squared_max_width / results[idx_d]) < squared_theta):
            results[idx_d + 1] = <DTYPE_t> cell.cumulative_size
            return idx + self.n_dimensions + 2

        else:
            # Recursively compute the summary in nodes
            for c in range(self.n_cells_per_cell):
                child_id = cell.children[c]
                if child_id != -1:
                    idx = self.summarize(point, results, squared_theta,
                                         child_id, idx)

        return idx

    cdef np.ndarray _get_cell_ndarray(self):
        """Wraps nodes as a NumPy struct array.
    
        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.cell_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Cell)
        cdef np.ndarray arr
        Py_INCREF(CELL_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray,
                                   CELL_DTYPE, 1, shape,
                                   strides, <void*> self.cells,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array!")
        return arr
    
    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.
    
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, SIZE_t capacity=SIZE_MAX) nogil except -1:
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.cells != NULL:
            return 0
    
        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 9  # default initial value to min
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.cells, capacity)

        # if capacity smaller than cell_count, adjust the counter
        if capacity < self.cell_count:
            self.cell_count = capacity
    
        self.capacity = capacity
        return 0

    def get_cell(self, point):
        """return the id of the cell containing the query point or raise
        ValueError if the point is not in the tree
        """
        cdef DTYPE_t[3] query_pt
        cdef int i

        assert len(point) == self.n_dimensions, (
            "Query point should be a point in dimension {}."
            .format(self.n_dimensions))

        for i in range(self.n_dimensions):
            query_pt[i] = point[i]

        return self._get_cell(query_pt, 0)

    cdef int _get_cell(self, DTYPE_t[3] point, SIZE_t cell_id=0
                       ) except -1 nogil:
        """guts of get_cell.

        Return the id of the cell containing the query point or raise ValueError
        if the point is not in the tree"""
        cdef:
            SIZE_t selected_child
            Cell* cell = &self.cells[cell_id]

        if cell.is_leaf:
            if self._is_duplicate(cell.barycenter, point):
                if self.verbose > 99:
                    printf("[OcTree] Found point in cell: %li\n",
                           cell.cell_id)
                return cell_id
            with gil:
                raise ValueError("Query point not in the Tree.")

        selected_child = self._select_child(point, cell)
        if selected_child > 0:
            if self.verbose > 99:
                printf("[OcTree] Selected_child: %li\n", selected_child)
            return self._get_cell(point, selected_child)
        with gil:
            raise ValueError("Query point not in the Tree.")

    # Pickling primitives

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (_OcTree, (self.n_dimensions, self.verbose), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["cell_count"] = self.cell_count
        d["capacity"] = self.capacity
        d["n_points"] = self.n_points
        d["cells"] = self._get_cell_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.cell_count = d["cell_count"]
        self.capacity = d["capacity"]
        self.n_points = d["n_points"]

        if 'cells' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        cell_ndarray = d['cells']

        if (cell_ndarray.ndim != 1 or
                cell_ndarray.dtype != CELL_DTYPE or
                not cell_ndarray.flags.c_contiguous):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = cell_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)

        cdef Cell[:] cell_mem_view = cell_ndarray
        memcpy(
            pto=self.cells,
            pfrom=&cell_mem_view[0],
            size=self.capacity * sizeof(Cell),
        )
    def _py_summarize(self, DTYPE_t[:] query_pt, DTYPE_t[:, :] X, float angle):
        # Used for testing summarize
        cdef:
            DTYPE_t[:] summary
            int n_samples

        n_samples = X.shape[0]
        summary = np.empty(4 * n_samples, dtype=np.float64)

        idx = self.summarize(&query_pt[0], &summary[0], angle * angle)
        return idx, summary

#################################################
# Dist and Dist Grad functions
#################################################
# NON-PRIORITY TODO: Change these to the Lorentz model distance and gradient
cdef DTYPE_t distance_q(DTYPE_t* u, DTYPE_t* v) nogil:
    return distance_lorentz(u, v)

# Distance gradient on the poincare model (takes two pointers as arguments)
cdef void distance_grad_q(DTYPE_t* u, DTYPE_t* v, DTYPE_t* res) nogil:
    distance_grad_lorentz(u, v, res)

# Pointwise distance on the poincare model
cpdef DTYPE_t distance(DTYPE_t u0, DTYPE_t u1, DTYPE_t v0, DTYPE_t v1) nogil:
    if fabs(u0 - v0) <= EPSILON and fabs(u1 - v1) <= EPSILON:
        return 0.

    cdef:
        DTYPE_t uv2 = ((u0 - v0) * (u0 - v0)) + ((u1 - v1) * (u1 - v1))
        DTYPE_t u_sq = clamp(u0 * u0 + u1 * u1, 0, BOUNDARY)
        DTYPE_t v_sq = clamp(v0 * v0 + v1 * v1, 0, BOUNDARY)
        DTYPE_t alpha = 1. - u_sq
        DTYPE_t beta = 1. - v_sq
        DTYPE_t result = acosh( 1. + 2. * uv2 / ( alpha * beta ) )

    return result

# Distance function for the hyperboloid model
cdef DTYPE_t distance_lorentz(DTYPE_t[3] lp1, DTYPE_t[3] lp2) nogil:
    return acosh( - minkowski_bilinear(lp1, lp2))

# Distance gradient on the poincare model
cdef DTYPE_t distance_grad(DTYPE_t u0, DTYPE_t u1, DTYPE_t v0, DTYPE_t v1, int ax) nogil:
    if fabs(u0 - v0) <= EPSILON and fabs(u1 - v1) <= EPSILON:
        return 0.

    cdef:
        DTYPE_t a = u0 - v0
        DTYPE_t b = u1 - v1
        DTYPE_t uv2 = a * a + b * b

        DTYPE_t u_sq = clamp(u0 * u0 + u1 * u1, 0, BOUNDARY)
        DTYPE_t v_sq = clamp(v0 * v0 + v1 * v1, 0, BOUNDARY)
        DTYPE_t alpha = 1. - u_sq
        DTYPE_t beta = 1. - v_sq

        DTYPE_t gamma = 1. + (2. / (alpha * beta)) * uv2
        DTYPE_t shared_scalar = 4. / fmax(beta * sqrt((gamma * gamma) - 1.), MACHINE_EPSILON)

        DTYPE_t u_scalar = (v_sq - 2. * (u0 * v0 + u1 * v1) + 1.) / (alpha * alpha)
        DTYPE_t v_scalar = 1. / alpha

    if ax == 0:
        return shared_scalar * (u_scalar * u0 - v_scalar * v0)
    else:
        return shared_scalar * (u_scalar * u1 - v_scalar * v1)

# Distance gradient on the lorentz model, with respect to u 
cdef void distance_grad_lorentz(DTYPE_t[3] u, DTYPE_t[3] v, DTYPE_t* res) nogil:
    cdef:
        DTYPE_t minkbil = minkowski_bilinear(u, v)
        DTYPE_t scalar = - 1 / sqrt(minkbil * minkbil - 1)

    for i in range(3):
        res[i] = scalar * v[i]

# Project the gradient on the tangent space at point p on the hyperboloid
cdef void project_to_tangent_space(DTYPE_t[3] p, DTYPE_t[3] grad, DTYPE_t* res) nogil:
    cdef DTYPE_t minkbil = minkowski_bilinear(p, grad)

    res[0] = grad[0] + p[0] * minkbil
    res[1] = grad[1] + p[1] * minkbil
    res[2] = grad[2] + p[2] * minkbil

cdef void exp_map_single(double* x, double* v, double* res) nogil:
    cdef double x_norm_sq, metric, v_norm, v_scalar
    cdef double* y = <double*> malloc(sizeof(double) * 2)

    x_norm_sq = clamp(x[0] ** 2 + x[1] ** 2, 0, BOUNDARY)

    metric = 2. / (1. - x_norm_sq)
    v_norm = sqrt(v[0] ** 2 + v[1] ** 2)

    v_scalar = tanh(clamp((metric * v_norm) / 2., -MAX_TANH, MAX_TANH))

    for j in range(2):
        y[j] = (v[j] / v_norm) * v_scalar

    mobius_addition(x, y, res)
    free(y)

# Projects a vector from the tangent space back on the hyperboloid
cdef void exp_map_single_lorentz(DTYPE_t[3] p, DTYPE_t[3] v, DTYPE_t* res) nogil:
    cdef DTYPE_t norm = sqrt(minkowski_bilinear(v, v))
    cdef DTYPE_t coshval = cosh(norm)
    cdef DTYPE_t sinhval = sinh(norm)

    for i in range(3):
        res[i] = coshval * p[i] + sinhval * v[i] / norm

#TODO: this method gets called outside
cpdef void exp_map(double[:, :] x, double[:, :] v, double[:, :] out, int num_threads) nogil:
    cdef double* exp_map_res = <double*> malloc(sizeof(double) * 2)

    for i in range(x.shape[0]):
        exp_map_single(&x[i, 0], &v[i, 0], exp_map_res)

        for j in range(2):
            out[i, j] = exp_map_res[j]

    free(exp_map_res)

# Applies exp_map_single_lorentz for all points in p and vectors in v
cpdef void exp_map_lorentz(DTYPE_t[:, :] p, DTYPE_t[:, :] v, DTYPE_t[:, :] res, int num_threads) nogil:
    cdef DTYPE_t* exp_map_res = <DTYPE_t*> malloc(sizeof(DTYPE_t) * 3)

    for i in range(p.shape[0]):
        exp_map_single_lorentz(&p[i, 0], &v[i, 0], exp_map_res)

        for j in range(3):
            res[i, j] = exp_map_res[j]

    free(exp_map_res)

cdef void mobius_addition(double* x, double* y, double* res) nogil:
    cdef double y_norm_sq, x_norm_sq, x_scalar, y_scalar, r_term, denominator

    x_norm_sq = clamp(x[0] ** 2 + x[1] ** 2, 0, BOUNDARY)
    y_norm_sq = y[0] ** 2 + y[1] ** 2

    r_term = 1. + 2. * (x[0] * y[0] + x[1] * y[1])

    x_scalar = (r_term + y_norm_sq)
    y_scalar = (1. - x_norm_sq)

    denominator = r_term + x_norm_sq * y_norm_sq

    for i in range(2):
        res[i] = (x_scalar * x[i] + y_scalar * y[i]) / denominator

cdef void log_map_single(double* x, double* y, double* res) nogil:
    cdef double x_norm_sq, metric, y_scalar

    x_norm_sq = clamp(x[0] ** 2 + x[1] ** 2, 0, BOUNDARY)

    metric = 2. / (1. - x_norm_sq)

    cdef double* u = <double*> malloc(sizeof(double) * 2)
    for j in range(2):
        u[j] = -x[j]

    cdef double* mobius_res = <double *> malloc(sizeof(double) * 2)
    mobius_addition(u, y, mobius_res)

    free(u)

    mob_add_norm = sqrt(mobius_res[0] ** 2 + mobius_res[1] ** 2)
    y_scalar = atanh(fmin(mob_add_norm, 1. - EPSILON))

    for j in range(2):
        res[j] = (2. / metric) * y_scalar * (mobius_res[j] / mob_add_norm)

    free(mobius_res)

# Compute the logarithmic map on the hyperboloid #TODO: idk if this is actually right
cdef void log_map_single_lorentz(DTYPE_t[3] p, DTYPE_t[3] q, DTYPE_t* res) nogil:
    cdef DTYPE_t beta = - minkowski_bilinear(p, q)
    cdef DTYPE_t mult = acosh(beta) / sqrt(beta * beta - 1)

    for i in range(3):
        res[i] = q[i] - beta * p[i]

#TODO: This method gets called in an experiment
cpdef void log_map(double[:, :] x, double[:, :] y, double[:, :] out, int num_threads) nogil:
    cdef double* log_map_res = <double*> malloc(sizeof(double) * 2)

    for i in range(x.shape[0]):
        log_map_single(&x[i, 0], &y[i, 0], log_map_res)

        for j in range(2):
            out[i, j] = log_map_res[j]

    free(log_map_res)

# Applies log_map_single_lorentz for every pair of points in x and y
cpdef void log_map_lorentz(DTYPE_t[:, :] x, DTYPE_t[:, :] y, DTYPE_t[:, :] out, int num_threads) nogil:
    cdef DTYPE_t* log_map_res = <DTYPE_t*> malloc(sizeof(DTYPE_t) * 3)

    for i in range(x.shape[0]):
        log_map_single_lorentz(&x[i, 0], &y[i, 0], log_map_res)

        for j in range(3):
            out[i, j] = log_map_res[j]

    free(log_map_res)

#TODO: this method gets called outside. check if needed
cpdef void constrain(double[:, :] y, double[:, :] out, int num_threads) nogil:
    for i in range(y.shape[0]):
        point_norm = sqrt(y[i, 0] ** 2 + y[i, 1] ** 2)

        for j in range(2):
            if point_norm >= BOUNDARY:
                out[i, j] = (y[i, j] / point_norm) - EPSILON
            else:
                out[i, j] = y[i, j]

cpdef void poincare_dists(double[:, :] y, double[:, :] out) nogil:
    cdef:
        long i, j

    with nogil, parallel(num_threads=12):
        for i in prange(0, y.shape[0], schedule='static'):
            for j in range(0, y.shape[0]):
                if i == j:
                    continue
                out[i, j] = distance(y[i, 0], y[i, 1], y[j, 0], y[j, 1])

def distance_grad_py(double[:] u, double[:] v, int ax):
    return distance_grad(u[0], u[1], v[0], v[1], ax)

def distance_py(double[:] u, double[:] v):
    return distance(u[0], u[1], v[0], v[1],)


#######################################
# Exact
#######################################
cdef double exact_compute_gradient(float[:] timings,
                            double[:] val_P,
                            double[:, :] pos_reference,
                            np.int64_t[:] neighbors,
                            np.int64_t[:] indptr,
                            double[:, :] tot_force,
                            _OcTree qt,
                            float theta,
                            int dof,
                            long start,
                            long stop,
                            bint compute_error,
                            int num_threads) nogil:
    # Having created the tree, calculate the gradient
    # in two components, the positive and negative forces
    cdef:
        long i, coord
        int ax
        long n_samples = pos_reference.shape[0]
        int n_dimensions = pos_reference.shape[1]
        double sQ
        double error
        clock_t t1 = 0, t2 = 0

    cdef double* neg_f = <double*> malloc(sizeof(double) * n_samples * n_dimensions)
    cdef double* pos_f = <double*> malloc(sizeof(double) * n_samples * n_dimensions)
    cdef DTYPE_t* tot_force_interm = <DTYPE_t*> malloc(sizeof(DTYPE_t) * 3)

    if TAKE_TIMING:
        t1 = clock()
    sQ = exact_compute_gradient_negative(pos_reference, neighbors, indptr, neg_f, qt, dof, theta, start,
                                   stop, num_threads)

    if TAKE_TIMING:
        t2 = clock()
        timings[2] = ((float) (t2 - t1)) / CLOCKS_PER_SEC

    if TAKE_TIMING:
        t1 = clock()

    error = compute_gradient_positive(val_P, pos_reference, neighbors, indptr,
                                      pos_f, n_dimensions, dof, sQ, start,
                                      qt.verbose, compute_error, num_threads)

    if TAKE_TIMING:
        t2 = clock()
        timings[3] = ((float) (t2 - t1)) / CLOCKS_PER_SEC

    for i in prange(start, n_samples, nogil=True, num_threads=num_threads, schedule='static'):
        for ax in range(n_dimensions):
            coord = i * n_dimensions + ax
            tot_force_interm[ax] = pos_f[coord] - (neg_f[coord] / sQ)
        project_to_tangent_space(&pos_reference[i, 0], tot_force_interm, &tot_force[i, 0])

    free(neg_f)
    free(pos_f)
    return error

#######################################
# Exact Negative
#######################################
cdef double exact_compute_gradient_negative(double[:, :] pos_reference,
                                      np.int64_t[:] neighbors,
                                      np.int64_t[:] indptr,
                                      double* neg_f,
                                      _OcTree qt,
                                      int dof,
                                      float theta,
                                      long start,
                                      long stop,
                                      int num_threads) nogil:
    cdef:
        int ax
        int n_dimensions = pos_reference.shape[1]
        int offset = n_dimensions + 2
        long i, j, k, idx
        long n = stop - start
        long dta = 0
        long dtb = 0
        double size, dist2s, mult
        double qijZ, sum_Q = 0.0
        long n_samples = indptr.shape[0] - 1
        double dij, qij, dij_sq
        DTYPE_t[3] neg_f_axes #XXX: n dims

    with nogil, parallel(num_threads=num_threads):
        for i in prange(start, n_samples, schedule='static'):
            # Init the gradient vector
            for ax in range(n_dimensions):
                neg_f[i * n_dimensions + ax] = 0.0

            for j in range(start, n_samples):
                if i == j:
                    continue
                dij = distance_q(&pos_reference[i, 0], &pos_reference[j, 0])
                dij_sq = dij * dij

                qij = 1. / (1. + dij_sq)

                if GRAD_FIX:
                    # New Fix
                    mult = qij * qij * dij
                else:
                    # Old Solution
                    mult = qij * qij

                sum_Q += qij
                distance_grad_q(&pos_reference[i, 0], &pos_reference[j, 0], neg_f_axes)
                for ax in range(n_dimensions):
                    neg_f[i * n_dimensions + ax] += mult * neg_f_axes[ax]

    # Put sum_Q to machine EPSILON to avoid divisions by 0
    sum_Q = max(sum_Q, FLOAT64_EPS)
    return sum_Q

#####################################################
# Grad
#####################################################
cdef double compute_gradient(float[:] timings,
                            double[:] val_P,
                            double[:, :] pos_reference,
                            np.int64_t[:] neighbors,
                            np.int64_t[:] indptr,
                            double[:, :] tot_force,
                            _OcTree qt,
                            float theta,
                            int dof,
                            long start,
                            long stop,
                            bint compute_error,
                            int num_threads) nogil:
    # Having created the tree, calculate the gradient
    # in two components, the positive and negative forces
    cdef:
        long i, coord
        int ax
        long n_samples = pos_reference.shape[0]
        int n_dimensions = pos_reference.shape[1]
        double sQ
        double error
        DTYPE_t[3] tot_force_interm
        clock_t t1 = 0, t2 = 0

    cdef double* neg_f = <double*> malloc(sizeof(double) * n_samples * n_dimensions)
    cdef double* pos_f = <double*> malloc(sizeof(double) * n_samples * n_dimensions)

    if TAKE_TIMING:
        t1 = clock()

    sQ = compute_gradient_negative(pos_reference, neg_f, qt, dof, theta, start,
                                   stop, num_threads)
    if TAKE_TIMING:
        t2 = clock()
        timings[2] = ((float) (t2 - t1)) / CLOCKS_PER_SEC

    if TAKE_TIMING:
        t1 = clock()

    error = compute_gradient_positive(val_P, pos_reference, neighbors, indptr,
                                      pos_f, n_dimensions, dof, sQ, start,
                                      qt.verbose, compute_error, num_threads)

    if TAKE_TIMING:
        t2 = clock()
        timings[3] = ((float) (t2 - t1)) / CLOCKS_PER_SEC

    for i in prange(start, n_samples, nogil=True, num_threads=num_threads, schedule='static'):
        for ax in range(n_dimensions):
            coord = i * n_dimensions + ax
            tot_force_interm[ax] = pos_f[coord] - (neg_f[coord] / sQ)

        project_to_tangent_space(&pos_reference[i, 0], tot_force_interm, &tot_force[i, 0])

    free(neg_f)
    free(pos_f)
    return error

cdef double compute_gradient_positive(double[:] val_P,
                                     double[:, :] pos_reference,
                                     np.int64_t[:] neighbors,
                                     np.int64_t[:] indptr,
                                     double* pos_f,
                                     int n_dimensions,
                                     int dof,
                                     double sum_Q,
                                     np.int64_t start,
                                     int verbose,
                                     bint compute_error,
                                     int num_threads) nogil:
    # Sum over the following expression for i not equal to j
    # grad_i = p_ij (1 + ||y_i - y_j||^2)^-1 (y_i - y_j)
    # This is equivalent to compute_edge_forces in the authors' code
    # It just goes over the nearest neighbors instead of all the data points
    # (unlike the non-nearest neighbors version of `compute_gradient_positive')
    cdef:
        int ax
        long i, j, k
        long n_samples = indptr.shape[0] - 1
        double C = 0.0
        double dij, qij, pij, mult, dij_sq
        DTYPE_t[3] dist_grad_pos

    with nogil, parallel(num_threads=num_threads):
        for i in prange(start, n_samples, schedule='static'):
            # Init the gradient vector
            for ax in range(n_dimensions):
                pos_f[i * n_dimensions + ax] = 0.0
            # Compute the positive interaction for the nearest neighbors
            for k in range(indptr[i], indptr[i+1]):
                j = neighbors[k]
                pij = val_P[k]

                dij = distance_q(&pos_reference[i, 0], &pos_reference[j, 0])
                dij_sq = dij * dij

                qij = 1. / (1. + dij_sq)

                if GRAD_FIX:
                    # New Fix
                    mult = pij * qij * dij
                else:
                    # Old solution
                    mult = pij * qij

                # only compute the error when needed
                if compute_error:
                    qij = qij / sum_Q
                    C += pij * log(max(pij, FLOAT32_TINY) / max(qij, FLOAT32_TINY))
                distance_grad_q(&pos_reference[i, 0], &pos_reference[j, 0], dist_grad_pos)
                for ax in range(n_dimensions):
                    pos_f[i * n_dimensions + ax] += mult * dist_grad_pos[ax]
    return C

cdef double compute_gradient_negative(double[:, :] pos_reference,
                                      double* neg_f,
                                      _OcTree qt,
                                      int dof,
                                      float theta,
                                      long start,
                                      long stop,
                                      int num_threads) nogil:
    if stop == -1:
        stop = pos_reference.shape[0]
    cdef:
        int ax
        int n_dimensions = pos_reference.shape[1]
        int offset = n_dimensions + 2
        long i, j, idx
        long n = stop - start
        long dta = 0
        long dtb = 0
        double size, dist2s, mult
        double qijZ, sum_Q = 0.0
        double* force
        double* summary
        double* pos
        double* neg_force

    with nogil, parallel(num_threads=num_threads):
        # Define thread-local buffers
        summary = <double *> malloc(sizeof(double) * n * offset)
        force = <double *> malloc(sizeof(double) * n_dimensions)
        pos = <double *> malloc(sizeof(double) * n_dimensions)
        neg_force = <double *> malloc(sizeof(double) * n_dimensions)

        for i in prange(start, stop, schedule='static'):
            # Clear the arrays
            for ax in range(n_dimensions):
                force[ax] = 0.0
                neg_force[ax] = 0.0
                pos[ax] = pos_reference[i, ax]

            idx = qt.summarize(pos, summary, theta*theta)

            # Compute the t-SNE negative force
            # for the digits dataset, walking the tree
            # is about 10-15x more expensive than the
            # following for loop
            for j in range(idx // offset):

                dist2s = summary[j * offset + n_dimensions]
                size = summary[j * offset + n_dimensions + 1]
                qijZ = 1. / (1. + dist2s)  # 1/(1+dist)

                sum_Q += size * qijZ   # size of the node * q

                if GRAD_FIX:
                    # New Fix
                    mult = size * qijZ * qijZ * sqrt(dist2s)
                else:
                    # Old Solution
                    mult = size * qijZ * qijZ

                for ax in range(n_dimensions):
                    neg_force[ax] += mult * summary[j * offset + ax]

            for ax in range(n_dimensions):
                neg_f[i * n_dimensions + ax] = neg_force[ax]

        free(force)
        free(pos)
        free(neg_force)
        free(summary)

    # Put sum_Q to machine EPSILON to avoid divisions by 0
    sum_Q = max(sum_Q, FLOAT64_EPS)
    return sum_Q

def gradient(float[:] timings,
             double[:] val_P,
             double[:, :] pos_output,
             np.int64_t[:] neighbors,
             np.int64_t[:] indptr,
             double[:, :] forces,
             float theta,
             int n_dimensions,
             int verbose,
             int dof=1,
             long skip_num_points=0,
             bint compute_error=1,
             int num_threads=1,
             bint exact=1,
             bint area_split=0,
             bint grad_fix=0):
    cdef double C
    cdef int n
    cdef _OcTree qt = _OcTree(pos_output.shape[1], verbose)
    cdef clock_t t1 = 0, t2 = 0

    global AREA_SPLIT
    AREA_SPLIT = area_split

    global GRAD_FIX
    GRAD_FIX = grad_fix

    if not exact:
        if TAKE_TIMING:
            t1 = clock()

        qt.build_tree(pos_output)

        if TAKE_TIMING:
            t2 = clock()
            timings[0] = ((float) (t2 - t1)) / CLOCKS_PER_SEC

    if TAKE_TIMING:
        t1 = clock()
    if exact:
        C = exact_compute_gradient(timings, val_P, pos_output, neighbors, indptr, forces,
                             qt, theta, dof, skip_num_points, -1, compute_error,
                             num_threads)
    else:
        C = compute_gradient(timings, val_P, pos_output, neighbors, indptr, forces,
                             qt, theta, dof, skip_num_points, -1, compute_error,
                             num_threads)
    if TAKE_TIMING:
        t2 = clock()
        timings[1] = ((float) (t2 - t1)) / CLOCKS_PER_SEC

    if not compute_error:
        C = np.nan
    return C
