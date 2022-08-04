# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# License: BSD 3 clause

# See _splitter.pyx for details.

from ._criterion cimport Criterion

from ._tree cimport DTYPE_t          # Type of X
from ._tree cimport DOUBLE_t         # Type of y, sample_weight
from ._tree cimport SIZE_t           # Type for indices and counters
from ._tree cimport INT32_t          # Signed 32 bit integer
from ._tree cimport UINT32_t         # Unsigned 32 bit integer
from ._tree cimport UINT8_t

cdef struct SplitRecord:
    # Data to track sample split
    SIZE_t feature         # Which feature to split on.
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    double threshold       # Threshold to split at.
    double improvement     # Impurity improvement given parent node.
    double impurity_left   # Impurity of the left split.
    double impurity_right  # Impurity of the right split.

cdef struct ArmRecord:
    # data to track each estimates left, right impurities
    double l_impurity
    double r_impurity

cdef class Splitter:
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # Internal structures
    cdef bint is_histogram

    cdef public Criterion criterion      # Impurity criterion
    cdef public SIZE_t max_features      # Number of features to test
    cdef public SIZE_t min_samples_leaf  # Min samples in a leaf
    cdef public double min_weight_leaf   # Minimum weight in a leaf

    cdef object random_state             # Random state
    cdef UINT32_t rand_r_state           # sklearn_rand_r random number state

    cdef SIZE_t[::1] samples             # Sample indices in X, y
    cdef SIZE_t n_samples                # X.shape[0]
    cdef double weighted_n_samples       # Weighted number of samples
    cdef SIZE_t[::1] features            # Feature indices in X
    cdef SIZE_t[::1] constant_features   # Constant features indices
    cdef SIZE_t n_features               # X.shape[1]
    cdef DTYPE_t[::1] feature_values     # temp. array holding feature values

    cdef SIZE_t start                    # Start position for the current node
    cdef SIZE_t end                      # End position for the current node

    cdef const DOUBLE_t[:, ::1] y
    cdef DOUBLE_t* sample_weight

    cdef UINT8_t[:,::1] X_binned         # Mappings of samples -> bins
    cdef DOUBLE_t[:,::1] bin_thresholds  # Thresholds per feature
    cdef SIZE_t[::1] batch_binned_col    # 1d column of samples_to_bins
                                         # -> will be cast from np.uint8 to np.intp

    # arrays needed for sampling
    cdef SIZE_t batch_size
    cdef SIZE_t[::1] batch_idcs
    cdef SIZE_t[::1] batch_y

    # access is information about the valid ROWS of candidates.
    cdef SIZE_t[::1] samples_mask        # without replacement
    cdef SIZE_t[::1] accesses
    cdef SIZE_t[:,::1] candidates

    cdef SIZE_t[:,::1] temp1
    cdef SIZE_t[:,::1] temp2
    cdef SIZE_t[:,::1] temp3

    # impurity left and right for each of the estimates
    # -> need to avoid re-computations
    cdef ArmRecord[:,::1] arm_records

    # Candidates can move from being excluded to included as the value of
    # the estimates change with more samples (the min ucb gets larger).
    cdef double[:,::1] estimates
    cdef double[:,::1] lcbs
    cdef double[:,::1] ucbs
    cdef double[:,::1] cb_delta
    cdef SIZE_t[:,::1] exact_mask

    # The samples vector `samples` is maintained by the Splitter object such
    # that the samples contained in a node are contiguous. With this setting,
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.

    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.

    # Methods
    cdef int init(self, object X, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight) except -1

    cdef int _init_mab(self, SIZE_t batch_size, SIZE_t num_bins) except -1

    cdef int mab_split(self, double impurity, SplitRecord* split) nogil except -1

    cdef int sample_targets(self, SIZE_t it, SIZE_t batch_size,
                            SIZE_t[:,::1] candidates, SIZE_t[::1] accesses,
                            double[:,::1] estimates, double[:,::1] cb_delta,
                            ArmRecord[:,::1] arm_records) nogil except -1

    cdef int update_best_split(self, double impurity, double[:,::1] estimates,
                                       ArmRecord[:,::1] arm_records, SplitRecord* split) except -1

    cdef int node_reset(self, bint first, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1

    cdef int node_split(self,
                        double impurity,   # Impurity of the node
                        SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1

    cdef void node_value(self, double* dest) nogil

    cdef double node_impurity(self) nogil
