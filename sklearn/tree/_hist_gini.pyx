# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs

import numpy as np
cimport numpy as cnp
cnp.import_array()

from numpy.math cimport INFINITY
from scipy.special.cython_special cimport xlogy

from ._utils cimport log
from ._utils cimport WeightedMedianCalculator
from ._criterion import Criterion

# EPSILON is used in the Poisson criterion
cdef double EPSILON = 10 * np.finfo('double').eps


cdef class HistGiniCriterion(Criterion):
    """
    Similar to the Gini Criterion class but with a few changes to use histograms instead.
    Some notable changes is that we no longer update sum_total/left/right but maintain histograms left/right instead.
    Also, the output y is of shape (n, 1) instead of (n, k).
    """

    def __cinit__(
        self,
        SIZE_t n_bins,
        SIZE_t n_features,
        SIZE_t n_classes
        # SIZE_t n_outputs,
        # cnp.ndarray[SIZE_t, ndim=1] n_classes
    ):
        """
        Initialize attributes for this criterion.

        Parameters
        ----------
        n_features: SIZE_t
            The number of features
        n_classes : SIZE_t
            The number of unique classes in the target
        """
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_bins = n_bins
        self.n_features = n_features
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.n_classes = n_classes

        # output_histograms[i, j] = struct(left, right)
        # -> left, right histograms for output i, feature j.
        HISTOGRAM_DTYPE = np.dtype([
            ('left', SIZE_t, (n_bins, n_classes)),  # left count of histograms
            ('right', SIZE_t, (n_bins, n_classes)),  # right count of histograms
        ])
        self.histograms = np.empty(n_features, dtype=HISTOGRAM_DTYPE)
        self.bin_counts = np.zeros((n_bins, n_classes), dtype=SIZE_t)
        self.current_hist = np.zeros((1, n_classes), dtype=SIZE_t)


    def __reduce__(self):
        # Todo: is it ok to change this??
        return (type(self),
                (self.n_bins, self.n_features, self.n_classes), self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight, double weighted_n_samples,
                  SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1:
        """
        Initialize the criterion.

        This initializes the criterion at node samples[start:end] and children
        samples[start:start] and samples[start:end]. This function is called by node_reset (per node), 
        meaning that samples must already be sorted accordingly after splitting parent node. 
        The output_histograms array must be reset as well. 

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            A SINGLE target stored as a buffer for memory efficiency
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample
        weighted_n_samples : double
            The total weight of all samples
        samples : array-like, dtype=SIZE_t
            A mask on the samples, showing which ones we want to use
        start : SIZE_t
            The first sample to use in the mask
        end : SIZE_t
            The last sample to use in the mask
        """
        # setting all the attributes accordingly
        cdef:
            SIZE_t i
            DOUBLE_t[::1] y_col

        # Todo: doing this every time is inefficient. Compute once when initialized instead??
        for i in range(y.shape[1]):
            y_col[i] = y[i, 0]

        self.y = y_col  # this is now a 1d array!
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        # resetting all changes by previous node
        self.pos = start
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        cdef:
            SIZE_t f
            SIZE_t b
            SIZE_t[:, ::1] left
            SIZE_t[:, ::1] right

        for f in range(self.n_features):
            # reset left, right histograms in the struct
            left = self.histograms[f].left
            right = self.histograms[f].right

            for b in range(self.n_bins):
                memset(&left[b, 0], 0, self.n_classes * sizeof(SIZE_t))
                memset(&right[b, 0], 0, self.n_classes * sizeof(SIZE_t))

        return 0

    cdef int insert_histograms(
            self,
            SIZE_t feature_idx, SIZE_t batch_size,
            DTYPE_t[::1] bin_idcs, DTYPE_t[::1] batch_y
    ) nogil:
        cdef:
            SIZE_t i, lj, rj
            SIZE_t bin, split
            SIZE_t[:, ::1] left = self.histograms[feature_idx].left
            SIZE_t[:, ::1] right = self.histograms[feature_idx].right

        for i in range(batch_size):
            # left
            for lj in range(bin, self.n_bins):
                left[lj, batch_y[i]] += 1

            # right
            for rj in range(bin):
                for c in range(n_classes):
                    right[rj, batch_y[i]] += 1


        """
        # update bin_counts -> analogous to "hist" in python implementation
        for bin in range(n_bins):
            for i in range(batch_size):
                if bin_idcs[i] == bin:
                    bin_counts[bin, batch_y[i]] += 1

        # update histograms left and right
        for bin in range(n_bins):
            # left
            for j in range(bin, n_bins):
                for c in range(n_classes):
                    left[j, c] += bin_counts[j, c]

            # right
            for jj in range(bin):
                for c in range(n_classes):
                    right[jj, c] += bin_counts[jj, c]
        """


    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """
        Updated statistics by moving samples[pos:new_pos] to the left child.
        

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : SIZE_t
            The new ending position for which to move samples from the right
            child to the left child. 
        """
        cdef:
            SIZE_t pos = self.pos
            SIZE_t end = self.end

            SIZE_t* samples = self.samples
            DOUBLE_t* sample_weight = self.sample_weight

            SIZE_t i
            SIZE_t p
            SIZE_t k
            SIZE_t c
            SIZE_t label_index
            DOUBLE_t w = 1.0

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    self.sum_left[k, <SIZE_t> self.y[i, k]] += w

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    self.sum_left[k, <SIZE_t> self.y[i, k]] -= w

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                self.sum_right[k, c] = self.sum_total[k, c] - self.sum_left[k, c]

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        """
        Evaluate the impurity of the current node.

        Evaluate the Gini criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        cdef double gini = 0.0
        cdef double sq_count
        cdef double count_k
        cdef SIZE_t c

        # left_sum = int(np.sum(h.left[b_idx, :], dtype=np.int64)) -> one "row"

        sq_count = 0.0
        for c in range(self.n_classes):
            count_k = self.sum_total[k, c]
            sq_count += count_k * count_k

        gini += 1.0 - sq_count / (self.weighted_n_node_samples *
                                  self.weighted_n_node_samples)

        return gini / self.n_outputs


    cdef void children_impurity(
        self,
        double* impurity_curr,
        double* impurity_left,
        double* impurity_right,

        double* variance_curr,
        double* variance_left,
        double* variance_right,

        SIZE_t f,
        SIZE_t bin
    ) nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]) using the Gini index.

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node to
        impurity_right : double pointer
            The memory address to save the impurity of the right node to
        """
        cdef:
            SIZE_t i
            double n = 0.0
            double left_size, right_size
            double left_sum = 0.0
            double right_sum = 0.0

            double p_left
            double p_right
            double sq_p_left = 0.0
            double sq_p_right = 0.0

            SIZE_t n_classes = self.n_classes
            SIZE_t n_node_samples = self.n_node_samples
            SIZE_t[:, ::1] left = self.histograms[f].left
            SIZE_t[:, ::1] right = self.histograms[f].right
            SIZE_t[:,::1] curr = self.current_hist

        # initialize relevant coefficients
        for i in range(n_classes):
            curr[0, i] = left[0, i] + right[0, i]   # getting histograms for curr_impurity
            n += curr[i, 0]

            left_sum += left[bin, i]
            right_sum += right[bin, i]

        left_weight = left_sum / n
        right_weight = right_sum / n

        # get impurities and variances
        get_gini(curr, 0.0, n, 1.0, n_node_samples, n_classes, impurity_curr, variance_curr)
        get_gini(left, bin, left_sum, left_weight, n_node_samples, n_classes, impurity_left, variance_left)
        get_gini(right, bin, right_sum, right_weight, n_node_samples, n_classes, impurity_right, variance_right)


cdef inline double[::1] get_gini(
        SIZE_t[:, ::1] histogram, SIZE_t bin,
        double n, double weight,
        SIZE_t pop_size, SIZE_t n_classes,
        double* impurity, double* variance
) nogil:
    cdef:
        SIZE_t c
        double p = 0.0
        double p_sq = 0.0

        double v_p = 0.0
        double v_g = 0.0

    # in the case where we have drawn no relevant samples yet
    if n == 0 or pop_size <= 1:
        impurity[0] = 0.0
        variance[0] = 0.0
        return 0

    cdef p_end = 2 * histogram[bin, n_classes] / n
    for c in range(n_classes):
        p += histogram[bin, c] / n
        p_sq += (p * p)

        if c == n_classes:
            break

        v_p = (p * (1 - p) * (pop_size - n)) / (n * (pop_size - 1))
        dG_dp = -2 * p + p_end
        v_g += dG_dp * dG_dp * v_p

    impurity[0] = weight * (1 - p_sq)
    variance[0] = weight * weight * v_g


    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] and save it into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memcpy(dest, &self.sum_total[k, 0], self.n_classes[k] * sizeof(double))
            dest += self.max_n_classes

