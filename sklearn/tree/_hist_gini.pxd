# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# License: BSD 3 clause
# See _criterion.pyx for implementation details.

from ._tree cimport DTYPE_t          # Type of X
from ._tree cimport DOUBLE_t         # Type of y, sample_weight
from ._tree cimport SIZE_t           # Type for indices and counters
from ._tree cimport INT32_t          # Signed 32 bit integer
from ._tree cimport UINT32_t         # Unsigned 32 bit integer


cdef class HistGiniCriterion(Criterion):
    """Abstract criterion for classification."""

    cdef SIZE_t[::1] n_classes
    cdef SIZE_t max_n_classes
    cdef HISTOGRAM_DTYPE[::1] histograms
    cdef SIZE_t[::1] bin_counts

    cdef packed struct histograms:
        # Same as histogram dtype but we need a struct to declare views. It needs
        # to be packed since by default numpy dtypes aren't aligned
        SIZE_t[:, ::1] left
        SIZE_t[:, ::1] right

    cdef int insert_histograms(self, SIZE_t[::1] bin_idcs) nogil
