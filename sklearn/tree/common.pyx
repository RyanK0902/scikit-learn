import numpy as np
from ._criterion cimport HistStruct
SIZE_t = np.intp

HISTOGRAM_DTYPE = np.dtype([
    ('left', SIZE_t*),  # left count of histograms
    ('right', SIZE_t*),  # right count of histograms
])
