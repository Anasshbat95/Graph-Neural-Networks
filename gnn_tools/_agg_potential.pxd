# cython: language_level=3

from cython cimport floating
from _types cimport integral

cdef _min_potential(
    const integral[:, :, ::1] labels,
    const floating[:, :, ::1] potential,
    floating[::1] agg_potential,
)

cdef _max_potential(
    const integral[:, :, ::1] labels,
    const floating[:, :, ::1] potential,
    floating[::1] agg_potential,
)

cdef _mean_potential(
    const integral[:, :, ::1] labels,
    const long[::1] label_count,
    const floating[:, :, ::1] potential,
    floating[::1] agg_potential,
)

cdef _median_potential(
    const integral[:, :, ::1] labels,
    const long[::1] label_count,
    const floating[:, :, ::1] potential,
    floating[::1] agg_potential,
)