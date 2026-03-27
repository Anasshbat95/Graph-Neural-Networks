# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3

cimport cython

import numpy as np
cimport numpy as cnp
cnp.import_array()

from cython.operator cimport dereference as deref
from libcpp.cmath cimport fmin, fmax
from libcpp.vector cimport vector
from libcpp.algorithm cimport nth_element, max_element

from cython cimport floating
from _types cimport integral

# TODO: Find a way to reduce the code duplication here
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _min_potential(
    const integral[:, :, ::1] labels,
    const floating[:, :, ::1] potential,
    floating[::1] agg_potential,
):
    cdef:
        long i0, i1, i2, gidx
        long[3] shape_labels = labels.shape
    
    for i0 in range(shape_labels[0]):
        for i1 in range(shape_labels[1]):
            for i2 in range(shape_labels[2]):
                gidx = labels[i0, i1, i2]
                agg_potential[gidx] = fmin(
                    potential[i0, i1, i2], 
                    agg_potential[gidx],
                )

@cython.wraparound(False)
@cython.boundscheck(False)    
cdef _max_potential(
    const integral[:, :, ::1] labels,
    const floating[:, :, ::1] potential,
    floating[::1] agg_potential,
):
    cdef:
        long i0, i1, i2, gidx
        long[3] shape_labels = labels.shape

    for i0 in range(shape_labels[0]):
        for i1 in range(shape_labels[1]):
            for i2 in range(shape_labels[2]):
                gidx = labels[i0, i1, i2]
                agg_potential[gidx] = fmax(
                    potential[i0, i1, i2],
                    agg_potential[gidx],
                )


@cython.wraparound(False)
@cython.boundscheck(False)
cdef _mean_potential(
    const integral[:, :, ::1] labels,
    const long[::1] label_count,
    const floating[:, :, ::1] potential,
    floating[::1] agg_potential,
):
    cdef:
        long i0, i1, i2, gidx
        long num_labels = label_count.size
        long[3] shape_labels = labels.shape
    
    for i0 in range(shape_labels[0]):
        for i1 in range(shape_labels[1]):
            for i2 in range(shape_labels[2]):
                gidx = labels[i0, i1, i2]
                agg_potential[gidx] += potential[i0, i1, i2]

    for gidx in range(num_labels):
        agg_potential[gidx] /= <floating>label_count[gidx]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef _median_potential(
    const integral[:, :, ::1] labels,
    const long[::1] label_count,
    const floating[:, :, ::1] potential,
    floating[::1] agg_potential,
):
    # Require the largest grain to have a upper boundary for the number of
    # potential values to store.
    cdef long num_labels = label_count.size
    iter_max = max_element(&label_count[0], &label_count[num_labels])

    potential_by_labels = np.empty(
        (label_count.size, deref( iter_max )), 
        dtype="float64" if floating is double else "float32",
    )

    cdef:
        long i0, i1, i2, gidx, idx_median
        long[3] shape_labels = labels.shape
        vector[size_t] counter = vector[size_t](num_labels, 0) 
        floating[:, ::1] potential_by_labels_view = potential_by_labels

    for i0 in range(shape_labels[0]):
        for i1 in range(shape_labels[1]):
            for i2 in range(shape_labels[2]):
                gidx = labels[i0, i1, i2]
                potential_by_labels_view[gidx, counter[gidx]] = potential[i0, i1, i2]
                counter[gidx] += 1

    for gidx in range(num_labels):
        # In case the number of potential values is uneven, take the value in the middle of
        # the sorted array. Otherwise, if the size is even, take the mean value of the two
        # values in the middle of the sorted array.
        idx_median = counter[gidx] // 2
        nth_element( 
            &potential_by_labels_view[gidx, 0],
            &potential_by_labels_view[gidx, 0] + idx_median,
            &potential_by_labels_view[gidx, 0] + counter[gidx],    
        )
        # Only the nth element is guaranteed to be in its sorted position. All other elements 
        # * before are just guranteed to be *smaller* (but potentially in unsorted order), and
        # * after are just guaranteed to be *larger* (but potentially in unsorted order).
        # That is, in case of *even* count, I have to determine the largest element in the 
        # first half of the array and use this to compute the median.
        if counter[gidx] % 2 != 0:
            agg_potential[gidx] = potential_by_labels_view[gidx, idx_median]
        else:
            max_iter = max_element(
                &potential_by_labels_view[gidx, 0],
                &potential_by_labels_view[gidx, 0] + idx_median
            )
            agg_potential[gidx] = 0.5 * (
                deref( max_iter ) + 
                potential_by_labels_view[gidx, idx_median]
            )

