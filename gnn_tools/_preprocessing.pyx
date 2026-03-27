# distutils: language = c++
# cython: language_level=3

cimport cython

import numpy as np

from cython cimport floating

from _types cimport integral # custom typedef
from gnn_tools._agg_potential cimport (
    _min_potential,
    _max_potential,
    _mean_potential,
    _median_potential,
)

from libcpp.cmath cimport sqrt, fmin, fmax
from libcpp.vector cimport vector
from libcpp.algorithm cimport fill
from libcpp.limits cimport numeric_limits
from libc.stdint cimport int32_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _insert_NN(
    int32_t label, int32_t label_nn,
    int32_t[:, ::1] nn_contact_areas,
    int32_t max_allowed_entries,
    vector[int32_t]& nn_pair_counter,
):
    cdef:
        int32_t kMaxIntValue = numeric_limits[int32_t].max()
        long idx_current

    idx_current = 0
    while (
        (nn_contact_areas[label, idx_current] != label_nn) and 
        (nn_contact_areas[label, idx_current] != kMaxIntValue) and
        (idx_current < <long>max_allowed_entries)
    ):
        idx_current += 1 
    
    if nn_contact_areas[label, idx_current] == kMaxIntValue:
        nn_pair_counter[label] += 1
        if 3 * nn_pair_counter[label] >= max_allowed_entries:
            raise ValueError("Number of NN pairs exceeds allowed limit.")
        nn_contact_areas[label, idx_current] = label
        nn_contact_areas[label, idx_current + 1] = label_nn
        nn_contact_areas[label, idx_current + 2] = -1 # initialize contact area count with -1
    elif nn_contact_areas[label, idx_current] == label_nn:
        nn_contact_areas[label, idx_current + 1] -= 1 # position next to label_nn is count
    else:
        raise ValueError("Could not find nearest neighbor.")


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void find_nearest_neighbors_axis(
    const int32_t[:, :, ::1] labels, 
    const int32_t[:, ::1] boundary_indices, 
    int32_t[:, ::1] nn_contact_areas, 
    long axis_to_check,
):
    cdef:
        long[3] labels_shape = labels.shape
        long[2] boundary_indices_shape = boundary_indices.shape
        long[2] nn_contact_areas_shape = nn_contact_areas.shape
    cdef:
        long idx, i, label_idx, label, label_nn
        long[3] label_inds
        long max_allowed_entries = nn_contact_areas_shape[1]

    # Used to count the number of discovered nearest neighbors per label.
    nn_pair_counter = vector[int32_t](nn_contact_areas_shape[0], 0)

    for idx in range(boundary_indices_shape[0]):
        for i in range(3):
            label_inds[i] = <long>boundary_indices[idx, i]
        if label_inds[axis_to_check] < (labels_shape[axis_to_check] - 1):
            label = labels[label_inds[0], label_inds[1], label_inds[2]]
            label_inds[axis_to_check] += 1
            label_nn = labels[label_inds[0], label_inds[1], label_inds[2]]
            _insert_NN(label, label_nn, nn_contact_areas, max_allowed_entries, nn_pair_counter)
            _insert_NN(label_nn, label, nn_contact_areas, max_allowed_entries, nn_pair_counter)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void node_distances_impl(
    const int32_t[:, ::1] nn_pairs,
    const floating[:, ::1] centers,
    floating[::1] center_distances,
):
    cdef:
        int32_t idx, idx_nn, pidx
        int32_t num_pairs = <int32_t>nn_pairs.shape[0]

    for pidx in range(num_pairs):
        idx = nn_pairs[pidx, 0]
        idx_nn = nn_pairs[pidx, 1]
        center_distances[pidx] = sqrt( 
            (centers[idx, 0] - centers[idx_nn, 0]) ** 2 + 
            (centers[idx, 1] - centers[idx_nn, 1]) ** 2 + 
            (centers[idx, 2] - centers[idx_nn, 2]) ** 2
        )

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void node_centers_impl(
    const integral[:, :, ::1] labels,
    const long[::1] label_counts,
    floating[:, ::1] centers,
):
    cdef:
        long cidx, i0, i1, i2
        long num_labels = label_counts.size
        long[3] labels_shape = labels.shape

    for i0 in range(labels_shape[0]):
        for i1 in range(labels_shape[1]):
            for i2 in range(labels_shape[2]):
                cidx = labels[i0, i1, i2]
                centers[cidx, 0] += <floating>i0
                centers[cidx, 1] += <floating>i1
                centers[cidx, 2] += <floating>i2

    for cidx in range(num_labels):
        for i0 in range(3):
            centers[cidx, i0] /= <floating>label_counts[cidx]


cpdef void agg_potential_impl(
    const integral[:, :, ::1] labels,
    const long[::1] label_count,
    const floating[:, :, ::1] potential,
    floating[::1] agg_potential,
    agg_type: str = "mean",
):
    if agg_type == "min":
        init_value: floating = (
            np.finfo("float64").max if floating is double 
            else np.finfo("float32").max
        )
        fill(&agg_potential[0], &agg_potential[-1], <floating>init_value)
        _min_potential(labels, potential, agg_potential)
    elif agg_type == "max":
        init_value: floating = (
            np.finfo("float64").min if floating is double 
            else np.finfo("float32").min
        )
        fill(&agg_potential[0], &agg_potential[-1], <floating>init_value)
        _max_potential(labels, potential, agg_potential)
    elif agg_type == "mean":
        _mean_potential(labels, label_count, potential, agg_potential)
    elif agg_type == "median":
        _median_potential(labels, label_count, potential, agg_potential)
    else:
        raise ValueError(f"{agg_type = } is not a valid option")