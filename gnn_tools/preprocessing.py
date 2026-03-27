import numpy as np
import pandas as pd

from scipy import sparse

from pathlib import Path

from gnn_tools._preprocessing import (
    find_nearest_neighbors_axis,
    node_distances_impl,
    node_centers_impl,
    agg_potential_impl,
)


def node_distances(
    nn_pairs: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    center_distances = np.empty((nn_pairs.shape[0],), dtype="float64")
    node_distances_impl(
        nn_pairs,
        centers,
        center_distances,
    )
    return center_distances


def node_centers_and_volumes(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    centers: np.ndarray = np.zeros((unique_labels.size, 3), dtype="float64")

    node_centers_impl(
        labels=labels,
        label_counts=label_counts,
        centers=centers,
    )

    return centers, label_counts


def find_node_neighbors(
    labels: np.ndarray,
    max_allowed_entries: int = 300,
) -> np.ndarray:
    int32_max_value = np.iinfo(np.int32).max

    nn_contact_areas = np.full(
        (labels.max() + 1, max_allowed_entries),
        int32_max_value,
        dtype="int32",
    )

    for axis in range(3):
        boundary_indices = np.column_stack(
            np.nonzero(
                np.roll(
                    np.abs(labels - np.roll(labels, 1, axis=axis)).astype(bool),
                    -1,
                    axis=axis,
                )
            ),
        ).astype("int32")
        find_nearest_neighbors_axis(
            labels,
            boundary_indices,
            nn_contact_areas,
            axis,
        )

    return np.abs(nn_contact_areas[nn_contact_areas != int32_max_value].reshape(-1, 3))


class GrainsToGraph:
    def __init__(self, filepath: Path):
        self.filepath: Path = filepath

        with np.load(filepath) as structure_file:
            self.grain_labels = structure_file["arr_0"]
            self.grain_labels -= self.grain_labels.min()  # assume 0-based indexing

        self.centeroids, self.volumes = node_centers_and_volumes(
            labels=self.grain_labels
        )

        self.neighbors_and_contact_areas = find_node_neighbors(labels=self.grain_labels)

        self.node_centeroid_distances = node_distances(
            # Passing a slice along axis-1 will *not* make the resulting array C-contiguous.
            # The 3rd column contains the contact area between the nodes listed in
            # the 1st and 2nd column. However, this quantity is not referenced in this function.
            nn_pairs=self.neighbors_and_contact_areas,
            centers=self.centeroids,
        )

    def node_information(
        self,
        as_frame: bool = False,
        dtype: str = "float32",
    ) -> np.ndarray:
        column_names_and_dtype = (
            ("center-coord-0", dtype),
            ("center-coord-1", dtype),
            ("center-coord-2", dtype),
            ("degree", "int32"),
            ("volume", dtype),
        )

        output: np.ndarray = np.hstack(
            (
                # 3 coordinates of the center position
                self.centeroids,
                # node degrees
                np.unique(
                    self.neighbors_and_contact_areas[:, 0],
                    return_counts=True,
                )[
                    1
                ][:, np.newaxis],
                # volume of the node
                self.volumes[:, np.newaxis],
            ),
        ).astype(dtype)

        return (
            output
            if not as_frame
            else pd.DataFrame(
                data=output,
                columns=tuple(x[0] for x in column_names_and_dtype),
            ).astype(
                {
                    colname: data_type
                    for (
                        colname,
                        data_type,
                    ) in column_names_and_dtype
                }
            )
        )

    def edge_information(
        self,
        as_frame: bool = False,
        dtype: str = "float32",
    ):
        column_names_and_dtype = (
            ("center_node", "int32"),
            ("neighbor_node", "int32"),
            ("contact_area", dtype),
            ("distance", dtype),
        )

        output: np.ndarray = np.hstack(
            (
                self.neighbors_and_contact_areas[:, -1, np.newaxis],  # contact area
                self.node_centeroid_distances[:, np.newaxis],
            )
        ).astype(dtype)

        return (
            output
            if not as_frame
            else pd.DataFrame(
                data=np.hstack((self.neighbor_pairs, output)),
                columns=tuple(x[0] for x in column_names_and_dtype),
            ).astype(
                {
                    colname: data_type
                    for (
                        colname,
                        data_type,
                    ) in column_names_and_dtype
                }
            )
        )

    @property
    def neighbor_pairs(self):
        return self.neighbors_and_contact_areas[:, :2]

    def adjacency_matrix(self, dtype="uint8") -> sparse.csr_array:
        """Returns the graph's adjacency matrix.

        Since the matrix can become quite large (~10 ** 5 x 10 ** 5) the
        matrix is by default returned as a CSR sparse matrix.
        """
        return sparse.csr_matrix(
            (
                np.ones(self.neighbor_pairs.shape[0], dtype=dtype),
                (self.neighbor_pairs[:, 0], self.neighbor_pairs[:, 1]),
            )
        )

    def distance_matrix(self, dtype: str = "float64") -> sparse.csr_array:
        """Returns matrix containing inter-node distances.

        Since the matrices can become quite large (~10 ** 5 x 10 ** 5) the
        matrix is by default returned as a CSR sparse matrix.
        """
        return sparse.csr_matrix(
            (
                (
                    self.node_centeroid_distances.astype(dtype)
                    if np.dtype(dtype) != self.node_centeroid_distances.dtype
                    else self.node_centeroid_distances
                ),
                (self.neighbor_pairs[:, 0], self.neighbor_pairs[:, 1]),
            )
        )


def __make_GrainsToGraph(filepath: Path) -> GrainsToGraph:
    return GrainsToGraph(filepath=filepath)


def get_all_graphs(
    directory_path: Path,
    file_globbing_pattern: str = "*.npz",
    n_jobs: int = 1,
    show_progress: bool = False,
) -> list[GrainsToGraph]:

    all_graphs: list[GrainsToGraph]

    if show_progress:
        from tqdm.contrib.concurrent import process_map

        all_paths = list(directory_path.glob(file_globbing_pattern))

        all_graphs = process_map(
            __make_GrainsToGraph,
            all_paths,
            max_workers=n_jobs,
            # for proper visualization of the progress bar
            total=len(all_paths),
        )

    else:
        from joblib import parallel_config, Parallel, delayed

        with parallel_config(n_jobs=n_jobs):
            all_graphs = Parallel()(
                delayed(GrainsToGraph)(filepath=filepath)
                for filepath in directory_path.glob(file_globbing_pattern)
            )

    return all_graphs


def agg_potential(
    graph: GrainsToGraph,
    potential: np.ndarray,
    agg_type: str = "mean",
    background: tuple[float, float] | None = None,
):
    """Aggregate the (real-valued) potential given on the 3D grid within all nodes (grains).

    :: Parameters
        graph: GrainsToGraph
            Graph corresponding to the potential

        potential: np.ndarray
            potential given as a 3D NumPy Array of real values

        agg_type: str
          type of aggregation specified as a string
          * "min"       --> take min value in each grain
          * "max"       --> take max value in each grain
          * "mean"      --> mean value within each grain
          * "median"    --> median value within each grain
          * "at_center" --> value at designated grain center

        background: tuple[float, float] | None
            Specify potential value at left / right electrode or None. If this parameter is not None
            a linearly increasing potential from left to right electrode is subtracted from the input
            potential.

            NOTE: Currently (2025-02) it is always assumed that the potential is increasing
                  from left to to right along the axis-0 direction.
    """
    assert (
        graph.grain_labels.shape == potential.shape
    ), f"structure shape {graph.grain_labels.shape} != potential shape {potential.shape}"

    agg_pot: np.ndarray = np.zeros((graph.volumes.size,), dtype="float64")

    # TODO: Allow subtraction of linear potential along each possible
    #       direction.
    # Subtract linear potential background along axis-0.
    if background is not None:
        assert (
            potential.shape[0] == potential.shape[1]
            and potential.shape[0] == potential.shape[2]
        )
        pot_left, pot_right = background[:2]
        potential = potential - np.tile(
            (
                pot_left
                + (pot_right - pot_left)
                / (potential.shape[0] - 1)
                * np.arange(potential.shape[0], dtype="float64")
            ).reshape((-1, 1, 1)),
            (1, *potential.shape[1:]),
        )

    if agg_type in (
        "min",
        "max",
        "mean",
        "median",
    ):
        agg_potential_impl(
            graph.grain_labels,
            graph.volumes,
            potential,
            agg_pot,
            agg_type,
        )
    elif agg_type == "at_center":
        agg_pot[:] = potential[tuple(graph.centeroids.astype(int).T)]
    else:
        raise ValueError(
            f"{agg_type = } is not a valid option for potential aggregration."
        )

    return agg_pot
