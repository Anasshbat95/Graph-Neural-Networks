import numpy as np

from matplotlib import pyplot as plt


from gnn_tools.preprocessing import GrainsToGraph


def __assert_equal_shapes(graph: GrainsToGraph, agg_potential: np.ndarray) -> None:
    assert (graph.centeroids.shape[0] == agg_potential.shape[0]) and (
        graph.volumes.shape[0] == agg_potential.size
    ), "Size of `agg_potential` does not match size of `graph.centeroids` or `graph.volumes`"


def plot_structure_3d(
    graph: GrainsToGraph,
    agg_potential: np.ndarray,
    show_node_labels: bool = False,
    node_size_scaling_factor: float = 1e-2,
    cmap: str = "viridis",
    title: str | None = None,
) -> tuple:
    __assert_equal_shapes(graph, agg_potential)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    if title is not None:
        assert isinstance(title, str)
        ax.set_title(title)

    # ax.set_title("Nodes with color and size given by node volumes")
    ax.set_xlabel("axis 0")
    ax.set_ylabel("axis 1")
    ax.set_zlabel("axis 2")

    for idx, ((x, y, z), vol) in enumerate(
        zip(
            graph.centeroids,
            graph.volumes,
        )
    ):
        if show_node_labels:
            ax.text(x, y, z, f"{idx}")
        p = ax.scatter(
            x,
            y,
            z,
            s=vol * node_size_scaling_factor,
            c=agg_potential[idx],
            cmap=cmap,
            vmin=agg_potential.min(),
            vmax=agg_potential.max(),
        )

    fig.colorbar(p)

    return fig, ax


def plot_structure_project_2d(
    graph: GrainsToGraph,
    agg_potential: np.ndarray,
    axis: int = 0,  # axis along which to project
    node_size_scaling_factor: float = 1e-2,
    figsize: tuple[int] = (10, 5),
    min_alpha: float = 0.25,
    title: str | None = None,
    cmap="viridis",
):
    # min / max coordinate along projection direction
    min_coord, max_coord = np.sort(graph.centeroids[:, axis])[[0, -1]]

    def alpha_mapper(x):
        """Map coordinate to alpha value (opacity)."""
        return np.max((min_alpha, (x - min_coord) / (max_coord - min_coord)))

    __assert_equal_shapes(graph, agg_potential)

    indices: list[int]
    if axis == 0:
        indices = [1, 2]
    elif axis == 1:
        indices = [0, 2]
    elif axis == 2:
        indices = [0, 1]
    else:
        raise ValueError(f"Invalid value for {axis = }")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if title is not None:
        assert isinstance(title, str)
        ax.set_title(title)

    ax.set_xlabel("horizontal position")
    ax.set_ylabel("vertical position")

    for idx, coord in enumerate(graph.centeroids):
        p = ax.scatter(
            *coord[indices],
            c=agg_potential[idx],
            s=graph.volumes[idx] * node_size_scaling_factor,
            vmin=agg_potential.min(),
            vmax=agg_potential.max(),
            alpha=alpha_mapper(x=coord[axis]),
            cmap=cmap,
        )

    fig.colorbar(p)

    return fig, ax
