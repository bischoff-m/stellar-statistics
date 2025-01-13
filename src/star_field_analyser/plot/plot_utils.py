from typing import Any, Callable

import matplotlib.pyplot as plt
from tqdm import tqdm

SubplotFunction = Callable[[plt.Axes], None]


def subplots_grid(
    subplot_funcs: list[list[SubplotFunction]],
    figsize: tuple[int, int],
    dpi: int = 100,
) -> plt.Figure:
    assert len(subplot_funcs) > 0, "Must provide at least one subplot function"
    assert isinstance(subplot_funcs[0], list), "Subplots must be a 2D list"
    assert all(
        len(subplot_funcs[0]) == len(subplot_funcs[i])
        for i in range(1, len(subplot_funcs))
    ), "All rows must have the same number of subplots"

    nrows = len(subplot_funcs)
    ncols = len(subplot_funcs[0])
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        layout="tight",
        dpi=dpi,
    )

    # Reshape axs to be a 2D list
    if nrows == 1:
        axs: list[list[plt.Axes]] = [axs]
    if ncols == 1:
        axs: list[list[plt.Axes]] = [[ax] for ax in axs]

    flat_axs = [(i, j, axs[i][j]) for i in range(nrows) for j in range(ncols)]
    for i, j, ax in tqdm(flat_axs):
        subplot_funcs[i][j](ax)

    plt.close()
    return fig


def wrap_subplot(func) -> Callable[..., SubplotFunction | Any]:
    """Decorator to allow for the function to be called with or without an axis.

    If the function is called without an axis, it returns a lambda function that
    takes an axis as an argument. Otherwise, it calls the function with the
    provided axis. This is useful to define the plot parameters without having
    to provide the axis in the function call.

    Parameters
    ----------
    func : Callable
        Function to be decorated.
    """

    def wrapper(*args, **kwargs):
        if "ax" in kwargs:
            return func(*args, **kwargs)
        else:
            return lambda ax: func(*args, ax=ax, **kwargs)

    return wrapper
