from typing import Callable

import matplotlib.pyplot as plt
from tqdm import tqdm

SubplotFunction = Callable[[plt.Axes], None]


def subplots_grid(
    subplot_funcs: list[list[SubplotFunction]],
    figsize: tuple[int, int],
    dpi: int = 100,
) -> plt.Figure:
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


def wrap_subplot(func):
    def wrapper(*args, **kwargs):
        if "ax" in kwargs:
            return func(*args, **kwargs)
        else:
            return lambda ax: func(*args, ax=ax, **kwargs)

    return wrapper
