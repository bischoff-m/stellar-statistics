import matplotlib.pyplot as plt
import numpy as np

from star_field_analyser.image import ImageNBit

from .plot_utils import SubplotFunction, wrap_subplot


def rgb_histograms(
    img: ImageNBit,
    bit_depth: int | None = None,
    n_bins: int = 600,
    title: str | None = None,
    yaxis_relative: bool = False,
    yaxis_log: bool = True,
) -> list[SubplotFunction]:
    if bit_depth is None:
        bit_depth = img.bit_depth
    else:
        img = img.to_bitdepth(bit_depth)

    return list(
        pixel_histogram(
            img,
            bit_depth=bit_depth,
            yaxis_relative=yaxis_relative,
            n_bins=n_bins,
            title=f"{title} ({c.capitalize()})",
            label=c,
            color=c,
            yaxis_log=yaxis_log,
        )
        for c, img in img.rgb_channels().items()
    )


@wrap_subplot
def pixel_histogram(
    img: ImageNBit,
    bit_depth: int | None = None,
    n_bins: int = 600,
    title: str | None = None,
    label: str | None = None,
    color: str | None = None,
    yaxis_relative: bool = False,
    yaxis_log: bool = True,
    ax: plt.Axes | None = None,
):
    if ax is None:
        raise ValueError("Must provide an axis to plot the histogram")
    if img.image.ndim != 2:
        raise ValueError("Input image must be raw sensor image 2D array")

    if bit_depth is None:
        bit_depth = img.bit_depth
    else:
        img = img.to_bitdepth(bit_depth)

    arr = img.image.flatten()
    arr = arr[~np.isnan(arr)]
    n_bins = min(n_bins, 2**bit_depth - 1) if bit_depth > 1 else n_bins
    bins = np.linspace(0, 2**bit_depth - 1, n_bins)

    # Calculate the mean and standard deviation
    mean = np.mean(arr)
    std = np.std(arr)
    label = f"{label}\n" if label is not None else ""
    # Plot the histogram
    ax.hist(
        arr,
        bins,
        color=color,
        label=f"{label}$mu$={mean:.2f}\n$sigma$={std:.2f}",
        weights=np.ones_like(arr) / len(arr) if yaxis_relative else None,
    )

    ax.grid()
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_xlim(0, 2**bit_depth - 1)
    if yaxis_log:
        ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)

    line_param = {"color": "#0008", "linestyle": "dashed", "linewidth": 1}
    ax.axvline(mean, **line_param)
    ax.axvline(mean - std, **line_param)
    ax.axvline(mean + std, **line_param)
