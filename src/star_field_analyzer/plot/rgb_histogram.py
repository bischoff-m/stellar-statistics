from typing import List

import matplotlib.pyplot as plt
import numpy as np

from star_field_analyzer.image import ImageNBit


def rgb_histogram(
    img: ImageNBit,
    bit_depth: int | None = None,
    yaxis_relative: bool = False,
    n_bins: int = 1000,
):
    if bit_depth is None:
        bit_depth = img.bit_depth
    else:
        img = img.to_bitdepth(bit_depth)

    # Histogram of the raw image
    channels = {
        c: img.channel(c).image.flatten() for c in ["red", "green", "blue"]
    }
    channels = {c: ch[~np.isnan(ch)] for c, ch in channels.items()}
    bins = np.linspace(0, 2**bit_depth - 1, min(2**bit_depth, n_bins))
    print(f"Number of bins: {len(bins)}")

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs: List[plt.Axes]

    for idx, (color, channel) in enumerate(channels.items()):
        # Calculate the mean and standard deviation
        mean = np.mean(channel)
        std = np.std(channel)
        print(f"{color.capitalize()} Channel: mean={mean:.2f}, std={std:.2f}")
        # Plot the histogram
        axs[idx].hist(
            channels[color],
            bins,
            color=color,
            label=f"{color.capitalize()}\n$mu$={mean:.2f}\n$sigma$={std:.2f}",
            weights=np.ones_like(channel) / len(channel)
            if yaxis_relative
            else None,
        )

        axs[idx].grid()
        axs[idx].set_axisbelow(True)
        axs[idx].legend()
        axs[idx].set_title(f"{color.capitalize()} Channel")
        line_param = {"color": "#0008", "linestyle": "dashed", "linewidth": 1}
        axs[idx].axvline(mean, **line_param)
        axs[idx].axvline(mean - std, **line_param)
        axs[idx].axvline(mean + std, **line_param)
        axs[idx].set_xlim(0, 2**bit_depth)

    fig.tight_layout()
    plt.close()
    return fig
