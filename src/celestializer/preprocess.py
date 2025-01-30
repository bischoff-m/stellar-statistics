import cv2
import numpy as np
import celestializer as cl
from scipy.ndimage import convolve
from numpy.typing import NDArray


def correct_vignetting(
    img: cl.ImageNBit, img_white: cl.ImageNBit
) -> cl.ImageNBit:
    # Divide by white image
    img_white = img_white.channel("green")
    assert img_white.bit_depth == 1, "White image must have values in [0, 1]"
    mean = np.nanmean(img_white)
    std = np.nanstd(img_white)
    max_val = np.nanmax(img_white)
    if max_val < mean + 3 * std:
        raise ValueError("White image has too many saturated pixels")
    img_white /= np.nanpercentile(img_white, 99.999)
    img_white = img_white.clip(0, 1)
    img_new = img / img_white
    img_new = img_new.clip(0, 1)
    return img_new


def correct_skyglow(img: cl.ImageNBit) -> cl.ImageNBit:
    assert not np.isnan(img).any(), "Image must not contain NaN values"
    kernel_size = min(img.shape) // 10 * 2 + 1
    img_blur = cv2.GaussianBlur(
        img, (kernel_size, kernel_size), kernel_size / 4
    )
    img_blur -= img_blur.min()
    img_blur = cl.ImageNBit(image=img_blur, bit_depth=img.bit_depth)
    img = (img - img_blur).clip(0, 2**img.bit_depth - 1)
    return img


def find_defects(
    img: cl.ImageNBit, threshold_hot: float = 3, threshold_dead: float = 1.2
) -> tuple[NDArray[np.bool], NDArray[np.bool]]:
    # Find pixels that deviate from their neighbors by more than threshold *
    # standard deviation

    # Right now, this assumes that the image only consists of the green channel
    assert img.bit_depth == 1, "Image values must be between 0 and 1"
    # Set up a kernel that averages neighboring pixels
    kernel = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=np.float32,
    )
    num_neighbors = kernel.sum()
    kernel /= num_neighbors

    # Calculate the mean and standard deviation of the image
    img_mean = convolve(img, kernel, mode="constant", cval=np.nan)
    img_sq_mean = convolve(img**2, kernel, mode="constant", cval=np.nan)
    img_var = (img_sq_mean - img_mean**2 / num_neighbors) / num_neighbors
    img_std = np.sqrt(img_var)

    # Find hot and dead pixels
    mask_hot = cl.ImageNBit(np.zeros_like(img, dtype=np.float32), 1)
    mask_dead = cl.ImageNBit(np.zeros_like(img, dtype=np.float32), 1)
    mask_hot[img > img_mean + threshold_hot * img_std] = 1
    mask_dead[img < img_mean - threshold_dead * img_std] = 1

    return mask_hot.astype(np.bool), mask_dead.astype(np.bool)


def replace_defects(img: cl.ImageNBit, mask: NDArray[np.bool]) -> cl.ImageNBit:
    # Replace with median
    img_median = np.median(img[~mask & ~np.isnan(img)])
    img_out = img.copy()
    img_out[mask] = img_median

    # Replace with average of neighbors
    kernel = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=np.float32,
    )
    kernel /= kernel.sum()
    img_out[mask] = convolve(img_out, kernel, mode="constant", cval=np.nan)[
        mask
    ]
    return img_out


def preprocess(
    img_sky: cl.ImageNBit,
    img_white: cl.ImageNBit | None = None,
    img_black: cl.ImageNBit | None = None,
) -> cl.ImageNBit:
    img = img_sky.channel("green")
    assert img.bit_depth == 1, "Image must have values in [0, 1]"

    # Correct vignetting
    if img_white:
        img = correct_vignetting(img, img_white)

    # Remove noise
    if img_black:
        # Subtract mean of black image
        img_black = img_black.channel("green")
        img = img - np.nanmean(img_black)
        img = img.clip(0, 1)

    # Correct hot and dead pixels
    mask_hot, mask_dead = cl.find_defects(img)
    img = cl.replace_defects(img, mask_hot | mask_dead)

    # Normalize and interpolate
    img -= np.nanmin(img)
    img /= np.nanmax(img)
    img = img.green_interpolated()

    # Correct skyglow
    img = correct_skyglow(img)
    return img
