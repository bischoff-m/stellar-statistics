import cv2
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import celestializer as cl
import numpy as np
from astrometry import Match
from numpy.typing import NDArray

from celestializer.image import ImageNBit


class StarCenter(BaseModel):
    x: int
    y: int


class StarMag(StarCenter):
    magnitude: float


class StarPixels(StarCenter):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bounding_box: cv2.typing.Rect
    mask: NDArray[np.bool]


def predict_magnitude(img: ImageNBit, star: StarPixels) -> StarMag:
    # Extract star pixels
    star_pixels = img[
        star.bounding_box[1] : star.bounding_box[1] + star.bounding_box[3],
        star.bounding_box[0] : star.bounding_box[0] + star.bounding_box[2],
    ]
    star_pixels = star_pixels[star.mask]
    magnitude = -2.5 * np.log10(star_pixels.mean())
    return StarMag(x=star.x, y=star.y, magnitude=magnitude)


def show_stars(
    img: cl.ImageNBit,
    stars: list[StarCenter] | list[StarMag],
    radius: int = 40,
    show_magnitude: bool = True,
) -> cl.ImageNBit:
    img = img.to_bitdepth(8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    half_radius = radius // 2
    for star in stars:
        img = cv2.rectangle(
            img,
            (star.x - half_radius, star.y - half_radius),
            (star.x + half_radius, star.y + half_radius),
            (0, 255, 0),
            2,
        )
        if not show_magnitude or not isinstance(star, StarMag):
            continue
        img = cv2.putText(
            img,
            f"{star.magnitude:.2f}",
            (star.x - half_radius - 5, star.y - half_radius - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    return cl.ImageNBit(image=img, bit_depth=8)


def stars_by_flood_fill(img: cl.ImageNBit) -> list[cl.StarPixels]:
    assert img.ndim == 2, "Image must be 2D"
    # Must be CV_8U for flood fill
    assert img.bit_depth == 8, "Image must be 8-bit"

    # Find darkest pixel
    min_loc = np.unravel_index(np.argmin(img), img.shape)
    # Flood fill from darkest pixel
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    # TODO: Find upDiff dynamically by counting number of single pixels in mask
    cv2.floodFill(
        img,
        mask,
        min_loc,
        255,
        loDiff=100,
        upDiff=3,
        flags=cv2.FLOODFILL_MASK_ONLY,
    )
    # Remove border and invert mask
    mask = 1 - mask[1:-1, 1:-1]

    # Remove single pixels from mask
    kernel = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find connected components
    n_stars, labels = cv2.connectedComponents(mask, connectivity=4)

    # Find star centers using moments
    stars = []
    for i in tqdm(range(1, n_stars)):
        mask_star = labels == i
        mask_star_int = mask_star.astype(np.uint8)
        moments = cv2.moments(mask_star_int)
        if moments["m00"] == 0:
            continue
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
        # Extract bounding box part of the mask
        bbox = cv2.boundingRect(mask_star_int)
        stars.append(
            cl.StarPixels(
                x=x,
                y=y,
                bounding_box=bbox,
                mask=mask_star[
                    bbox[1] : bbox[1] + bbox[3],
                    bbox[0] : bbox[0] + bbox[2],
                ].copy(),
            )
        )
    return stars


def stars_by_template(
    img: cl.ImageNBit, threshold: float = 0.4
) -> list[StarCenter]:
    assert not np.isnan(img).any(), "Image must not contain NaN values"
    # 2D Gaussian kernel as template
    kernel_size = 21
    template = cv2.getGaussianKernel(kernel_size, 5)
    template = template * template.T
    template /= template.max()
    template *= 255
    template = cl.ImageNBit(image=template, bit_depth=8).cutoff(0.5)
    template = template.astype(np.float32)

    # Match template
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)

    def remove_duplicates(arr: np.ndarray, tolerance: int) -> np.ndarray:
        sort_indices = np.lexsort((arr[:, 0], arr[:, 1]))
        arr = arr[sort_indices]
        diff = np.diff(arr, axis=0)
        dropped = np.sqrt((diff**2).sum(axis=1)) < tolerance / 2
        return arr[~np.concatenate(([False], dropped))]

    rect_size = 40
    # Array of (y, x) coordinates, will be flipped later
    matches = np.asarray([loc[0], loc[1]]).T
    matches = remove_duplicates(matches, rect_size)
    matches = np.flip(matches, axis=1)
    matches = remove_duplicates(matches, rect_size)
    matches += kernel_size // 2
    return [StarCenter(x=x, y=y) for x, y in matches]


def fill_holes(img: cl.ImageNBit, mask: NDArray[np.bool]) -> cl.ImageNBit:
    # Get most common color in img
    (hist, _) = np.histogram(img, bins=np.arange(257))
    color = np.argmax(hist)

    # Fill mask with color
    img_filled = img.copy()
    img_filled[mask] = color
    return cl.ImageNBit(img_filled, img.bit_depth)


def fit_vignette(img: cl.ImageNBit) -> LinearRegression:
    pass


def correct_vignette(img: cl.ImageNBit) -> cl.ImageNBit:
    kernel_size = min(img.shape) // 10 * 2 + 1
    img_blur = cv2.GaussianBlur(
        img, (kernel_size, kernel_size), kernel_size / 4
    )
    img_blur -= img_blur.min()
    img_blur = cl.ImageNBit(image=img_blur, bit_depth=img.bit_depth)
    img = (img - img_blur).clip(0, 2**img.bit_depth - 1)
    return img


def find_coordinates(img: cl.ImageNBit, camera: cl.CameraInfo) -> Match:
    pass


def preprocess(
    img_sky: cl.ImageNBit,
    img_black: cl.ImageNBit | None = None,
    img_white: cl.ImageNBit | None = None,
) -> cl.ImageNBit:
    img = img_sky.channel("green")
    assert img.bit_depth == 1, "Image must have values in [0, 1]"

    if img_black:
        # Subtract mean of black image
        img_black = img_black.channel("green")
        img = img - np.nanmean(img_black)
        img = img.clip(0, 1)

    if img_white:
        # Apply vignetting correction
        model = fit_vignette(img_white)
        img = correct_vignette(img, model)

    # Interpolate and normalize
    img = img.green_interpolated()
    img /= img.max()
    return img


def find_star_magnitudes(img: cl.ImageNBit) -> list[StarMag]:
    pass
