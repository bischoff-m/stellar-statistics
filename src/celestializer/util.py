from typing import List
import pandas as pd
import celestializer as cl
import numpy as np
from numpy.typing import NDArray
import cv2


def list_observations() -> List[cl.RawImage]:
    return [
        cl.RawImage(file)
        for file in [
            *cl.Paths.observations.glob("**/*"),
            *cl.Paths.calibration.glob("**/*"),
        ]
        if file.suffix.lower() in [".cr2", ".dng"]
    ]


def raw_to_df(files: List[cl.RawImage]) -> pd.DataFrame:
    data = []
    for file in files:
        data.append(
            {
                "filepath": file.filepath.relative_to(cl.Paths.data).as_posix(),
            }
            | file.metadata.model_dump()
        )
    df = pd.DataFrame(data)
    df.sort_values("filepath", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def grab_series(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Get continuous series of images that only differ in column 'by'.

    Returns the longest (or first if tie) series of images that were taken
    after each other and have the same values in all columns except 'by'. This
    is helpful if you took a series of images to test a specific parameter (e.g.
    different ISO values).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the images.
    by : str
        Column to segment the images by.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the longest series of images that only differ in
        column 'by'.
    """
    if len(df) == 0:
        raise ValueError("DataFrame is empty.")

    # Sort df by timestamp
    df = df.sort_values("timestamp")
    fixed_columns = [
        "width",
        "height",
        "focal_length",
        "aperture",
        "shutter",
        "iso",
        "camera",
    ]
    fixed_columns.remove(by)
    segmented = []
    cur_start = 0
    # Find continuous series of images that differs in column 'by' and has the
    # same values in other columns
    for i in range(1, len(df)):
        same_by_col: bool = df.iloc[i][by] == df.iloc[i - 1][by]
        diff_other_cols: bool = any(
            df[fixed_columns].iloc[i] != df[fixed_columns].iloc[i - 1]
        )
        if same_by_col or diff_other_cols:
            segmented.append((cur_start, i))
            cur_start = i
    segmented.append((cur_start, len(df)))

    segmented = [
        (start, end)
        for start, end in segmented
        if df.iloc[start:end][fixed_columns].nunique().max() == 1
    ]
    # Sort by length
    segmented.sort(key=lambda x: x[1] - x[0], reverse=True)
    # Return the longest series
    start, end = segmented[0]
    return df.iloc[start:end].copy().sort_values(by)


def fill_holes(img: cl.ImageNBit, mask: NDArray[np.bool]) -> cl.ImageNBit:
    assert img.bit_depth == 8, "Not sure if this works for other bit depths"
    assert not np.isnan(img).any(), "Not sure if this works for NaN values"
    # Get most common color in img
    (hist, _) = np.histogram(img, bins=np.arange(257))
    color = np.argmax(hist)

    # Fill mask with color
    img_filled = img.copy()
    img_filled[mask] = color
    return cl.ImageNBit(img_filled, img.bit_depth)


def show_stars(
    img: cl.ImageNBit,
    stars: list[cl.StarCenter] | list[cl.StarMag],
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
        if not show_magnitude or not hasattr(star, "magnitude"):
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
