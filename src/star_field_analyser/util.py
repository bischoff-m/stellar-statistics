from typing import List

import pandas as pd

from star_field_analyser.definitions import Paths
from star_field_analyser.image import RawImage


def list_observations() -> List[RawImage]:
    return [
        RawImage(file)
        for file in [
            *Paths.observations.glob("**/*"),
            *Paths.calibration.glob("**/*"),
        ]
        if file.suffix.lower() in [".cr2", ".dng"]
    ]


def raw_to_df(files: List[RawImage]) -> pd.DataFrame:
    data = []
    for file in files:
        data.append(
            {
                "filepath": file.filepath.relative_to(Paths.data).as_posix(),
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
    df.sort_values("timestamp", inplace=True)
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
