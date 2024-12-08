from typing import List

import pandas as pd

from definitions import Paths
from star_field_analyzer.image import RawImage


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
                "filepath": file.filepath.relative_to(
                    Paths.root / "data"
                ).as_posix(),
            }
            | file.metadata.model_dump()
        )
    df = pd.DataFrame(data)
    df.sort_values("filepath", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
