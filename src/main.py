# %%
from pathlib import Path
from typing import List
from pydantic import BaseModel
from rawkit.raw import Raw
from definitions import Paths
import pandas as pd
import numpy as np
import datetime
from PIL import Image


class CameraInfo:
    bit_depth: int = 14


class Metadata(BaseModel):
    width: int
    height: int
    focal_length: float
    aperture: float
    shutter: float
    iso: int
    timestamp: datetime.datetime
    camera: str


class RawImage:
    def __init__(self, filepath: Path, bit_depth: int = CameraInfo.bit_depth):
        self.filepath = filepath
        self.bit_depth = bit_depth
        with Raw(filepath.as_posix()) as raw:
            self.metadata = Metadata(
                width=raw.metadata.width,
                height=raw.metadata.height,
                focal_length=raw.metadata.focal_length,
                aperture=raw.metadata.aperture,
                shutter=np.round(raw.metadata.shutter, decimals=8),
                iso=raw.metadata.iso,
                timestamp=datetime.datetime.fromtimestamp(
                    raw.metadata.timestamp
                ),
                camera=f"{raw.metadata.make.decode('utf-8')} {raw.metadata.model.decode('utf-8')}",
            )
            self.raw = np.empty(0)

    def load(self):
        """Load the raw image data as a numpy array.

        This method should be called before using the image data. It is not
        called in the constructor because it can be slow and we may not need the
        image data for all instances of RawImage.
        """
        with Raw(self.filepath.as_posix()) as raw:
            self.raw = np.asarray(raw.raw_image()) / (2**self.bit_depth - 1)

    def is_loaded(self):
        return self.raw.size > 0

    def dead_pixels(self, threshold: float = 0.5) -> np.ndarray:
        assert self.is_loaded(), "Image data not loaded. Call load() first."
        dead = self.raw < threshold
        assert (
            np.sum(dead) < self.metadata.width * self.metadata.height * 0.1
        ), "More than 10% of pixels are dead. Did you use a white image?"
        return dead

    def hot_pixels(self, threshold: float = 0.5) -> np.ndarray:
        assert self.is_loaded(), "Image data not loaded. Call load() first."
        hot = self.raw > threshold
        assert (
            np.sum(hot) < self.metadata.width * self.metadata.height * 0.1
        ), "More than 10% of pixels are hot. Did you use a black image?"
        return hot

    def show_brightness(self):
        assert self.is_loaded(), "Image data not loaded. Call load() first."
        image = Image.fromarray((self.raw * 255).astype(np.uint8))
        image.show()


def list_observations() -> List[RawImage]:
    return [RawImage(file) for file in Paths.observations.glob("**/*.CR2")]


def raw_to_df(files: List[RawImage]) -> pd.DataFrame:
    data = []
    for file in files:
        data.append(
            {
                "filepath": file.filepath.relative_to(
                    Paths.observations
                ).as_posix(),
            }
            | file.metadata.model_dump()
        )
    df = pd.DataFrame(data)
    df.sort_values("filepath", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# %%
files = list_observations()
df = raw_to_df(files)
df

# %%
img = RawImage(Paths.observations / "2024-11-05/IMG_3320.CR2")
img.load()
img.show_brightness()
