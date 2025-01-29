import datetime
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel
from rawkit.raw import Raw

from .ImageNBit import ImageNBit


class CameraInfo:
    bit_depth: int = 14
    filter_pattern: str = "RGGB"


class Metadata(BaseModel):
    width: int
    height: int
    focal_length: float
    aperture: float
    shutter: float
    iso: int
    timestamp: datetime.datetime
    camera: str
    bit_depth: int


def require_loaded(func):
    def wrapper(self: "RawImage", *args, **kwargs):
        if not self.is_loaded():
            self.load()
        assert self.is_loaded(), "Image data could not be loaded."
        return func(self, *args, **kwargs)

    return wrapper


class RawImage:
    metadata: Metadata
    _raw: ImageNBit | None = None

    def __init__(self, filepath: Path, bit_depth: int = CameraInfo.bit_depth):
        if bit_depth < 1 or bit_depth > 16:
            raise ValueError("Bit depth must be between or equal to 1 and 16.")
        self.filepath = filepath
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
                bit_depth=bit_depth,
            )

    def load(self) -> "RawImage":
        """Load the raw image data as a numpy array.

        Called by the `require_loaded` decorator if the image data is not
        loaded. It is not called in the constructor because it can be slow and
        we may not need the image data for all instances of RawImage.
        """
        with Raw(self.filepath.as_posix()) as raw:
            # Read raw image data
            raw_data = np.asarray(raw.raw_image(), dtype=np.uint16)
            self._raw = ImageNBit(
                image=raw_data, bit_depth=self.metadata.bit_depth
            )
        return self

    def is_loaded(self) -> bool:
        """Check if the raw image data is loaded.

        Returns
        -------
        bool
            True if the raw image data is loaded.
        """
        return self._raw is not None

    @require_loaded
    def raw(self) -> ImageNBit:
        """Getter to load the raw image data if not already loaded.

        Returns
        -------
        ImageNBit
            Raw image data.
        """
        return self._raw

    @require_loaded
    def to_rgb_rawkit(self) -> ImageNBit:
        with Raw(self.filepath.as_posix()) as raw:
            img = np.asarray(raw.to_buffer())
            img = img.reshape(self.metadata.height, self.metadata.width, 3)
        return ImageNBit(image=img, bit_depth=8)

    @require_loaded
    def to_rgb_cv2(self) -> ImageNBit:
        img = self._raw.to_bitdepth(16)
        if CameraInfo.filter_pattern == "RGGB":
            img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
        else:
            raise NotImplementedError(
                f"Filter pattern {CameraInfo.filter_pattern} not implemented."
            )
        return ImageNBit(image=img, bit_depth=16).to_bitdepth(
            self._raw.bit_depth
        )
