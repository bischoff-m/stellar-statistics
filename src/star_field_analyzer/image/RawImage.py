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


def require_loaded(func):
    def wrapper(self, *args, **kwargs):
        if not self.is_loaded():
            self.load()
        assert self.is_loaded(), "Image data could not be loaded."
        return func(self, *args, **kwargs)

    return wrapper


class RawImage:
    def __init__(self, filepath: Path, bit_depth: int = CameraInfo.bit_depth):
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
            )
            # Raw image data
            self._raw = ImageNBit(image=np.empty(0), bit_depth=bit_depth)

    def load(self) -> "RawImage":
        """Load the raw image data as a numpy array.

        Called by the `require_loaded` decorator if the image data is not
        loaded. It is not called in the constructor because it can be slow and
        we may not need the image data for all instances of RawImage.
        """
        with Raw(self.filepath.as_posix()) as raw:
            # Read raw image data
            raw_data = np.asarray(raw.raw_image(), dtype=np.uint16)
            self._raw = ImageNBit(image=raw_data, bit_depth=self._raw.bit_depth)
        return self

    def is_loaded(self) -> bool:
        """Check if the raw image data is loaded.

        Returns
        -------
        bool
            True if the raw image data is loaded.
        """
        return self._raw.image.size > 0

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
    def dead_pixels(self, threshold: float = 0.5) -> np.ndarray:
        dead = self._raw.image < self._raw.norm_to_nbit(threshold)
        assert (
            np.sum(dead) < self.metadata.width * self.metadata.height * 0.1
        ), "More than 10% of pixels are dead. Did you use a white image?"
        return dead

    @require_loaded
    def hot_pixels(self, threshold: float = 0.5) -> np.ndarray:
        hot = self._raw.image > self._raw.norm_to_nbit(threshold)
        assert (
            np.sum(hot) < self.metadata.width * self.metadata.height * 0.1
        ), "More than 10% of pixels are hot. Did you use a black image?"
        return hot

    @require_loaded
    def to_rgb_rawkit(self) -> ImageNBit:
        with Raw(self.filepath.as_posix()) as raw:
            img = np.asarray(raw.to_buffer())
            img = img.reshape(self.metadata.height, self.metadata.width, 3)
        return ImageNBit(image=img, bit_depth=8)

    @require_loaded
    def to_rgb_cv2(self) -> ImageNBit:
        img = np.asarray(self._raw.image, dtype=np.uint16)
        if CameraInfo.filter_pattern == "RGGB":
            img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
        else:
            raise NotImplementedError(
                f"Filter pattern {CameraInfo.filter_pattern} not implemented."
            )
        return ImageNBit(image=img, bit_depth=self._raw.bit_depth)

    @require_loaded
    def green_interpolated(self) -> ImageNBit:
        """Interpolate green pixels in the raw image data.

        Sets the value of green pixels to the mean of the 8 neighboring pixels.

        Returns
        -------
        ImageNBit
            Image with interpolated green pixels.
        """
        raw = self.raw()
        green_nan = raw.channel("green").image
        if green_nan.shape[0] % 2 != 0:
            green_nan = green_nan[:-1, :]
        if green_nan.shape[1] % 2 != 0:
            green_nan = green_nan[:, :-1]

        rows_odd = green_nan[::2, 1::2]
        rows_even = green_nan[1::2, ::2]
        green_only = np.zeros(
            (rows_odd.shape[0] + rows_even.shape[0], rows_odd.shape[1]),
        )
        green_only[::2, :] = rows_odd
        green_only[1::2, :] = rows_even

        filter_odd = np.asarray(
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 1, 0],
            ]
        )
        filter_even = np.asarray(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
            ]
        )
        # Every new pixel corresponds to the cell on the left
        new_odd = cv2.filter2D(green_only, -1, filter_odd)
        new_odd /= np.sum(filter_odd)
        # Every new pixel corresponds to the cell on the right
        new_even = cv2.filter2D(green_only, -1, filter_even)
        new_even /= np.sum(filter_even)

        new_img = green_nan.copy()
        new_img[::2, ::2] = new_odd[::2, :]
        new_img[1::2, 1::2] = new_even[1::2, :]

        return ImageNBit(image=new_img, bit_depth=raw.bit_depth)
