import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel
from rawkit.raw import Raw


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


class ImageNBit:
    def __init__(self, image: np.ndarray, bit_depth: int):
        self.image = image
        self.bit_depth = bit_depth

    def to_bitdepth(self, bit_depth: int) -> "ImageNBit":
        if bit_depth == self.bit_depth:
            return self
        new_img = self.image / (2**self.bit_depth - 1) * (2**bit_depth - 1)
        if bit_depth <= 8:
            new_img = np.round(new_img).astype(np.uint8)
        else:
            new_img = np.round(new_img).astype(np.uint16)

        return ImageNBit(new_img, bit_depth)

    def show(self):
        image = Image.fromarray(self.to_bitdepth(8).image)
        image.show()

    def norm_to_nbit(self, value: float) -> int:
        return np.round(value * (2**self.bit_depth - 1)).astype(np.uint16)


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
            self.raw = ImageNBit(np.empty(0), bit_depth)

    def load(self) -> "RawImage":
        """Load the raw image data as a numpy array.

        This method should be called before using the image data. It is not
        called in the constructor because it can be slow and we may not need the
        image data for all instances of RawImage.
        """
        with Raw(self.filepath.as_posix()) as raw:
            # Read raw image data
            raw_data = np.asarray(raw.raw_image(), dtype=np.uint16)
            self.raw = ImageNBit(raw_data, self.raw.bit_depth)
        return self

    def is_loaded(self) -> bool:
        return self.raw.image.size > 0

    @require_loaded
    def get_raw(self) -> ImageNBit:
        return self.raw

    @require_loaded
    def dead_pixels(self, threshold: float = 0.5) -> np.ndarray:
        dead = self.raw.image < self.raw.norm_to_nbit(threshold)
        assert (
            np.sum(dead) < self.metadata.width * self.metadata.height * 0.1
        ), "More than 10% of pixels are dead. Did you use a white image?"
        return dead

    @require_loaded
    def hot_pixels(self, threshold: float = 0.5) -> np.ndarray:
        hot = self.raw.image > self.raw.norm_to_nbit(threshold)
        assert (
            np.sum(hot) < self.metadata.width * self.metadata.height * 0.1
        ), "More than 10% of pixels are hot. Did you use a black image?"
        return hot

    @require_loaded
    def to_rgb_rawkit(self) -> ImageNBit:
        with Raw(self.filepath.as_posix()) as raw:
            img = np.asarray(raw.to_buffer())
            img = img.reshape(self.metadata.height, self.metadata.width, 3)
        return ImageNBit(img, 8)

    @require_loaded
    def to_rgb_cv2(self) -> ImageNBit:
        img = np.asarray(self.raw.image, dtype=np.uint16)
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
        return ImageNBit(img, self.raw.bit_depth)
