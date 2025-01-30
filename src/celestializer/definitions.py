from pathlib import Path
from pydantic import BaseModel, ConfigDict
import cv2
from numpy.typing import NDArray
import numpy as np
import celestializer as cl


class Paths:
    root = Path(__file__).parent.parent.parent
    data = root / "data"
    observations = data / "observations"
    calibration = data / "calibration"
    saved = data / "saved"


class StarCenter(BaseModel):
    x: int
    y: int


class StarPixelSum(StarCenter):
    pixel_sum: float


class StarMag(StarCenter):
    magnitude: float


class SkyCoordMag(BaseModel):
    """Sky coordinates with magnitude. RA and Dec are in degrees."""

    ra: float
    dec: float
    magnitude: float


class StarMasked(StarCenter):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: cv2.typing.Rect
    mask: NDArray[np.bool]

    def sum_pixels(self, img: cl.ImageNBit) -> StarPixelSum:
        assert img.bit_depth == 1, "Image must have values in [0, 1]"
        star_pixels = img[
            self.bbox[1] : self.bbox[1] + self.bbox[3],
            self.bbox[0] : self.bbox[0] + self.bbox[2],
        ]
        star_pixels = star_pixels[self.mask]
        pixel_sum = star_pixels.sum()
        return StarPixelSum(x=self.x, y=self.y, pixel_sum=pixel_sum)

    def to_image(self, img: cl.ImageNBit, padding: int = 0) -> cl.ImageNBit:
        # Cut out star from image
        top = max(0, self.bbox[1] - padding)
        left = max(0, self.bbox[0] - padding)
        bottom = min(img.shape[0], self.bbox[1] + self.bbox[3] + padding)
        right = min(img.shape[1], self.bbox[0] + self.bbox[2] + padding)
        star_img = img[top:bottom, left:right].copy()

        # Convert image to 3 channels
        star_color = np.zeros(
            (star_img.shape[0], star_img.shape[1], 3), dtype=img.dtype
        )
        mask_pad = np.pad(self.mask, padding, mode="constant")
        for i in range(3):
            star_color[:, :, i] = star_img
        star_color[~mask_pad, 1] = 0
        star_color[~mask_pad, 2] = 0
        star_color[mask_pad, 0] = 0
        star_color[mask_pad, 2] = 0
        return cl.ImageNBit(star_color, img.bit_depth)
