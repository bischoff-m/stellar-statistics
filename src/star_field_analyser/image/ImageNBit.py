from pathlib import Path
from typing import Literal, Self

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageNBit(BaseModel, np.ndarray):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray = Field(..., allow_mutation=False)
    bit_depth: int = Field(..., allow_mutation=False, ge=1)
    pil_cache: Image.Image | None = None

    @model_validator(mode="after")
    def check_and_clip_image(self) -> Self:
        # Skip if the image is empty
        if self.image.shape == (0,):
            return self
        if self.image.min() < 0 or self.image.max() > 2**self.bit_depth - 1:
            raise ValueError(
                "Image values must be between 0 and 2^bit_depth - 1."
            )
        return self

    def to_bitdepth(
        self, bit_depth: int, astype: np.dtype | None = None
    ) -> "ImageNBit":
        """Lossy conversion of image bit depth.

        Parameters
        ----------
        bit_depth : int
            Target bit depth. If 1, the image is normalized to 0-1 and has type
            np.float64. If 8 or 16, the image is scaled to the target bit depth
            and has type np.uint8 or np.uint16, respectively.

        Returns
        -------
        ImageNBit
            Image with the target bit depth.
        """
        if bit_depth < 1:
            raise ValueError("Bit depth must be at least 1.")
        if bit_depth == self.bit_depth:
            return self
        if bit_depth == 1:
            new_img = self.image / (2**self.bit_depth - 1)
            return ImageNBit(
                image=new_img, bit_depth=bit_depth, pil_cache=self.pil_cache
            )

        new_img = self.image.copy()
        new_img[np.isnan(new_img)] = 0
        new_img = new_img / (2**self.bit_depth - 1) * (2**bit_depth - 1)
        new_img = np.round(new_img)
        new_type = (
            astype
            if astype is not None
            else (np.uint8 if bit_depth <= 8 else np.uint16)
        )
        new_img = np.asarray(new_img, dtype=new_type)

        return ImageNBit(
            image=new_img, bit_depth=bit_depth, pil_cache=self.pil_cache
        )

    def to_pil(self) -> Image.Image:
        """Convert the image to a PIL Image.

        Entries with NaN are replaced with 0 and the image is converted to
        8-bit. The result is cached for subsequent calls.

        Returns
        -------
        Image.Image
            PIL Image.
        """
        if self.pil_cache is not None:
            return self.pil_cache

        # PIL expects 8-bit images
        arr = self.to_bitdepth(8).image
        # Replace NaN with 0
        arr = np.nan_to_num(arr, nan=0)
        # To PIL and update cache
        self.pil_cache = Image.fromarray(arr)
        if arr.ndim == 2:
            self.pil_cache = self.pil_cache.convert("L")
        return self.pil_cache

    def show(self) -> None:
        """Display the image."""
        pil_img = self.to_pil()
        pil_img.show()

    def preview(self, max_cols_or_rows=800) -> Image.Image:
        """Downscale and display the image.

        Parameters
        ----------
        max_cols_or_rows : int
            Number of pixels that the longest side of the image can have.

        Returns
        -------
        Image.Image
            Downscaled image.
        """
        pil_img = self.to_pil()
        size = pil_img.size
        factor = max_cols_or_rows / max(size)
        pil_img = pil_img.resize((int(size[0] * factor), int(size[1] * factor)))
        return pil_img

    def save(self, filepath: Path) -> None:
        """Save the image to a file.

        Parameters
        ----------
        filepath : Path
            Path to save the image.
        """
        pil_img = self.to_pil()
        pil_img.save(filepath.as_posix())

    def norm_to_nbit(self, value: float) -> int:
        """Scale a value between 0 and 1 to the image bit depth.

        Parameters
        ----------
        value : float
            Value between 0 and 1 to

        Returns
        -------
        int
            Value scaled to the image bit depth.
        """
        return np.round(value * (2**self.bit_depth - 1)).astype(np.uint16)

    def channel(
        self, channel: Literal["red"] | Literal["green"] | Literal["blue"]
    ) -> "ImageNBit":
        """Get a single channel of the image.

        Sets all pixels except the specified channel to np.nan in a Bayer
        sensor image with a filter pattern of RGGB.

        Parameters
        ----------
        channel : Literal["red"] | Literal["green"] | Literal["blue"]
            Channel to keep.

        Returns
        -------
        ImageNBit
            Image with only the specified channel.
        """
        img = self.image.copy().astype(np.float64)
        if channel == "red":
            img[::2, 1::2] = np.nan  # G
            img[1::2, ::2] = np.nan  # G
            img[1::2, 1::2] = np.nan  # B
        elif channel == "green":
            img[::2, ::2] = np.nan  # R
            img[1::2, 1::2] = np.nan  # B
        elif channel == "blue":
            img[::2, ::2] = np.nan  # R
            img[::2, 1::2] = np.nan  # G
            img[1::2, ::2] = np.nan  # G
        else:
            raise ValueError(f"Invalid channel {channel}.")
        return ImageNBit(image=img, bit_depth=self.bit_depth)

    def rgb_channels(self) -> dict[str, "ImageNBit"]:
        """Get the red, green, and blue channels of the image.

        Returns
        -------
        tuple[ImageNBit, ImageNBit, ImageNBit]
            Red, green, and blue channels.
        """
        return {
            "red": self.channel("red"),
            "green": self.channel("green"),
            "blue": self.channel("blue"),
        }

    def green_interpolated(self) -> "ImageNBit":
        """Interpolate green pixels in the raw image data.

        Sets the value of green pixels to the mean of the 8 neighboring pixels.

        Returns
        -------
        ImageNBit
            Image with interpolated green pixels.
        """
        green_nan = self.channel("green").image
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

        # Clip to the original bit depth
        new_img = np.clip(new_img, 0, 2**self.bit_depth - 1)
        return ImageNBit(image=new_img, bit_depth=self.bit_depth)

    def __add__(self, other) -> "ImageNBit":
        arr = self.image.__add__(other)
        return ImageNBit(image=arr, bit_depth=self.bit_depth)

    def __sub__(self, other) -> "ImageNBit":
        arr = self.image.__sub__(other)
        return ImageNBit(image=arr, bit_depth=self.bit_depth)

    def __mul__(self, other) -> "ImageNBit":
        arr = self.image.__mul__(other)
        return ImageNBit(image=arr, bit_depth=self.bit_depth)

    def __truediv__(self, other) -> "ImageNBit":
        arr = self.image.__truediv__(other)
        return ImageNBit(image=arr, bit_depth=self.bit_depth)

    def __len__(self) -> int:
        return len(self.image)

    def __lt__(self, other) -> bool:
        return self.image < other

    def __gt__(self, other) -> bool:
        return self.image > other

    def __le__(self, other) -> bool:
        return self.image <= other

    def __ge__(self, other) -> bool:
        return self.image >= other

    def __eq__(self, other) -> bool:
        return self.image == other

    def __ne__(self, other) -> bool:
        return self.image != other

    def __array__(self):
        return self.image

    def __getitem__(self, key):
        return self.image[key]

    def __setitem__(self, key, value):
        self.image[key] = value

    def clip(
        self, min: int | None = None, max: int | None = None
    ) -> "ImageNBit":
        return ImageNBit(
            image=np.clip(self.image, min, max), bit_depth=self.bit_depth
        )

    # def min(self) -> int:
    #     return self.image.min()

    # def max(self) -> int:
    #     return self.image.max()

    def __getattr__(self, name: str):
        return getattr(self.image, name)
