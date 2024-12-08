from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class ImageNBit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray = Field(..., allow_mutation=False)
    bit_depth: int = Field(..., allow_mutation=False)
    pil_cache: Image.Image | None = None

    def to_bitdepth(self, bit_depth: int) -> "ImageNBit":
        """Lossy conversion of image bit depth.

        Parameters
        ----------
        bit_depth : int
            Target bit depth. If 0, the image is normalized to 0-1 and has type
            np.float64. If 8 or 16, the image is scaled to the target bit depth
            and has type np.uint8 or np.uint16, respectively.

        Returns
        -------
        ImageNBit
            Image with the target bit depth.
        """
        if bit_depth == self.bit_depth:
            return self

        if bit_depth == 0:
            new_img = self.image / (2**self.bit_depth - 1)
            return ImageNBit(
                image=new_img, bit_depth=bit_depth, pil_cache=self.pil_cache
            )

        new_img = self.image.copy()
        new_img[np.isnan(new_img)] = 0
        new_img = new_img / (2**self.bit_depth - 1) * (2**bit_depth - 1)
        new_img = np.round(new_img)
        if bit_depth <= 8:
            new_img = np.asarray(new_img, dtype=np.uint8)
        else:
            new_img = np.asarray(new_img, dtype=np.uint16)

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
