from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image
import xxhash
from numpy.typing import NDArray


class ImageNBit(np.ndarray):
    bit_depth: int
    pil_cache: tuple[int, Image.Image] | None = None

    def __new__(
        cls,
        image: NDArray,
        bit_depth: int,
        pil_cache: tuple[int, Image.Image] | None = None,
    ):
        """Create a new ImageNBit object.

        Parameters
        ----------
        image : NDArray
            Image data. If the image has 3 dimensions, it must have 3 channels.
        bit_depth : int
            Bit depth of the image. If 1, the image is normalized to 0-1 and has
            type np.float32. If <= 8 or <= 16, the image is scaled to the target
            bit depth and has type np.uint8 or np.uint16, respectively.
        pil_cache : tuple[int, Image.Image] | None, optional
            PIL cache, by default None

        Returns
        -------
        ImageNBit
            ImageNBit object.
        """
        if image.ndim not in (2, 3):
            raise ValueError("Image must have 2 or 3 dimensions.")
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError(
                "If image has 3 dimensions, it must have 3 channels."
            )
        if type(bit_depth) is not int:
            raise ValueError("Bit depth must be an integer.")
        if bit_depth < 1:
            raise ValueError("Bit depth must be at least 1.")
        if bit_depth > 16:
            raise ValueError("Bit depth must be at most 16.")
        if bit_depth != 1 and np.isnan(image).any():
            raise ValueError("If bit depth is not 1, image cannot have NaN.")
        if image.min() < 0 or image.max() > 2**bit_depth - 1:
            raise ValueError(
                "Image values must be between 0 and 2^bit_depth - 1."
            )
        dtype = (
            np.float32
            if bit_depth == 1
            else np.uint8
            if bit_depth <= 8
            else np.uint16
        )
        obj = np.asarray(image, dtype=dtype).view(cls)
        obj.bit_depth = bit_depth
        obj.pil_cache = pil_cache
        return obj

    def __array_finalize__(self, obj):
        """Finalize the creation of the object. Needed for numpy ndarray
        subclasses."""
        if obj is None:
            return
        self.bit_depth = getattr(obj, "bit_depth", None)
        self.pil_cache = getattr(obj, "pil_cache", None)

    def to_bitdepth(self, bit_depth: int) -> "ImageNBit":
        """Lossy conversion of image bit depth.

        Parameters
        ----------
        bit_depth : int
            Target bit depth. If 1, the image is normalized to 0-1 and has type
            np.float32. If <= 8 or <= 16, the image is scaled to the target bit
            depth and has type np.uint8 or np.uint16, respectively.

        Returns
        -------
        ImageNBit
            Image with the target bit depth.
        """
        if bit_depth < 1:
            raise ValueError("Bit depth must be at least 1.")
        if bit_depth > 16:
            raise ValueError("Bit depth must be at most 16.")
        if bit_depth == self.bit_depth:
            return self
        if bit_depth == 1:
            new_img = self / (2**self.bit_depth - 1)
        else:
            new_img = self.copy()
            new_img[np.isnan(new_img)] = 0
            new_img = new_img / (2**self.bit_depth - 1) * (2**bit_depth - 1)
            new_img = np.round(new_img)
            new_type = np.uint8 if bit_depth <= 8 else np.uint16
            new_img = np.asarray(new_img, dtype=new_type)
        # Keep the pil_cache because the image data is the same
        new_cache = (
            None
            if self.pil_cache is None
            else (self._get_hash(), self.pil_cache[1])
        )
        return ImageNBit(
            image=new_img,
            bit_depth=bit_depth,
            pil_cache=new_cache,
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
        data_hash = self._get_hash()
        if self.pil_cache is not None and self.pil_cache[0] == data_hash:
            return self.pil_cache[1]

        # PIL expects 8-bit images
        arr = self.to_bitdepth(8)
        # Replace NaN with 0
        arr = np.nan_to_num(arr, nan=0).view(np.uint8)
        # To PIL and update cache
        pil_img = Image.fromarray(arr)
        if arr.ndim == 2:
            pil_img = pil_img.convert("L")
        elif arr.ndim == 3:
            pil_img = pil_img.convert("RGB")
        self.pil_cache = (data_hash, pil_img)
        return self.pil_cache[1]

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
        if value < 0 or value > 1:
            raise ValueError("Value must be between 0 and 1.")
        if self.bit_depth == 1:
            return value
        return np.round(value * (2**self.bit_depth - 1)).astype(self.dtype)

    def channel(
        self, channel: Literal["red"] | Literal["green"] | Literal["blue"]
    ) -> "ImageNBit":
        """Get a single channel of the image.

        Converts the image to bitdepth=1 and sets all pixels except the
        specified channel to np.nan in a Bayer sensor image with a filter
        pattern of RGGB.

        Parameters
        ----------
        channel : Literal["red"] | Literal["green"] | Literal["blue"]
            Channel to keep.

        Returns
        -------
        ImageNBit
            Image with only the specified channel.
        """
        if self.ndim != 2:
            raise ValueError("Image must have 2 dimensions.")
        img = self.to_bitdepth(1)
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
        return img

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

    def green_interpolated(self, keep_bitdepth: bool = False) -> "ImageNBit":
        """Interpolate green pixels in the raw image data.

        Sets the value of green pixels to the mean of the 8 neighboring pixels.
        If the image has an odd number of rows or columns, the last row or
        column is removed.

        Parameters
        ----------
        keep_bitdepth : bool, optional
            If False, the image is normalized to 0-1. Otherwise, the original
            bit depth is restored, by default False

        Returns
        -------
        ImageNBit
            Image with interpolated green pixels.
        """
        green_nan = self.channel("green")
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

        new_img: ImageNBit = green_nan.copy()
        new_img[::2, ::2] = new_odd[::2, :]
        new_img[1::2, 1::2] = new_even[1::2, :]

        new_img = np.clip(new_img, 0, 1)
        if keep_bitdepth:
            new_img = new_img.to_bitdepth(self.bit_depth)
        return new_img

    def cutoff(self, threshold: float) -> "ImageNBit":
        """Set all values below a threshold to 0.

        Parameters
        ----------
        threshold : float
            Threshold value.

        Returns
        -------
        ImageNBit
            Image with values below the threshold set to 0.
        """
        img = self.copy()
        img[img < threshold] = 0
        return img

    def upscale(self, factor: int) -> "ImageNBit":
        """Upscale the image by a factor.

        Parameters
        ----------
        factor : int
            Upscale factor.

        Returns
        -------
        ImageNBit
            Upscaled image.
        """
        if factor < 2:
            raise ValueError("Upscale factor must be at least 2.")
        if self.ndim == 2:
            return np.kron(self, np.ones((factor, factor)))
        else:
            return np.kron(self, np.ones((factor, factor, 1)))

    def _get_hash(self) -> int:
        """Get a hash of the image data.

        Returns
        -------
        int
            Hash of the image data.
        """
        return xxhash.xxh64_intdigest(self.data.tobytes())
