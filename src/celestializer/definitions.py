from pathlib import Path
from pydantic import BaseModel, ConfigDict
import cv2
from numpy.typing import NDArray
import numpy as np


class Paths:
    root = Path(__file__).parent.parent.parent
    data = root / "data"
    observations = data / "observations"
    calibration = data / "calibration"
    saved = data / "saved"


class StarCenter(BaseModel):
    x: int
    y: int


class StarMag(StarCenter):
    magnitude: float


class StarPixels(StarCenter):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: cv2.typing.Rect
    mask: NDArray[np.bool]
