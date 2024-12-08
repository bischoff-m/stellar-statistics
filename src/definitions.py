from pathlib import Path


class Paths:
    root = Path(__file__).parent.parent
    data = root / "data"
    observations = data / "observations"
    calibration = data / "calibration"
