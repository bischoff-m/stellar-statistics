from pathlib import Path


class Paths:
    root = Path(__file__).parent.parent
    observations = root / "data/observations"
    calibration = root / "data/calibration"
