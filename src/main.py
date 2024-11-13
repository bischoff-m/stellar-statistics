from rawkit.raw import Raw
from definitions import ROOT_DIR

if __name__ == "__main__":
    with Raw(ROOT_DIR / "data/observations/IMG_3316.CR2") as raw:
        print(raw.metadata)
