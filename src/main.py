from rawkit.raw import Raw
from definitions import ROOT_DIR

if __name__ == "__main__":
    observe_dir = ROOT_DIR / "data/observations"
    for file in observe_dir.iterdir():
        with Raw(file.as_posix()) as raw:
            print(raw.metadata)
