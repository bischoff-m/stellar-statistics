# stellar-statistics

Processing and analysis tool for astronomical imaging.

This tool was developed as part of my project on Stellar Statistics for the
laboratory course Astronomy and Astrophysics in the winter semester 2024/25 at
the RWTH Aachen University.

Supervisors: Mia Do, M.Sc.; Prof. Dr. Chr. Wiebusch; Prof. Dr. O. Pooth

## Installation

To load raw images, the tool requires the
[rawkit](https://github.com/photoshell/rawkit) library, which itself requires
[libraw](https://www.libraw.org/) (version 16, 17 or 18). I could not get the
installation to work on Windows, so I will only provide instructions for
Linux.

Install all OpenCV dependencies with:

```bash
sudo apt-get install python3-opencv
```

### Libraw

Follow the instructions on the [libraw
website](https://www.libraw.org/docs/Install-LibRaw-eng.html) or run these
commands:

```bash
wget https://www.libraw.org/data/LibRaw-0.18.13.tar.gz
tar xzvf LibRaw-0.18.13.tar.gz
cd LibRaw-0.18.13/
./configure
make
sudo make install
cd ..
rm -r LibRaw-0.18.13/
rm LibRaw-0.18.13.tar.gz
```

If `rawkit` cannot find the right version of `libraw`, make sure to uninstall
other versions, e.g. with `sudo apt remove libraw20`.

### Python dependencies

I use [uv](https://github.com/astral-sh/uv) as a package manager. To install
the dependencies, run:

```bash
uv venv
```

Otherwise, you can install the dependencies listed in `pyproject.toml` manually.
