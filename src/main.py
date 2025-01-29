"""
Things to try:
- Finding stars
    - Start with brightest pixel, remove star and repeat
    - SIFT with Gaussian kernel
    - matchTemplate with Gaussian kernel of varying sizes
    - DAOPHOT
        https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract
- Reduce noise
    - Erosion, Opening
        https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    - Contour detection
        https://stackoverflow.com/questions/60603243/detect-small-dots-in-image
        https://www.geeksforgeeks.org/white-and-black-dot-detection-using-opencv-python/
    - floodFill
- Identify pixels that belong to stars
    - Fit Gaussian with full covariance matrix
- Vignette correction
    - Subtract blurred image
"""

import astrometry
import logging
import celestializer as cl

logging.getLogger().setLevel(logging.INFO)

solver = astrometry.Solver(
    astrometry.series_5200.index_files(
        cache_directory=cl.Paths.data / "astrometry_cache",
    )
)

solution = solver.solve(
    stars=[(0, 0), (0, 1), (0, 2), (0, 3)],
    size_hint=astrometry.SizeHint(
        lower_arcsec_per_pixel=40,
        upper_arcsec_per_pixel=50,
    ),
    position_hint=None,
    solution_parameters=astrometry.SolutionParameters(),
)

if solution.has_match():
    print("Solution found")
    best_match = solution.best_match()
    print(f"{best_match.center_ra_deg=}")
    print(f"{best_match.center_dec_deg=}")
    print(f"{best_match.scale_arcsec_per_pixel=}")
else:
    print("No solution found")
