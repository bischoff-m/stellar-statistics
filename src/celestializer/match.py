import celestializer as cl
import astrometry
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astropy import units as u


def query_simbad(
    img: cl.ImageNBit, match: astrometry.Match, max_magnitude: float = 8
) -> list[cl.SkyCoordMag]:
    # https://simbad.cds.unistra.fr/simbad/tap/help/adqlHelp.html
    # Calculate radius of image
    diag = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    radius_img = diag / 2 * match.scale_arcsec_per_pixel / 3600 * u.deg * 1.01

    # Send query to SIMBAD
    print(f"Querying SIMBAD for radius {radius_img}")
    simbad = Simbad()
    simbad.reset_votable_fields()
    simbad.add_votable_fields("main_id", "ra", "dec", "otype", "V")
    result = simbad.query_region(
        SkyCoord(
            ra=match.center_ra_deg,
            dec=match.center_dec_deg,
            unit=(u.deg, u.deg),
        ),
        radius=radius_img,
        criteria=f'"otype"=\'Star..\' and "V" < {max_magnitude}',
    )
    # Convert result to list of SkyCoordMag
    return [
        cl.SkyCoordMag(
            ra=float(row["ra"]),
            dec=float(row["dec"]),
            magnitude=float(row["V"]),
        )
        for row in result
    ]


def find_coordinates(
    stars: list[cl.StarCenter], camera: cl.CameraInfo
) -> astrometry.Match:
    """Get a pixel-to-sky transformation from a list of stars using the
    astrometry library.

    Parameters
    ----------
    stars : list[cl.StarCenter]
        List of stars to use for the matching.
    camera : cl.CameraInfo
        Camera information. Used to calculate arcsec per pixel.

    Returns
    -------
    Match
        Best match found by the astrometry library.

    Raises
    ------
    ValueError
        If no solution is found.
    """
    print("Finding coordinates")
    solver = astrometry.Solver(
        astrometry.series_4100.index_files(
            cache_directory=cl.Paths.data / "astrometry_cache",
        )
    )
    # TODO: Use camera info to set size_hint
    solution = solver.solve(
        stars=[(star.x, star.y) for star in stars],
        size_hint=astrometry.SizeHint(
            lower_arcsec_per_pixel=40,
            upper_arcsec_per_pixel=50,
        ),
        position_hint=None,
        solution_parameters=astrometry.SolutionParameters(),
    )

    if not solution.has_match():
        raise ValueError("No solution found")
    return solution.best_match()
