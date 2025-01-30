import cv2
from tqdm import tqdm
import celestializer as cl
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


def classify_star_pixels(
    img: cl.ImageNBit, tolerance: int = 5
) -> NDArray[np.bool]:
    """Apply flood fill to find pixels belonging to stars in an image.

    Parameters
    ----------
    img : cl.ImageNBit
        Image to classify.
    tolerance : int, optional
        Tolerance for flood fill, by default 5

    Returns
    -------
    NDArray[np.bool]
        Mask of star pixels. True for star pixels, False for background.
    """
    assert img.ndim == 2, "Image must be 2D"
    # Must be CV_8U for flood fill
    assert img.bit_depth == 8, "Image must be 8-bit"
    assert not np.isnan(img).any(), "Image must not contain NaN values"

    # Find darkest pixel
    min_loc = np.unravel_index(np.argmin(img), img.shape)
    # Flood fill from darkest pixel
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    # TODO: Find upDiff dynamically by counting number of single pixels in mask
    cv2.floodFill(
        img,
        mask,
        [min_loc[1], min_loc[0]],
        255,
        loDiff=100,
        upDiff=tolerance,
        flags=cv2.FLOODFILL_MASK_ONLY,
    )
    # Remove border and invert mask
    mask = 1 - mask[1:-1, 1:-1]

    # Remove single pixels from mask
    kernel = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(np.bool)


def stars_from_mask(mask: NDArray[np.bool]) -> list[cl.StarMasked]:
    """Find stars in a mask of star pixels.

    Parameters
    ----------
    mask : NDArray[np.bool]
        Mask of star pixels. True for star pixels, False for background.

    Returns
    -------
    list[cl.StarPixels]
        List of stars found in the mask.
    """
    mask = mask.astype(np.uint8)
    # Find connected components
    n_stars, labels = cv2.connectedComponents(mask, connectivity=4)
    # Find star centers using moments
    stars = []
    for i in tqdm(range(1, n_stars)):
        mask_star = labels == i
        mask_star_int = mask_star.astype(np.uint8)
        # Extract bounding box part of the mask
        bbox = cv2.boundingRect(mask_star_int)
        star_only = mask_star_int[
            bbox[1] : bbox[1] + bbox[3],
            bbox[0] : bbox[0] + bbox[2],
        ].copy()
        # Use moments to find center
        moments = cv2.moments(star_only)
        if moments["m00"] == 0:
            continue
        x = int(moments["m10"] / moments["m00"]) + bbox[0]
        y = int(moments["m01"] / moments["m00"]) + bbox[1]
        stars.append(
            cl.StarMasked(
                x=x,
                y=y,
                bbox=bbox,
                mask=star_only.astype(np.bool),
            )
        )
    return stars


def stars_by_template(
    img: cl.ImageNBit, threshold: float = 0.4
) -> list[cl.StarCenter]:
    """Find stars in an image by template matching.

    Parameters
    ----------
    img : cl.ImageNBit
        Image to find stars in.
    threshold : float, optional
        Threshold for template matching, by default 0.4

    Returns
    -------
    list[cl.StarCenter]
        List of stars found in the image.
    """
    assert not np.isnan(img).any(), "Image must not contain NaN values"
    # 2D Gaussian kernel as template
    kernel_size = 21
    template = cv2.getGaussianKernel(kernel_size, 5)
    template = template * template.T
    template /= template.max()
    template *= 255
    template = cl.ImageNBit(image=template, bit_depth=8).cutoff(0.5)
    template = template.astype(np.float32)

    # Match template
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)

    def remove_duplicates(arr: np.ndarray, tolerance: int) -> np.ndarray:
        sort_indices = np.lexsort((arr[:, 0], arr[:, 1]))
        arr = arr[sort_indices]
        diff = np.diff(arr, axis=0)
        dropped = np.sqrt((diff**2).sum(axis=1)) < tolerance / 2
        return arr[~np.concatenate(([False], dropped))]

    rect_size = 40
    # Array of (y, x) coordinates, will be flipped later
    matches = np.asarray([loc[0], loc[1]]).T
    matches = remove_duplicates(matches, rect_size)
    matches = np.flip(matches, axis=1)
    matches = remove_duplicates(matches, rect_size)
    matches += kernel_size // 2
    return [cl.StarCenter(x=x, y=y) for x, y in matches]


def refine_mask(img: cl.ImageNBit, star: cl.StarMasked) -> cl.StarMasked:
    # Must be 2D for padding
    assert img.ndim == 2, "Image must be 2D"
    # (top, bottom), (left, right)
    grow = ((2, 2), (2, 2))
    # Reduce grow if at the edge
    grow = (
        (
            min(grow[0][0], star.bbox[1]),
            min(grow[0][1], img.shape[0] - star.bbox[1] - star.bbox[3]),
        ),
        (
            min(grow[1][0], star.bbox[0]),
            min(grow[1][1], img.shape[1] - star.bbox[0] - star.bbox[2]),
        ),
    )

    # Dilate the mask
    kernel = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    mask_padded = np.pad(star.mask.astype(np.uint8), grow, mode="constant")
    mask_dilated = cv2.dilate(mask_padded, kernel, iterations=2)
    mask_dilated = mask_dilated.astype(bool)

    # Apply the mask
    img_star = img[
        star.bbox[1] - grow[0][0] : star.bbox[1] + star.bbox[3] + grow[0][1],
        star.bbox[0] - grow[1][0] : star.bbox[0] + star.bbox[2] + grow[1][1],
    ].copy()

    # Get new mask by clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(
        img_star[mask_dilated].reshape(-1, 1)
    )
    pred = kmeans.predict(img_star.reshape(-1, 1)).reshape(img_star.shape)
    right_cluster = np.argmax(kmeans.cluster_centers_)
    new_mask = pred == right_cluster

    # Find new bounding box
    new_bbox = cv2.boundingRect(new_mask.astype(np.uint8))
    # Cut the mask
    new_mask = new_mask[
        new_bbox[1] : new_bbox[1] + new_bbox[3],
        new_bbox[0] : new_bbox[0] + new_bbox[2],
    ]
    new_bbox = (
        star.bbox[0] - grow[1][0] + new_bbox[0],
        star.bbox[1] - grow[0][0] + new_bbox[1],
        new_bbox[2],
        new_bbox[3],
    )
    new_star = cl.StarMasked(x=star.x, y=star.y, mask=new_mask, bbox=new_bbox)
    return new_star
