import numpy as np
from scipy import ndimage


def compute_elongation_from_image(data: np.ndarray) -> tuple[float, float]:
    """
    Calculate image elongation and trail-width continuity.

    Parameters
    ----------
    data : numpy.ndarray
        2D grayscale image containing pixel intensity values.

    Returns
    -------
    tuple[float, float]
        (elongation, continuity)
    - The elongation is a number greater or equal to 1. '1' means round, '2' means twice as long as wide, and so on.
    - The continuity is a number between 0 and 1. A value close to '0' means good continuity, a value close to '1' means bad continuity (variability of the width).

    Returns np.nan for both values in the following case:
    - input data is empty, has unexpected type, or has not exactly 2 dimensions.
    - the trail cannot be found near the center of the input image, or contains less than 5 pixels.
    - the trail width, length and orientation cannot be determined.
    - the trail is shorter than 8 pixels, or is not at least 3 pixels wide on 3 portions.
    """
    # No data: cannot compute elongation
    if data is None:
        return np.nan, np.nan

    # Convert the data into floating values, turning NaN into zero
    try:
        data = np.nan_to_num(data).astype(float)
    except (TypeError, ValueError):
        return np.nan, np.nan

    # Verify that the data has 2 dimensions as expected
    if data.ndim != 2:
        return np.nan, np.nan

    # --------------------------------------------------
    # 1. Compute the mask
    # --------------------------------------------------

    # Estimate noise using MAD
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    sigma = 1.4826 * mad if mad > 0 else np.std(data)

    # Threshold: keep only significant pixels
    threshold = median + 2 * sigma
    mask = data > threshold

    # Identify connected components (8-connectivity)
    structure = np.ones((3, 3), dtype=int)
    labels, num_objects = ndimage.label(mask, structure=structure)

    if num_objects == 0:
        return np.nan, np.nan

    # Geometrical center of the image
    cy = (data.shape[0] - 1) / 2
    cx = (data.shape[1] - 1) / 2

    # Select object containing the center pixel, if present
    iy, ix = int(round(cy)), int(round(cx))
    center_label = labels[iy, ix]

    if center_label != 0:
        mask = labels == center_label
    else:
        # Otherwise select the object with the nearest pixel
        y, x = np.where(labels > 0)

        distances = (y - cy) ** 2 + (x - cx) ** 2
        nearest_idx = np.argmin(distances)

        nearest_label = labels[y[nearest_idx], x[nearest_idx]]
        mask = labels == nearest_label

    # If too few pixels: skip
    if np.sum(mask) < 5:
        return np.nan, np.nan

    # --------------------------------------------------
    # 2. Look for the centroid
    # --------------------------------------------------

    # Get all points coordinates in the mask
    y, x = np.where(mask)

    # Get the corresponding pixel values (used as weights for the centroid)
    weights = data[mask]

    # Compute the total of the weights
    total = np.sum(weights)
    if total == 0:
        return np.nan, np.nan

    # Compute the centroid coordinates
    x_c = np.sum(x * weights) / total
    y_c = np.sum(y * weights) / total

    # --------------------------------------------------
    # 3. Compute the Elongation
    # --------------------------------------------------

    # Compute the distance to the centroid for each pixel in the mask (x and y components)
    dx = x - x_c
    dy = y - y_c

    # Compute the brightness-weighted horizontal spread of the object
    ixx = np.sum(weights * dx * dx) / total
    # Calculate the brightness-weighted vertical spread of the object
    iyy = np.sum(weights * dy * dy) / total

    # Compute the relationship between the horizontal and the vertical spread
    # This detects the tilt (diagonal) of the object
    ixy = np.sum(weights * dx * dy) / total

    # Build the covariance Matrix (brightness-weighted second-moment matrix)
    M = np.array([[ixx, ixy], [ixy, iyy]])

    # Compute eigen values and vectors for this matrix
    # i.e. compute the spread on the width direction, and the spread on the length direction of the trail
    try:
        eigvals, eigvecs = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    width_spread, length_spread = eigvals[-1], eigvals[0]

    if length_spread <= 0:
        return np.nan, np.nan

    # Compute the elongation from the 2 previous values
    # 1 means round, 2 means twice as long as wide, and so on
    elongation = float(np.sqrt(width_spread / length_spread))

    # --------------------------------------------------
    # 4. Compute Trail width continuity
    # --------------------------------------------------

    # Group x and y coordinates together
    coordinates = np.column_stack((x, y))
    # Move the reference frame to the centroid
    centred = coordinates - np.array([x_c, y_c])

    # Get the long and short axis eigen vectors already calculated in previous step
    long_axis_vector = eigvecs[:, -1]
    short_axis_vector = eigvecs[:, 0]

    # Multiply the matrix with the vectors to estimate the distance of each pixel to the axis
    longitudinal = centred @ long_axis_vector
    transverse = centred @ short_axis_vector

    # Call the peak-to-peak function to compute the total length of the trail
    trail_length = np.ptp(longitudinal)

    # Under 8 pixel, the trail is too small to compute continuity
    if trail_length < 8:
        return np.nan, np.nan

    # We will divide the tail into several bins
    # Bin count is one fifth of the trail length, minimum 3 and maximum 20
    n_bins = min(20, max(3, int(trail_length / 5)))

    # Compute the position of the edges of all bins
    bin_edges = np.linspace(longitudinal.min(), longitudinal.max(), n_bins + 1)

    # variable to store the width of the trail in each bin
    widths = []

    # Iterate on the bins
    for i in range(n_bins):
        # Select all pixels located in this bin
        in_bin = (longitudinal >= bin_edges[i]) & (longitudinal < bin_edges[i + 1])

        # For the last bin, select all pixels in the bin, including those on the edge
        if i == n_bins - 1:
            in_bin = (longitudinal >= bin_edges[i]) & (longitudinal <= bin_edges[i + 1])

        # Take the transversal position of the pixels in the pin
        pixels_across = transverse[in_bin]

        # Ignore bin if we have less than 3 pixels
        if len(pixels_across) < 3:
            continue

        # Compute the width with 90% of the distribution, in order to ignore isolated pixels in extreme positions
        width = np.percentile(pixels_across, 95) - np.percentile(pixels_across, 5)

        widths.append(width)

    # Under 3 widths, the continuity calculation is not reliable
    if len(widths) < 3:
        return np.nan, np.nan

    # Convert into a numpy array
    widths = np.asarray(widths)

    # Compute the coefficient of variation
    width_variation = float(np.std(widths) / np.mean(widths))

    return elongation, width_variation
