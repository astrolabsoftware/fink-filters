import numpy as np

def compute_elongation_from_image(data):
    """
    Robust elongation using thresholded significant pixels.
    """
    try:
        data = np.nan_to_num(data).astype(float)

        # Estimate noise using MAD
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        sigma = 1.4826 * mad if mad > 0 else np.std(data)

        # Threshold: keep only significant pixels
        threshold = median + 3 * sigma

        mask = data > threshold

        # If too few pixels → skip
        if np.sum(mask) < 5:
            return np.nan

        # Use ONLY significant pixels
        y, x = np.where(mask)
        weights = data[mask]

        # Centroid
        total = np.sum(weights)
        x_c = np.sum(x * weights) / total
        y_c = np.sum(y * weights) / total

        dx = x - x_c
        dy = y - y_c

        ixx = np.sum(weights * dx * dx) / total
        iyy = np.sum(weights * dy * dy) / total
        ixy = np.sum(weights * dx * dy) / total

        M = np.array([[ixx, ixy],
                      [ixy, iyy]])

        eigvals = np.linalg.eigvals(M)

        l1, l2 = np.max(eigvals), np.min(eigvals)

        if l2 <= 0:
            return np.nan

        elongation = np.sqrt(l1 / l2)

        return float(elongation)

    except Exception as e:
        return np.nan
