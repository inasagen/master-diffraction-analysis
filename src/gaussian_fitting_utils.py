import numpy as np 

# ==== Curve fit utils ====

def crop_image(image, roi_row):
    """
    Crops the image based on the region of interest (ROI) provided.

    Parameters:
        image (ndarray): The full 2D image array.
        roi_row (DataFrame row): A row from a DataFrame containing 'x_min', 'x_max', 'y_min', 'y_max' columns.

    Returns:
        tuple: (cropped_image, y_min, x_min)
            - cropped_image: The cropped portion of the image.
            - y_min, x_min: Coordinates of the top-left corner of the crop (used to adjust coordinates later).
    """
    if not roi_row.empty:
        y_min = int(roi_row['y_min'].values[0])
        y_max = int(roi_row['y_max'].values[0])
        x_min = int(roi_row['x_min'].values[0])
        x_max = int(roi_row['x_max'].values[0])
        cropped_image = image[y_min:y_max, x_min:x_max]
    else:
        print("ROI row empty: skipping cropping.")
        cropped_image, y_min, x_min = None, None, None
    return cropped_image, y_min, x_min


def normalize_intensity(image, min_val=0, max_val=None):
    """
    Normalizes the intensity of the image between 0 and 1.

    Parameters:
        image (ndarray): Input image array.
        min_val (float): Minimum intensity value to subtract.
        max_val (float): Maximum intensity value to scale with. If None, uses the image max.

    Returns:
        ndarray: Normalized image array.
    """
    if max_val is None:
        max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)


def gaussian2D(coords, A, mu_x, mu_y, sigma_x, sigma_y, rho):
    """
    Evaluates a single 2D elliptical Gaussian at given coordinates.

    Parameters:
        coords (ndarray): Nx2 array of (x, y) coordinates.
        A (float): Amplitude of the Gaussian.
        mu_x, mu_y (float): Mean (center) of the Gaussian.
        sigma_x, sigma_y (float): Standard deviations along x and y axes.
        rho (float): Correlation coefficient between x and y.

    Returns:
        ndarray: Gaussian values at each coordinate.
    """
    mu = np.array([mu_x, mu_y])
    cov = np.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])
    cov_inv = np.linalg.inv(cov)
    diff = coords - mu
    return A * np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))


def sum_of_gaussians(coords, *params):
    """
    Computes the sum of multiple 2D Gaussians with a constant offset.

    Parameters:
        coords (ndarray): Nx2 array of (x, y) coordinates.
        *params: Sequence of parameters (6 per Gaussian) + 1 final offset:
                 [A1, mu_x1, mu_y1, sigma_x1, sigma_y1, rho1, ..., offset]

    Returns:
        ndarray: Resulting summed Gaussian values plus offset.
    """
    num_gaussians = len(params[:-1]) // 6
    offset = params[-1]
    result = np.zeros(coords.shape[0])
    for i in range(num_gaussians):
        A, mu_x, mu_y, sigma_x, sigma_y, rho = params[i * 6:(i + 1) * 6]
        result += gaussian2D(coords, A, mu_x, mu_y, sigma_x, sigma_y, rho)
    return result + offset


def sum_of_gaussians_zerooffset(coords, *params):
    """
    Computes the sum of multiple 2D Gaussians without an offset.

    Parameters:
        coords (ndarray): Nx2 array of (x, y) coordinates.
        *params: Sequence of parameters (6 per Gaussian):
                 [A1, mu_x1, mu_y1, sigma_x1, sigma_y1, rho1, ...]

    Returns:
        ndarray: Resulting summed Gaussian values.
    """
    num_gaussians = len(params) // 6
    result = np.zeros(coords.shape[0])
    for i in range(num_gaussians):
        A, mu_x, mu_y, sigma_x, sigma_y, rho = params[i*6:(i+1)*6]
        result += gaussian2D(coords, A, mu_x, mu_y, sigma_x, sigma_y, rho)
    return result.ravel()


def diffscatter_gaussian(coords, A, mu_x, mu_y, sigma_x, sigma_y):
    """
    Computes a single 2D Gaussian (with zero correlation) and adds a constant offset.
    Used for modeling diffuse scattering backgrounds.

    Parameters:
        coords (ndarray): Nx2 array of (x, y) coordinates.
        A (float): Amplitude of the Gaussian.
        mu_x, mu_y (float): Mean (center) of the Gaussian.
        sigma_x, sigma_y (float): Standard deviations.
        offset (float): Constant background offset.

    Returns:
        ndarray: Gaussian + offset values at each coordinate.
    """
    return gaussian2D(coords, A, mu_x, mu_y, sigma_x, sigma_y, 0)



# ==== GMM utils ==== 

def sample_from_image(image, num_samples):
    """
    Samples (row, col) coordinates from the image using pixel intensity as weights.
    """
    flat_image = image.ravel()
    total = flat_image.sum()

    if total == 0:
        raise ValueError("Image intensity sum is zero, cannot sample.")
    
    probabilities = flat_image / total
    indices = np.random.choice(flat_image.size, size=num_samples, p=probabilities)
    coords = np.column_stack(np.unravel_index(indices, image.shape))

    # Add small jitter
    return coords + np.random.normal(0, 0.5, coords.shape)


def weight_to_amplitude(weight, sigma_x, sigma_y, rho, total_intensity):
    """
    Converts GMM weight to amplitude for a 2D Gaussian.
    """
    det_cov = sigma_x**2 * sigma_y**2 * (1 - rho**2)
    return (weight * total_intensity) / (2 * np.pi * np.sqrt(det_cov))