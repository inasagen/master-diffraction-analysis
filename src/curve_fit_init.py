import os
import fabio
import time
import numpy as np
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

# Custom utilities for image processing and fitting
from gaussian_fitting_utils import normalize_intensity, crop_image, sum_of_gaussians


# === Configuration Parameters ===

# Project root directory (one level up from this script)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths to data and output
DIFF_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'diffraction_images')
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results', 'curve_fit_results')
ROI_CSV_PATH = os.path.join(PROJECT_ROOT, 'results', 'ROI', 'ROI_800thresh.csv')
OUTPUT_FILENAME = 'curve_fit_init.csv'

# Peak detection and parallelization settings
PEAK_THRESHOLD = 9000
NUM_JOBS = 6

def fit_gaussians_to_image(image):
    """
    Detects local maxima and fits multiple 2D Gaussians plus an offset to the image.

    Args:
        image (np.ndarray): Input 2D image array.

    Returns:
        pd.DataFrame: Fitted parameters for each Gaussian and global offset.
    """
    peaks = peak_local_max(image, threshold_abs=PEAK_THRESHOLD)
    normalized_image = normalize_intensity(image)

    initial_guess, lower_bound, upper_bound = [], [], []
    
    # Build initial guesses and bounds for each detected peak
    for peak in peaks:
        initial_guess.extend([normalized_image[peak[0], peak[1]], peak[1], peak[0], 2.5, 2.5, 0])
        lower_bound.extend([0, peak[1] - 20, peak[0] - 20, 0, 0, -0.99])
        upper_bound.extend([1, peak[1] + 20, peak[0] + 20, 10, 10, 0.99])

    # Add global offset
    initial_guess.append(np.median(normalized_image))
    lower_bound.append(0)
    upper_bound.append(1)
    bounds = (lower_bound, upper_bound)

    # Generate coordinates for curve fitting
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    coords = np.stack([x.ravel(), y.ravel()], axis=1)

    try:
        # Fit the sum of Gaussians
        fitted_params, _ = curve_fit(
            sum_of_gaussians, coords, normalized_image.ravel(),
            p0=initial_guess, bounds=bounds, maxfev=10000
        )

        # Build DataFrame of parameters (rescale amplitudes and offset to original intensity range)
        params_df = pd.DataFrame(fitted_params[:-1].reshape(-1, 6),
                                 columns=['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho'])
        params_df['offset'] = fitted_params[-1] * np.max(image)
        params_df['amplitude'] *= np.max(image)

    except Exception as e:
        print(f"Error fitting {len(peaks)} peaks: {e}")
        params_df = pd.DataFrame([[np.nan] * 7], columns=[
            'amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho', 'offset'])

    return params_df


def process_image_file(file_path, roi_df):
    """
    Loads and processes a single image file.

    Args:
        file_path (str): Path to the image file.
        roi_df (pd.DataFrame): DataFrame containing ROI coordinates.

    Returns:
        pd.DataFrame: Fitting results for the image.
    """
    image = fabio.open(file_path).data
    filename = os.path.basename(file_path)
    roi_row = roi_df[roi_df['filename'] == filename]

    # Crop image to ROI
    cropped_image, y_offset, x_offset = crop_image(image, roi_row)

    # Handle missing ROI
    if cropped_image is None:
        return pd.DataFrame([[np.nan]*7 + [filename]], columns=[
            'amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho', 'offset', 'filename'])

    # Fit Gaussians
    fitted_params = fit_gaussians_to_image(cropped_image)
    fitted_params['filename'] = filename

    # Adjust coordinates to original image space
    if y_offset is not None and x_offset is not None:
        fitted_params['mean_x'] += x_offset
        fitted_params['mean_y'] += y_offset

    return fitted_params


# === Main Script ===

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Load files and ROI info
    files = sorted([f for f in os.listdir(DIFF_IMAGE_FOLDER) if f.endswith('.cbf')])
    file_paths = [os.path.join(DIFF_IMAGE_FOLDER, file) for file in files]
    roi_df = pd.read_csv(ROI_CSV_PATH)

    print(f"Using {NUM_JOBS} parallel jobs for processing.")

    # Process all images in parallel
    start_time = time.time()
    results = Parallel(n_jobs=NUM_JOBS)(
        delayed(process_image_file)(file_path, roi_df)
        for file_path in tqdm(file_paths, desc="Processing images")
    )
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds.")

    # Write log
    log_file_path = os.path.join(RESULTS_FOLDER, f'{OUTPUT_FILENAME}_log.txt')
    with open(log_file_path, 'w') as f:
        f.write(f"Total time taken for processing images: {processing_time:.2f} seconds\n")

    # Save results
    combined_results = pd.concat(results, ignore_index=True)
    combined_results.to_csv(os.path.join(RESULTS_FOLDER, f"{OUTPUT_FILENAME}.csv"), index=False)