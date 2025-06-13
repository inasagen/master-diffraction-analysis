import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit 
from joblib import Parallel, delayed
from tqdm import tqdm
import fabio

# Custom utilities for cropping, normalization, and Gaussian models
from gaussian_fitting_utils import (
    crop_image,
    normalize_intensity,
    sum_of_gaussians_zerooffset,
    diffscatter_gaussian
)

# === Configuration Parameters ===

# Define root and data/output paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIFF_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'diffraction_images')
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results', 'curve_fit_results')
ROI_CSV_PATH = os.path.join(PROJECT_ROOT, 'results', 'ROI', 'ROI_800thresh.csv')
GAUSSIAN_PARAMS_CSV = os.path.join(RESULTS_FOLDER, 'curve_fit_init.csv')
DIFFUSE_SCATTERING_CSV = os.path.join(PROJECT_ROOT, 'results', 'diffuse_scattering', 'diffuse_scattering_residual.csv')
OUTPUT_FILENAME = 'curve_fit_refined_residual_diffscatter'

# Threshold and parallelization settings
BACKGROUND_THRESHOLD = 70
NUM_PARALLEL_JOBS = 6


# === Image Processing and Fitting Functions ===

def refined_gaussian_fit(image, diffscatter_params_df, params_df):
    """
    Perform Gaussian fitting on an image after subtracting the diffuse scattering background.

    Args:
        image (np.ndarray): 2D input image.
        diffscatter_params_df (pd.DataFrame): Background Gaussian parameters.
        params_df (pd.DataFrame): Initial Gaussian peak parameters.

    Returns:
        pd.DataFrame: Refined fit parameters including amplitude, means, sigmas, correlation, and offset.
    """
    if params_df.isnull().values.any():
        return pd.DataFrame([[np.nan] * 7], columns=['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho', 'offset'])

    # Prepare meshgrid coordinates for fitting
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    coords = np.stack([x.ravel(), y.ravel()], axis=1)

    # Compute and subtract diffuse scattering background
    diff_params = diffscatter_params_df[['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y']].values[0]
    diffuse_scattering = diffscatter_gaussian(coords, *diff_params).reshape(image.shape)
    corrected = image - diffuse_scattering - BACKGROUND_THRESHOLD
    corrected[corrected < 0] = 0

    # Normalize intensity
    normalized = normalize_intensity(corrected)

    # Normalize input amplitudes for stable fitting
    params_df = params_df.copy()
    params_df['amplitude'] /= np.max(params_df['amplitude'])

    # Build initial guesses and bounds
    initial_guess, lower_bounds, upper_bounds = [], [], []
    for _, row in params_df.iterrows():
        popt = row[['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho']].values
        initial_guess.extend(popt)
        lower_bounds.extend([0, row['mean_x'] - 20, row['mean_y'] - 20, 0, 0, -0.99])
        upper_bounds.extend([1, row['mean_x'] + 20, row['mean_y'] + 20, 10, 10, 0.99])
    bounds = (lower_bounds, upper_bounds)

    # Perform curve fitting
    try:
        fitted, _ = curve_fit(
            sum_of_gaussians_zerooffset,
            coords,
            normalized.ravel(),
            p0=initial_guess,
            maxfev=10000,
            bounds=bounds
        )
        result_df = pd.DataFrame(fitted.reshape(-1, 6), columns=['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho'])
        result_df['amplitude'] *= np.max(corrected)
        return result_df

    except Exception as e:
        print(f"Fit failed: {e}")
        return pd.DataFrame([[np.nan] * 6], columns=['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho'])


def process_image_file(file_path, diffscatter_df, all_params_df, roi_df):
    """
    Crop an image, subtract background, and refine Gaussian fitting.

    Args:
        file_path (str): Path to the image file.
        diffscatter_df (pd.DataFrame): Background parameters.
        all_params_df (pd.DataFrame): Initial fit parameters.
        roi_df (pd.DataFrame): ROI definitions.

    Returns:
        pd.DataFrame: Fitting results with image filename.
    """
    filename = os.path.basename(file_path)
    image = fabio.open(file_path).data

    roi_row = roi_df[roi_df['filename'] == filename]
    cropped_image, y_offset, x_offset = crop_image(image, roi_row)

    if cropped_image is None:
        return pd.DataFrame([[np.nan] * 6 + [filename]], columns=[
            'amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho', 'filename'])

    # Adjust peak and background parameters for cropped coordinates
    params = all_params_df[all_params_df['filename'] == filename].copy()
    params['mean_x'] -= x_offset
    params['mean_y'] -= y_offset

    diff_params = diffscatter_df[diffscatter_df['filename'] == filename].copy()
    diff_params['mean_x'] -= x_offset
    diff_params['mean_y'] -= y_offset

    # Perform refined fit
    fit_result = refined_gaussian_fit(cropped_image, diff_params, params)
    fit_result['filename'] = filename
    fit_result['mean_x'] += x_offset
    fit_result['mean_y'] += y_offset

    return fit_result


# === Main Script ===

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Load file paths and parameter data
    file_paths = sorted([
        os.path.join(DIFF_IMAGE_FOLDER, f)
        for f in os.listdir(DIFF_IMAGE_FOLDER) if f.endswith('.cbf')
    ])
    roi_df = pd.read_csv(ROI_CSV_PATH)
    all_params_df = pd.read_csv(GAUSSIAN_PARAMS_CSV)
    diffscatter_df = pd.read_csv(DIFFUSE_SCATTERING_CSV)

    print(f"Using {NUM_PARALLEL_JOBS} parallel jobs for processing.")

    # Process all images in parallel
    start_time = time.time()
    results = Parallel(n_jobs=NUM_PARALLEL_JOBS)(
        delayed(process_image_file)(file_path, diffscatter_df, all_params_df, roi_df)
        for file_path in tqdm(file_paths, desc="Processing images")
    )
    total_time = time.time() - start_time
    print(f"Processing completed in {total_time:.2f} seconds.")

    # Save log and output
    log_path = os.path.join(RESULTS_FOLDER, f"{OUTPUT_FILENAME}_log.txt")
    with open(log_path, 'w') as f:
        f.write(f"Total processing time: {total_time:.2f} seconds\n")

    combined_df = pd.concat(results, ignore_index=True)
    combined_df.to_csv(os.path.join(RESULTS_FOLDER, f"{OUTPUT_FILENAME}.csv"), index=False)