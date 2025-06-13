
import os
import fabio
import time
import numpy as np
from scipy.optimize import curve_fit 
from joblib import Parallel, delayed
import pandas as pd
from skimage.feature import peak_local_max
from tqdm import tqdm

# Custom utilities for image processing and fitting
from gaussian_fitting_utils import normalize_intensity, sum_of_gaussians_zerooffset, diffscatter_gaussian


# === Configuration ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DIFF_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'diffraction_images')
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results', 'curve_fit_results')
ROI_CSV_PATH = os.path.join(PROJECT_ROOT, 'results', 'ROI', 'ROI_800thresh.csv')
DIFFSCATTER_CSV_PATH = os.path.join(PROJECT_ROOT, 'results', 'diffuse_scattering', 'diffuse_scattering_1900thresh.csv')
OUTPUT_FILENAME = 'curve_fit_1900thresh_diffscatter'

# Threshold and parallelization settings
NUM_JOBS = 6
BACKGROUND_THRESHOLD = 70
PEAK_THRESHOLD = 9000


# === Functions ===

def gaussian_fit(image):
    """Fit multiple 2D Gaussians to an image based on peak locations."""
    y_coords, x_coords = np.indices(image.shape)
    coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

    peaks = peak_local_max(image, threshold_abs=PEAK_THRESHOLD)
    if len(peaks) == 0:
        return pd.DataFrame([[np.nan] * 6], columns=['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho'])

    norm_image = normalize_intensity(image)
    initial_guess, lower_bounds, upper_bounds = [], [], []

    for y, x in peaks:
        amp = norm_image[y, x]
        initial_guess += [amp, x, y, 2.5, 2.5, 0]
        lower_bounds += [0, x - 20, y - 20, 0, 0, -0.99]
        upper_bounds += [1, x + 20, y + 20, 10, 10, 0.99]

    try:
        fitted, _ = curve_fit(sum_of_gaussians_zerooffset, coords, norm_image.ravel(),
                              p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=10000)
        params_df = pd.DataFrame(fitted.reshape(-1, 6),
                                 columns=['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho'])
        params_df['amplitude'] *= np.max(image)
        return params_df
    except Exception as e:
        print(f"Fit failed: {e}")
        return pd.DataFrame([[np.nan] * 6], columns=['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho'])


def preprocess_image(image, diff_row, roi_row):
    """Subtract diffuse background and crop to ROI."""
    if diff_row.empty or roi_row.empty:
        return None

    y_min, y_max = int(roi_row['y_min'].values[0]), int(roi_row['y_max'].values[0])
    x_min, x_max = int(roi_row['x_min'].values[0]), int(roi_row['x_max'].values[0])

    y, x = np.indices(image.shape)
    params = diff_row.iloc[0][['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'offset']].values
    background = diffscatter_gaussian((x, y), *params[:-1]).reshape(image.shape)

    corrected = np.clip(image - background - BACKGROUND_THRESHOLD, 0, None)
    return corrected[y_min:y_max, x_min:x_max], x_min, y_min


def process_image_file(file_path, diff_df, roi_df):
    """Load, preprocess, fit, and record results for one image."""
    filename = os.path.basename(file_path)
    image = fabio.open(file_path).data

    roi_row = roi_df[roi_df['filename'] == filename]
    diff_row = diff_df[diff_df['filename'] == filename]

    result_df = pd.DataFrame([[np.nan] * 7], columns=['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho', 'filename'])

    preprocessed = preprocess_image(image, diff_row, roi_row)
    if preprocessed is None:
        print(f"Skipping {filename} due to missing ROI or diffuse data.")
        return result_df

    cropped_image, x_offset, y_offset = preprocessed
    fitted = gaussian_fit(cropped_image)
    fitted['filename'] = filename
    fitted['mean_x'] += x_offset
    fitted['mean_y'] += y_offset

    return fitted


# === Main ===

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    file_paths = sorted([os.path.join(DIFF_IMAGE_FOLDER, f) for f in os.listdir(DIFF_IMAGE_FOLDER) if f.endswith('.cbf')])
    roi_df = pd.read_csv(ROI_CSV_PATH)
    diff_df = pd.read_csv(DIFFSCATTER_CSV_PATH).interpolate()

    print(f"Using {NUM_JOBS} parallel jobs for processing {len(file_paths)} images.")

    start_time = time.time()
    results = Parallel(n_jobs=NUM_JOBS)(
        delayed(process_image_file)(file_path, diff_df, roi_df)
        for file_path in tqdm(file_paths, desc="Processing images")
    )
    total_time = time.time() - start_time
    print(f"Processing completed in {total_time:.2f} seconds.")

    # Save results
    log_path = os.path.join(RESULTS_FOLDER, f'{OUTPUT_FILENAME}_log.txt')
    with open(log_path, 'w') as f:
        f.write(f"Total processing time: {total_time:.2f} seconds\n")

    combined_df = pd.concat(results, ignore_index=True)
    combined_df.to_csv(os.path.join(RESULTS_FOLDER, f'{OUTPUT_FILENAME}.csv'), index=False)