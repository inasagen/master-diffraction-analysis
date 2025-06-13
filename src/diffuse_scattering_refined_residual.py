import os
import fabio
import time
import numpy as np
from scipy.optimize import curve_fit 
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Custom utilities for image processing and fitting
from gaussian_fitting_utils import crop_image, sum_of_gaussians, diffscatter_gaussian

# === Configuration ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DIFF_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'diffraction_images')
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results', 'diffuse_scattering')
ROI_CSV_PATH = os.path.join(PROJECT_ROOT, 'results', 'ROI', 'ROI_70thresh.csv')
INIT_PARAMS = os.path.join(PROJECT_ROOT, 'results', 'curve_fit_results', 'curve_fit_init.csv')
OUTPUT_FILENAME = 'diffuse_scattering_residual'

# Parallelization settings
NUM_JOBS = 6


# === Core Functions ===

def refined_gaussian_fit(image, params, roi_row):
    if params.empty or params.isnull().values.any() or roi_row.empty:
        return pd.DataFrame([[np.nan]*6], columns=['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho'])

    cropped_image, y_offset, x_offset = crop_image(image, roi_row)
    if cropped_image is None:
        return pd.DataFrame([[np.nan]*6], columns=['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho'])

    params = params.copy()
    params['mean_x'] -= x_offset
    params['mean_y'] -= y_offset

    # Prepare input parameters
    all_params = []
    for _, row in params.iterrows():
        all_params.extend(row[['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho']].values)
    all_params.append(0)  # Add offset term

    y_len, x_len = cropped_image.shape
    x, y = np.meshgrid(np.arange(x_len), np.arange(y_len))
    coords = np.stack([x.ravel(), y.ravel()], axis=1)

    # Compute residual from initial multi-Gaussian fit
    try:
        fitted = sum_of_gaussians(coords, *all_params).reshape(cropped_image.shape)
        residual = cropped_image - fitted
    except Exception as e:
        print(f"Error in Gaussian sum fitting: {e}")
        return pd.DataFrame([[np.nan]*6], columns=['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho'])

    # Clip and smooth residual
    vmin, vmax = np.percentile(residual, [1, 99])
    residual_clipped = np.clip(residual, vmin, vmax)
    smoothed_residual = gaussian_filter(residual_clipped, sigma=3, mode='nearest')

    # Fit single 2D Gaussian to smoothed residual
    initial_guess = (np.max(smoothed_residual), x_len / 2, y_len / 2, x_len / 4, y_len / 4)
    bounds = ([0, 0, 0, 1, 1], [np.inf, x_len, y_len, 100, 100])

    try:
        popt, _ = curve_fit(diffscatter_gaussian, (x.ravel(), y.ravel()), smoothed_residual.ravel(),
                            p0=initial_guess, bounds=bounds, maxfev=10000)
        amp, x0, y0, sigx, sigy = popt
        result = pd.DataFrame([{
            'amplitude': amp,
            'mean_x': x0 + x_offset,
            'mean_y': y0 + y_offset,
            'sigma_x': sigx,
            'sigma_y': sigy,
        }])
    except Exception as e:
        print(f"Residual Gaussian fit failed: {e}")
        result = pd.DataFrame([[np.nan]*5], columns=['amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y'])

    return result


def process_image_file(file_path, all_params, roi_df):
    filename = os.path.basename(file_path)
    image = fabio.open(file_path).data

    roi_row = roi_df[roi_df['filename'] == filename]
    param_rows = all_params[all_params['filename'] == filename]

    fitted_df = refined_gaussian_fit(image, param_rows, roi_row)
    fitted_df['filename'] = filename
    return fitted_df


# === Main Script ===

if __name__ == '__main__':
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    files = sorted(f for f in os.listdir(DIFF_IMAGE_FOLDER) if f.endswith('.cbf'))
    file_paths = [os.path.join(DIFF_IMAGE_FOLDER, f) for f in files]

    roi_df = pd.read_csv(ROI_CSV_PATH)
    all_params = pd.read_csv(INIT_PARAMS)

    print(f"Using {NUM_JOBS} parallel jobs to process {len(file_paths)} images...")

    start_time = time.time()
    results = Parallel(n_jobs=NUM_JOBS)(
        delayed(process_image_file)(fp, all_params, roi_df)
        for fp in tqdm(file_paths, desc="Processing images")
    )
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds.")

    # Save results
    combined_results = pd.concat(results, ignore_index=True)
    combined_results.to_csv(os.path.join(RESULTS_FOLDER, f'{OUTPUT_FILENAME}.csv'), index=False)

    with open(os.path.join(RESULTS_FOLDER, f'{OUTPUT_FILENAME}_log.txt'), 'w') as f:
        f.write(f"Total processing time: {processing_time:.2f} seconds\n")