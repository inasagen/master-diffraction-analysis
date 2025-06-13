import os
import fabio
from scipy.ndimage import label
from scipy.optimize import curve_fit 
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from gaussian_fitting_utils import diffscatter_gaussian


# === Configuration ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DIFF_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'diffraction_images')
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results', 'diffuse_scattering')
ROI_CSV_PATH = os.path.join(PROJECT_ROOT, 'results', 'ROI', 'ROI_70thresh.csv')

# Threshold and parallelization settings
THRESHOLD = 800
BACKGROUND_THRESHOLD = 70
NUM_JOBS = 6

RESULT_CSV = os.path.join(RESULTS_FOLDER, f'diffuse_scattering_{THRESHOLD}thresh.csv')


# === Image Processing Function ===

def process_image(file_path, roi_df, threshold):
    filename = os.path.basename(file_path)
    image = fabio.open(file_path).data

    roi_row = roi_df[roi_df['filename'] == filename]
    if roi_row.empty:
        print(f"ROI not found for {filename}")
        return None

    # Extract ROI
    y_min, y_max = int(roi_row['y_min'].iloc[0]), int(roi_row['y_max'].iloc[0])
    x_min, x_max = int(roi_row['x_min'].iloc[0]), int(roi_row['x_max'].iloc[0])
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Background subtraction and normalization
    cropped_image = np.clip(cropped_image - BACKGROUND_THRESHOLD, 0, None)
    max_intensity = np.max(cropped_image)
    if max_intensity == 0:
        print(f"No intensity in {filename}")
        return None
    cropped_image = cropped_image / max_intensity

    # Create meshgrid
    ny, nx = cropped_image.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))

    # Mask and prepare data
    mask = cropped_image > (threshold / max_intensity)
    z = cropped_image[~mask]
    x_flat = x[~mask].ravel()
    y_flat = y[~mask].ravel()

    if len(z) < 10:
        print(f"Not enough background pixels for {filename}")
        return None

    # Fit Gaussian to background
    initial_guess = (np.max(z), nx / 2, ny / 2, 10, 10)
    try:
        params, _ = curve_fit(diffscatter_gaussian, (x_flat, y_flat), z,
                              p0=initial_guess, maxfev=10000)
    except RuntimeError:
        print(f"Fit failed for {filename}")
        return None

    # Save fitted parameters
    amp, x0, y0, sigx, sigy = params
    result = pd.DataFrame([{
        'filename': filename,
        'amplitude': amp * max_intensity,
        'mean_x': x0 + x_min,
        'mean_y': y0 + y_min,
        'sigma_x': sigx,
        'sigma_y': sigy
    }])
    return result


# === Main Execution ===

if __name__ == '__main__':
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    files = sorted(f for f in os.listdir(DIFF_IMAGE_FOLDER) if f.endswith('.cbf'))
    file_paths = [os.path.join(DIFF_IMAGE_FOLDER, f) for f in files]
    roi_df = pd.read_csv(ROI_CSV_PATH)

    print(f"Using {NUM_JOBS} parallel jobs to process {len(file_paths)} images...")

    start_time = time.time()
    results = Parallel(n_jobs=NUM_JOBS)(
        delayed(process_image)(fp, roi_df, THRESHOLD)
        for fp in tqdm(file_paths, desc="Processing images")
    )
    processing_time = time.time() - start_time

    # Filter valid results
    results = [r for r in results if r is not None]
    print(f"Processing completed in {processing_time:.2f} seconds with {len(results)} successful fits.")

    # Combine and save results
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        combined_df.to_csv(RESULT_CSV, index=False)
        print(f"Results saved to {RESULT_CSV}")
    else:
        print("No results to save.")