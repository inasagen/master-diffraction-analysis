import os
import fabio
import time
import pandas as pd
import numpy as np 
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed

# Custom utilities
from gaussian_fitting_utils import diffscatter_gaussian, sample_from_image, weight_to_amplitude


# === Configuration ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIFF_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'diffraction_images')
ROI_CSV_PATH = os.path.join(PROJECT_ROOT, 'results', 'ROI', 'ROI_800thresh.csv')
DIFFSCATTER_CSV_PATH = os.path.join(PROJECT_ROOT, 'results', 'diffuse_scattering', 'diffuse_scattering_1900thresh.csv')
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results', 'gmm_results_1900_thresh_diffscatter')

# GMM params and parallelization settings
NUM_JOBS = 6
N_GAUSSIANS_ARRAY = [5, 10, 20, 30, 40]
N_SAMPLES_ARRAY = [100, 500, 1000, 2000, 5000]


def fit_gaussians_to_image(filename, image, n_gaussians, n_samples):
    """
    Fits a Gaussian Mixture Model to the image using intensity-based sampling.
    Returns parameters and evaluation metrics (BIC, AIC, time).
    """
    start_time = time.time()
    num_samples = n_samples * n_gaussians
    coords = sample_from_image(image, num_samples)
    coords_df = pd.DataFrame(coords, columns=['y', 'x'])[['x', 'y']]

    gmm = GaussianMixture(
        n_components=n_gaussians, covariance_type='full',
        reg_covar=1e-3, max_iter=500
    )
    gmm.fit(coords_df)
    processing_time = time.time() - start_time

    bic_value = gmm.bic(coords_df)
    aic_value = gmm.aic(coords_df)

    height, width = image.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    coords_grid = np.column_stack([xx.ravel(), yy.ravel()])

    total_intensity = np.sum(image)
    params_list = []

    for i in range(n_gaussians):
        mean_x, mean_y = gmm.means_[i]
        cov_matrix = gmm.covariances_[i]
        sigma_x = np.sqrt(cov_matrix[0, 0])
        sigma_y = np.sqrt(cov_matrix[1, 1])
        rho = cov_matrix[0, 1] / (sigma_x * sigma_y) if sigma_x * sigma_y > 0 else 0
        amplitude = weight_to_amplitude(gmm.weights_[i], sigma_x, sigma_y, rho, total_intensity)

        params_list.append([filename, amplitude, mean_x, mean_y, sigma_x, sigma_y, rho])

    params_df = pd.DataFrame(params_list, columns=[
        'filename', 'amplitude', 'mean_x', 'mean_y', 'sigma_x', 'sigma_y', 'rho'
    ])
    eval_df = pd.DataFrame([{
        'filename': filename,
        'n_gaussians': n_gaussians,
        'n_samples': n_samples,
        'BIC': bic_value,
        'AIC': aic_value,
        'time': processing_time
    }])
    return params_df, eval_df


def preprocess_image(image, diffscatter_row, roi_row):
    """
    Subtracts fitted background Gaussian and crops the image using ROI.
    """
    ny, nx = image.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))

    params = diffscatter_row.iloc[0].values.flatten()
    background = diffscatter_gaussian((x, y), *params[:-1]).reshape(image.shape)
    corrected = np.clip(image - background - 70, 0, None)

    y_min, y_max = roi_row['y_min'].iloc[0], roi_row['y_max'].iloc[0]
    x_min, x_max = roi_row['x_min'].iloc[0], roi_row['x_max'].iloc[0]

    return corrected[y_min:y_max, x_min:x_max]


def process_image_file(file_path, diffscatter_df, roi_df, n_gaussians, n_samples):
    """
    Processes a single image file: background correction, cropping, GMM fitting.
    Returns parameter and evaluation DataFrames.
    """
    filename = os.path.basename(file_path)
    image = fabio.open(file_path).data

    roi_row = roi_df[roi_df['filename'] == filename]
    diffscatter_row = diffscatter_df[diffscatter_df['filename'] == filename]

    if roi_row.empty or diffscatter_row.empty:
        print(f"[SKIP] {filename}: Missing ROI ({roi_row.empty}) or diff scatter ({diffscatter_row.empty})")
        return (
            pd.DataFrame([{
                'filename': filename,
                'amplitude': np.nan, 'mean_x': np.nan, 'mean_y': np.nan,
                'sigma_x': np.nan, 'sigma_y': np.nan, 'rho': np.nan
            }]),
            pd.DataFrame([{
                'filename': filename,
                'n_gaussians': n_gaussians,
                'n_samples': n_samples,
                'BIC': np.nan, 'AIC': np.nan, 'time': np.nan
            }])
        )

    image_cropped = preprocess_image(image, diffscatter_row, roi_row)
    params_df, eval_df = fit_gaussians_to_image(filename, image_cropped, n_gaussians, n_samples)

    # Shift fitted means back to original image coordinates
    params_df['mean_x'] += roi_row['x_min'].iloc[0]
    params_df['mean_y'] += roi_row['y_min'].iloc[0]

    return params_df, eval_df


# === Main Execution ===
if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Load data and file paths
    file_paths = sorted([
        os.path.join(DIFF_IMAGE_FOLDER, f) for f in os.listdir(DIFF_IMAGE_FOLDER) if f.endswith('.cbf')
    ])
    roi_df = pd.read_csv(ROI_CSV_PATH)
    diffscatter_df = pd.read_csv(DIFFSCATTER_CSV_PATH)

    eval_results = []

    for n_gaussians in N_GAUSSIANS_ARRAY:
        for n_samples in N_SAMPLES_ARRAY:
            output_csv = os.path.join(RESULTS_FOLDER, f"gmm_fits_{n_gaussians}g_{n_samples}s.csv")
            if os.path.exists(output_csv):
                print(f"[SKIP] Already exists: {output_csv}")
                continue

            print(f"[RUN] GMM: {n_gaussians} Gaussians, {n_samples} Samples")

            results = Parallel(n_jobs=NUM_JOBS)(
                delayed(process_image_file)(fp, diffscatter_df, roi_df, n_gaussians, n_samples)
                for fp in tqdm(file_paths, desc=f"{n_gaussians}G_{n_samples}S")
            )

            params_dfs, eval_dfs = zip(*results)
            pd.concat(params_dfs, ignore_index=True).to_csv(output_csv, index=False)
            eval_results.extend(eval_dfs)

    # Save final evaluation metrics
    eval_path = os.path.join(RESULTS_FOLDER, "gmm_fits_aic_bic_time.csv")
    pd.concat(eval_results, ignore_index=True).to_csv(eval_path, index=False)

    print("[DONE] All images processed.")