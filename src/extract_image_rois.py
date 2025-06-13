import numpy as np 
from tqdm import tqdm 
import os
import fabio
import pandas as pd
from scipy.ndimage import label


# === Configuration Parameters ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIFF_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'diffraction_images')

THRESHOLD_INTENSITY = 70
MIN_REGION_AREA = 800
OUTPUT_CSV = os.path.join(PROJECT_ROOT,'results', 'ROI', f"ROI_{THRESHOLD_INTENSITY}thresh.csv")


def find_initial_bounding_box(image: np.ndarray, threshold: int):
    """
    Finds the bounding box around all pixels in the image that exceed the given threshold.
    
    Returns:
        (y_min, y_max, x_min, x_max) if such pixels exist, otherwise None.
    """
    mask = image > threshold
    coords = np.argwhere(mask)

    if coords.size == 0:
        print("No pixels above threshold.")
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return y_min, y_max, x_min, x_max


def find_largest_connected_region(image: np.ndarray, threshold: int, area_threshold: int):
    """
    Identifies the largest connected region in the image that exceeds the threshold intensity and minimum area.
    
    Returns:
        (y_min, y_max, x_min, x_max) of the largest region, or None if not found.
    """
    binary_mask = image > threshold
    labeled_mask, num_features = label(binary_mask)

    if num_features == 0:
        print("No connected regions found.")
        return None

    region_sizes = [(labeled_mask == label_id).sum() for label_id in range(1, num_features + 1)]
    large_regions = [(i + 1, size) for i, size in enumerate(region_sizes) if size > area_threshold]

    if not large_regions:
        print("No connected regions > minimum area.")
        return None

    # Get label of largest region
    largest_label = max(large_regions, key=lambda x: x[1])[0]
    coords = np.argwhere(labeled_mask == largest_label)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return y_min, y_max, x_min, x_max

def extract_image_metadata(filename, image, bounds):
    """
    Extracts metadata including bounding box and size from the bounds.
    """
    if bounds is None:
        return {
            'filename': filename,
            'y_min': np.nan,
            'y_max': np.nan,
            'x_min': np.nan,
            'x_max': np.nan,
            'height': np.nan,
            'width': np.nan,
            'size': np.nan
        }
    y_min, y_max, x_min, x_max = bounds
    height = y_max - y_min
    width = x_max - x_min
    size = height * width

    return {
        'filename': filename,
        'y_min': y_min,
        'y_max': y_max,
        'x_min': x_min,
        'x_max': x_max,
        'height': height,
        'width': width,
        'size': size
    }


# === Main Script ===

if __name__ == '__main__':

    # Collect all .cbf image filenames
    image_files = sorted([f for f in os.listdir(DIFF_IMAGE_FOLDER) if f.endswith('.cbf')])

    # Tier 1: Initial bounding boxes
    initial_crop_data = []
    for filename in tqdm(image_files, desc='Phase 1: Initial bounding boxes'):
        file_path = os.path.join(DIFF_IMAGE_FOLDER, filename)
        image = fabio.open(file_path).data
        bounds = find_initial_bounding_box(image, threshold=THRESHOLD_INTENSITY)
        metadata = extract_image_metadata(filename, image, bounds)
        initial_crop_data.append(metadata)

    initial_df = pd.DataFrame(initial_crop_data)

    # Tier 2: Refine using largest connected region for large areas
    median_size = np.median(initial_df['size'])
    size_map = dict(zip(initial_df['filename'], initial_df['size']))
    refined_crop_data = []

    for filename in tqdm(image_files, desc='Phase 2: Refine large regions'):
        if size_map.get(filename, 0) <= median_size:
            continue

        file_path = os.path.join(DIFF_IMAGE_FOLDER, filename)
        image = fabio.open(file_path).data
        bounds = find_largest_connected_region(image, threshold=THRESHOLD_INTENSITY, area_threshold=MIN_REGION_AREA)

        if bounds is not None:
            metadata = extract_image_metadata(filename, image, bounds)
            refined_crop_data.append(metadata)

    refined_df = pd.DataFrame(refined_crop_data)

    # Merge refined data back into the original
    initial_df.set_index('filename', inplace=True)
    refined_df.set_index('filename', inplace=True)
    initial_df.update(refined_df)
    initial_df.reset_index(inplace=True)

    # Export final data
    initial_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved output to {OUTPUT_CSV}")