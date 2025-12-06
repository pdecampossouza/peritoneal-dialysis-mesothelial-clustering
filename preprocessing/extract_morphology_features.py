"""
extract_morphology_features.py

Phase B1: Extract classical, interpretable morphology features from
microscope images.

This script:

1. Loads the table "data_image_with_patient.xlsx", which contains
   one row per image with:
   - patient_id
   - well
   - zone
   - mode (1 = RGB, 2 = BW)
   - relative path to the image file

2. Selects ONLY the BW images (mode = 2), which are more suitable for
   segmentation and texture analysis.

3. For each BW image, it:
   - loads the image in grayscale
   - creates a circular mask to keep only the central field of view
   - applies a Gaussian filter and Otsu threshold to segment cells
   - removes very small objects (likely noise)
   - computes region-based features from the segmented objects:
       * cell_count
       * cell_density (cells per pixel in the valid masked area)
       * mean_cell_area, std_cell_area
       * mean_eccentricity, std_eccentricity
       * mean_aspect_ratio (major_axis_length / minor_axis_length)
       * mean_solidity
   - computes simple GLCM texture features from the masked grayscale image:
       * glcm_contrast
       * glcm_homogeneity
       * glcm_energy
       * glcm_correlation

4. Saves all features in:
       "image_morphology_features.xlsx"

This script is EXPLORATORY and aims to create biologically interpretable
features describing cell density, shape, and texture.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops


def create_circular_mask(h: int, w: int, radius_scale: float = 0.95):
    """
    Create a boolean circular mask for an image of size (h, w).

    Parameters
    ----------
    h, w : int
        Height and width of the image.
    radius_scale : float
        Scale factor for the radius relative to half of the
        smallest dimension. Values < 1.0 shrink the radius slightly
        to avoid the dark border of the microscope field.

    Returns
    -------
    mask : np.ndarray of bool
        True inside the circle, False outside.
    """
    center_y, center_x = h / 2.0, w / 2.0
    radius = radius_scale * min(h, w) / 2.0

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = dist_from_center <= radius
    return mask


def compute_glcm_features(image_gray: np.ndarray, mask: np.ndarray):
    """
    Compute simple GLCM texture features from a masked grayscale image.

    The image is first rescaled to 8-bit (0..255), and pixels outside
    the mask are set to 0. Then a grey-level co-occurrence matrix is
    computed with distance 1 and angle 0.

    Returns
    -------
    dict with keys:
        glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation
    """
    # Apply mask
    img_masked = image_gray.copy()
    img_masked[~mask] = 0

    # Rescale to 8-bit
    img_min = img_masked.min()
    img_max = img_masked.max()
    if img_max > img_min:
        img_norm = (img_masked - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img_masked)

    img_uint8 = (img_norm * 255).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(
        img_uint8,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    feats = {}
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop)[0, 0])

    return feats


def compute_morphology_features(binary_mask: np.ndarray, valid_mask: np.ndarray):
    """
    Compute object-based morphology features from a binary segmentation.

    Parameters
    ----------
    binary_mask : np.ndarray of bool
        Binary mask where True indicates cell pixels.
    valid_mask : np.ndarray of bool
        Mask indicating the valid field of view (circular area).

    Returns
    -------
    dict with keys:
        cell_count, cell_density, mean_cell_area, std_cell_area,
        mean_eccentricity, std_eccentricity,
        mean_aspect_ratio, mean_solidity
    """
    # Restrict segmentation to valid area
    seg = binary_mask & valid_mask

    # Remove very small objects (noise)
    seg_clean = remove_small_objects(seg, min_size=30)

    labeled = label(seg_clean)
    props = regionprops(labeled)

    cell_count = len(props)
    valid_area = float(valid_mask.sum())  # in pixels

    if valid_area > 0:
        cell_density = cell_count / valid_area
    else:
        cell_density = np.nan

    areas = []
    eccentricities = []
    aspect_ratios = []
    solidities = []

    for region in props:
        areas.append(region.area)
        eccentricities.append(region.eccentricity)
        solidities.append(region.solidity)

        # Aspect ratio = major / minor axis length
        maj = region.major_axis_length
        min_ax = region.minor_axis_length
        if min_ax > 0:
            aspect_ratios.append(maj / min_ax)

    def safe_mean(values):
        return float(np.mean(values)) if len(values) > 0 else np.nan

    def safe_std(values):
        return float(np.std(values, ddof=1)) if len(values) > 1 else np.nan

    feats = {
        "cell_count": float(cell_count),
        "cell_density": float(cell_density),
        "mean_cell_area": safe_mean(areas),
        "std_cell_area": safe_std(areas),
        "mean_eccentricity": safe_mean(eccentricities),
        "std_eccentricity": safe_std(eccentricities),
        "mean_aspect_ratio": safe_mean(aspect_ratios),
        "mean_solidity": safe_mean(solidities),
    }

    return feats


def main():
    print("=" * 70)
    print("EXTRACT MORPHOLOGY FEATURES – START")
    print("=" * 70)

    root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {root}")

    path_merged = root / "data_image_with_patient.xlsx"
    if not path_merged.exists():
        print(f"[ERROR] File not found: {path_merged}")
        print("        Please run 'prepare_dataset_overview.py' first.")
        return

    print(f"[INFO] Loading image table: {path_merged.name}")
    df = pd.read_excel(path_merged)
    print(f"[INFO] Table size: {df.shape[0]} rows, {df.shape[1]} columns.")

    # We expect at least these columns to exist
    required_cols = ["patient_id", "well", "zone", "mode_code", "relative_path"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Column '{col}' not found in table.")
            return

    # Keep only BW images (mode_code == 2)
    df_bw = df[df["mode_code"] == 2].copy()
    n_bw = df_bw.shape[0]
    print(f"[INFO] Number of BW images (mode 2): {n_bw}")

    if n_bw == 0:
        print("[ERROR] No BW images found (mode_code == 2). Nothing to do.")
        return

    features_list = []

    for idx, row in df_bw.iterrows():
        img_rel = row["relative_path"]
        img_path = root / img_rel

        print(f"[INFO] Processing image {len(features_list)+1} / {n_bw}: {img_rel}")

        if not img_path.exists():
            print(f"  [WARNING] Image file not found: {img_path}")
            continue

        # ------------------------------------------------------------------
        # Load image in grayscale (using skimage, which handles accents)
        # ------------------------------------------------------------------
        try:
            # as_gray=True já devolve uma imagem 2D (float, 0..1)
            img_gray = imread(str(img_path), as_gray=True)
        except Exception as e:
            print(f"  [WARNING] Failed to read image with skimage: {img_path}")
            print(f"           Error: {e}")
            continue

        if img_gray is None:
            print(f"  [WARNING] Failed to read image (None returned): {img_path}")
            continue

        h, w = img_gray.shape

        # ------------------------------------------------------------------
        # Create circular mask to keep only the central field of view
        # ------------------------------------------------------------------
        mask_circle = create_circular_mask(h, w, radius_scale=0.95)

        # ------------------------------------------------------------------
        # Smooth image and apply Otsu threshold for segmentation
        # ------------------------------------------------------------------
        img_float = img_gray.astype(np.float32) / 255.0
        img_smooth = gaussian(img_float, sigma=1.0)

        thresh = threshold_otsu(img_smooth[mask_circle])
        # Cells tend to be slightly darker than background
        binary = img_smooth < thresh
        binary &= mask_circle

        # ------------------------------------------------------------------
        # Compute morphology features
        # ------------------------------------------------------------------
        morph_feats = compute_morphology_features(binary, mask_circle)

        # ------------------------------------------------------------------
        # Compute texture features (GLCM)
        # ------------------------------------------------------------------
        glcm_feats = compute_glcm_features(img_gray, mask_circle)

        # ------------------------------------------------------------------
        # Combine all features + metadata
        # ------------------------------------------------------------------
        feat_row = {
            "patient_id": row["patient_id"],
            "well": row["well"],
            "zone": row["zone"],
            "mode_code": row["mode_code"],
            "relative_path": row["relative_path"],
        }
        feat_row.update(morph_feats)
        feat_row.update(glcm_feats)

        features_list.append(feat_row)

    if len(features_list) == 0:
        print("[ERROR] No features could be computed. Please check the images.")
        return

    df_features = pd.DataFrame(features_list)
    print("\n[INFO] Morphology features table size:", df_features.shape)

    output_path = root / "image_morphology_features.xlsx"
    df_features.to_excel(output_path, index=False)

    print("\n" + "=" * 70)
    print("EXTRACT MORPHOLOGY FEATURES – DONE")
    print("=" * 70)
    print(f"[SUMMARY] Number of BW images processed: {df_features.shape[0]}")
    print(f"[SUMMARY] Output file saved as: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
