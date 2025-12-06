"""
extract_basic_image_features.py

This script:

1. Loads "data_image_with_patient.xlsx", which contains one row per image
   (including patient-level information).
2. For each image file, it:
   - loads the image from disk (using the 'relative_path' column),
   - converts the image to grayscale,
   - computes a few simple features:
       * mean intensity
       * standard deviation of intensity
       * min and max intensity
       * mean Sobel edge magnitude (a simple measure of "edge energy")
3. Stores all features in a new Excel file called "image_basic_features.xlsx".

This is a FIRST, SIMPLE FEATURE SET.
Later we can add more advanced and more biologically meaningful features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import sobel


def compute_basic_features(gray_array: np.ndarray) -> dict:
    """
    Compute simple features from a grayscale image array.

    Parameters
    ----------
    gray_array : np.ndarray
        2D array (height x width) with values in [0, 255] or [0.0, 1.0].

    Returns
    -------
    dict
        Dictionary with basic features.
    """
    # Convert to float64 for safe numeric operations
    img = gray_array.astype("float64")

    # If values are in 0–255, we can keep that scale;
    # features are scale-invariant for mean/std, but this is fine.
    mean_intensity = float(img.mean())
    std_intensity = float(img.std())
    min_intensity = float(img.min())
    max_intensity = float(img.max())

    # Sobel edge magnitude
    # The 'sobel' function expects a 2D image (grayscale).
    edge_mag = sobel(img)
    mean_edge_mag = float(edge_mag.mean())
    std_edge_mag = float(edge_mag.std())

    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "mean_sobel_edge": mean_edge_mag,
        "std_sobel_edge": std_edge_mag,
    }


def main():
    print("=" * 70)
    print("EXTRACT BASIC IMAGE FEATURES – START")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Define project root and read the merged Excel file
    # ------------------------------------------------------------------
    root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {root}")

    path_merged = root / "data_image_with_patient.xlsx"

    if not path_merged.exists():
        print(f"[ERROR] File not found: {path_merged}")
        print("        Please run 'prepare_dataset_overview.py' first.")
        return

    print(f"[INFO] Loading merged table: {path_merged.name}")
    df = pd.read_excel(path_merged)

    print(f"[INFO] Table size: {df.shape[0]} rows, {df.shape[1]} columns.")

    # ------------------------------------------------------------------
    # 2. Loop over images and compute features
    # ------------------------------------------------------------------
    print("\n[STEP] Computing basic features for each image...\n")

    feature_records = []
    total = len(df)

    for idx, row in df.iterrows():
        # Progress log every 50 images (you can adjust this)
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"[INFO] Processing image {idx + 1} / {total}")

        relative_path = row["relative_path"]
        img_path = root / relative_path

        if not img_path.exists():
            print(f"[WARNING] Image file not found: {img_path}")
            continue

        try:
            # Open image with PIL
            with Image.open(img_path) as img:
                # Convert to grayscale ("L" mode in PIL)
                img_gray = img.convert("L")
                gray_array = np.array(img_gray)
        except Exception as e:
            print(f"[WARNING] Could not process image '{img_path}': {e}")
            continue

        # Compute features from the grayscale array
        feats = compute_basic_features(gray_array)

        # Build a combined record: keep original metadata + new features
        record = row.to_dict()
        record.update(feats)

        feature_records.append(record)

    # ------------------------------------------------------------------
    # 3. Create DataFrame with features and save to Excel
    # ------------------------------------------------------------------
    if not feature_records:
        print("[ERROR] No features were computed. Please check warnings above.")
        return

    df_features = pd.DataFrame(feature_records)

    output_path = root / "image_basic_features.xlsx"
    df_features.to_excel(output_path, index=False)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    n_images_original = df.shape[0]
    n_images_processed = df_features.shape[0]
    n_patients = df_features["patient_id"].nunique()

    print("\n" + "=" * 70)
    print("EXTRACT BASIC IMAGE FEATURES – DONE")
    print("=" * 70)
    print(f"[SUMMARY] Original number of images:  {n_images_original}")
    print(f"[SUMMARY] Images successfully processed: {n_images_processed}")
    print(f"[SUMMARY] Number of patients (in features table): {n_patients}")
    print(f"[SUMMARY] Features file saved as: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
