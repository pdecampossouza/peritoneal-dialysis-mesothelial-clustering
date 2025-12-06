"""
aggregate_features_by_patient.py

This script:

1. Loads "image_basic_features.xlsx", which contains one row per image
   (including patient-level information and basic image features).
2. Checks how many images exist for each patient.
3. Aggregates image-level features to the patient level by computing the
   mean value of each feature across all images of that patient.
4. Keeps patient-level clinical information (sex, age, incident/prevalent,
   time_in_culture_days).
5. Saves the result as "patient_basic_features.xlsx", with one row per patient.

This table is a starting point for:
- exploratory analysis,
- dimensionality reduction (PCA/UMAP),
- clustering of patients.
"""

from pathlib import Path
import pandas as pd


def main():
    print("=" * 70)
    print("AGGREGATE FEATURES BY PATIENT – START")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Define project root and read the features file
    # ------------------------------------------------------------------
    root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {root}")

    path_features = root / "image_basic_features.xlsx"

    if not path_features.exists():
        print(f"[ERROR] File not found: {path_features}")
        print("        Please run 'extract_basic_image_features.py' first.")
        return

    print(f"[INFO] Loading features table: {path_features.name}")
    df = pd.read_excel(path_features)

    print(f"[INFO] Table size: {df.shape[0]} rows, {df.shape[1]} columns.")

    # ------------------------------------------------------------------
    # 2. Check number of images per patient
    # ------------------------------------------------------------------
    print("\n[STEP] Checking number of images per patient...\n")

    images_per_patient = df.groupby("patient_id")["filename"].count()
    expected_per_patient = 5 * 5 * 2  # 5 wells × 5 zones × 2 modes = 50

    for pid, count in images_per_patient.items():
        if count == expected_per_patient:
            print(f"[OK] Patient {pid}: {count} images (as expected).")
        else:
            print(
                f"[WARNING] Patient {pid}: {count} images "
                f"(expected {expected_per_patient})."
            )

    # ------------------------------------------------------------------
    # 3. Define which columns are clinical and which are features
    # ------------------------------------------------------------------
    print("\n[STEP] Defining clinical vs feature columns...\n")

    # Columns that identify or describe the patient (clinical-level)
    clinical_cols = [
        "patient_id",
        "time_in_culture_days",
        "sex",
        "age_years",
        "incident_prevalent",
    ]

    # Basic image features we computed earlier
    feature_cols = [
        "mean_intensity",
        "std_intensity",
        "min_intensity",
        "max_intensity",
        "mean_sobel_edge",
        "std_sobel_edge",
    ]

    # Sanity check: make sure columns exist
    for col in clinical_cols + feature_cols:
        if col not in df.columns:
            print(f"[WARNING] Column '{col}' not found in the table.")

    # ------------------------------------------------------------------
    # 4. Aggregate features by patient (mean across images)
    # ------------------------------------------------------------------
    print("[STEP] Aggregating image features at patient level (mean)...\n")

    # Group by patient_id and compute mean of feature columns
    df_features_mean = (
        df[["patient_id"] + feature_cols].groupby("patient_id", as_index=False).mean()
    )

    # ------------------------------------------------------------------
    # 5. Extract one row of clinical data per patient
    # ------------------------------------------------------------------
    print("[STEP] Extracting clinical data per patient...\n")

    # We assume clinical variables are constant for each patient.
    # We can take the first row per patient.
    df_clinical = (
        df[clinical_cols].drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
    )

    # Merge clinical data with aggregated features
    df_patient = df_clinical.merge(df_features_mean, on="patient_id", how="inner")

    # ------------------------------------------------------------------
    # 6. Save result to Excel
    # ------------------------------------------------------------------
    output_path = root / "patient_basic_features.xlsx"
    df_patient.to_excel(output_path, index=False)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    n_patients = df_patient["patient_id"].nunique()

    print("\n" + "=" * 70)
    print("AGGREGATE FEATURES BY PATIENT – DONE")
    print("=" * 70)
    print(f"[SUMMARY] Number of patients in final table: {n_patients}")
    print(f"[SUMMARY] Output file saved as:              {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
