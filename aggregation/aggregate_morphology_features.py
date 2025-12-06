"""
B2 – Aggregate morphology features from image level to patient level.

Inputs
------
- image_morphology_features.xlsx
    One row per image (BW), including at least:
    - patient_id
    - morphology / texture features (numeric columns)

- patient_basic_features.xlsx
    One row per patient with clinical info and previous image-derived features.

Outputs
-------
- results_morphology_aggregation/patient_morphology_features.xlsx
    One row per patient with aggregated morphology features (mean, std).

- results_morphology_aggregation/patient_features_with_morphology.xlsx
    Merge of patient_basic_features.xlsx with aggregated morphology features.

This script corresponds to Phase B2 of the analysis plan:
"Subir do nível imagem -> nível paciente (features morfológicas)".
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    print("=" * 70)
    print("AGGREGATE MORPHOLOGY FEATURES (IMAGE → PATIENT) – START")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Define project root and file paths
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {project_root}")

    morph_file = project_root / "image_morphology_features.xlsx"
    basic_file = project_root / "patient_basic_features.xlsx"

    out_dir = project_root / "results_morphology_aggregation"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved in: {out_dir}")

    # ------------------------------------------------------------------
    # 2. Load morphology features (image-level)
    # ------------------------------------------------------------------
    if not morph_file.exists():
        raise FileNotFoundError(f"Cannot find morphology features file: {morph_file}")

    df_morph = pd.read_excel(morph_file)
    print(f"[INFO] Morphology features table loaded: {morph_file.name}")
    print(f"[INFO] Table size (images x cols): {df_morph.shape}")

    if "patient_id" not in df_morph.columns:
        raise KeyError(
            "Column 'patient_id' not found in image_morphology_features.xlsx. "
            "Please ensure the extraction script included 'patient_id' for each image."
        )

    # Ensure patient_id is treated consistently (string)
    df_morph["patient_id"] = df_morph["patient_id"].astype(str)

    # ------------------------------------------------------------------
    # 3. Identify numeric feature columns to aggregate
    # ------------------------------------------------------------------
    # We select all numeric columns except patient_id (and obviously any
    # non-feature identifiers, if present).
    numeric_cols = df_morph.select_dtypes(include=[np.number]).columns.tolist()

    # Remove any columns you don't want to aggregate here (if they exist)
    cols_to_exclude = []  # e.g., ["mode"] if it is numeric and not a feature
    numeric_cols = [c for c in numeric_cols if c not in cols_to_exclude]

    print(f"[INFO] Numeric feature columns to aggregate ({len(numeric_cols)}):")
    for c in numeric_cols:
        print(f"   - {c}")

    if len(numeric_cols) == 0:
        raise ValueError(
            "No numeric columns found to aggregate. Please check the file."
        )

    # ------------------------------------------------------------------
    # 4. Aggregate image-level features → patient-level (mean, std)
    # ------------------------------------------------------------------
    print("\n[STEP] Aggregating image-level morphology features to patient-level...")

    agg_dict = {col: ["mean", "std"] for col in numeric_cols}

    df_patient_morph = df_morph.groupby("patient_id")[numeric_cols].agg(agg_dict)

    # Flatten MultiIndex columns: e.g., "area_mean_mean", "area_mean_std"
    df_patient_morph.columns = [
        f"{feat}_{stat}" for feat, stat in df_patient_morph.columns.to_flat_index()
    ]
    df_patient_morph = df_patient_morph.reset_index()

    print(
        f"[INFO] Aggregated patient morphology features table size: {df_patient_morph.shape}"
    )

    # Save patient-level morphology features
    out_morph = out_dir / "patient_morphology_features.xlsx"
    df_patient_morph.to_excel(out_morph, index=False)
    print(f"[INFO] Saved patient-level morphology features to: {out_morph}")

    # ------------------------------------------------------------------
    # 5. Load basic patient features and merge
    # ------------------------------------------------------------------
    if not basic_file.exists():
        print(
            f"[WARNING] Basic patient features file not found: {basic_file}\n"
            f"          Skipping merge step. You will only get patient_morphology_features.xlsx."
        )
        print("=" * 70)
        print("AGGREGATE MORPHOLOGY FEATURES – DONE (NO MERGE)")
        print("=" * 70)
        return

    df_basic = pd.read_excel(basic_file)
    print(f"\n[INFO] Basic patient features table loaded: {basic_file.name}")
    print(f"[INFO] Basic table size (patients x cols): {df_basic.shape}")

    if "patient_id" not in df_basic.columns:
        raise KeyError(
            "Column 'patient_id' not found in patient_basic_features.xlsx. "
            "Please ensure the basic table has a 'patient_id' column."
        )

    df_basic["patient_id"] = df_basic["patient_id"].astype(str)

    # Merge: one row per patient with clinical + previous image features + new morphology features
    print("\n[STEP] Merging basic patient features with morphology features...")
    df_merged = pd.merge(df_basic, df_patient_morph, on="patient_id", how="left")

    print(f"[INFO] Merged table size (patients x cols): {df_merged.shape}")

    out_merged = out_dir / "patient_features_with_morphology.xlsx"
    df_merged.to_excel(out_merged, index=False)
    print(
        f"[INFO] Saved merged patient features (clinical + morphology) to: {out_merged}"
    )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AGGREGATE MORPHOLOGY FEATURES – DONE")
    print("=" * 70)
    print(
        f"[SUMMARY] Number of images in original morphology table: {df_morph.shape[0]}"
    )
    print(
        f"[SUMMARY] Number of patients with morphology features:  {df_patient_morph.shape[0]}"
    )
    print(f"[SUMMARY] Patient morphology table:                    {out_morph}")
    print(f"[SUMMARY] Merged patient features (if basic file ok):  {out_merged}")
    print("=" * 70)


if __name__ == "__main__":
    main()
