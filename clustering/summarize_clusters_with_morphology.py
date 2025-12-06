"""
B5 – Cluster-level descriptive statistics (clinical + image + morphology).

Input
-----
- results_clustering_morphology/patient_features_with_morphology_clusters.xlsx
    One row per patient.
    Contains:
        - patient_id
        - clinical features
        - image-derived features
        - aggregated morphology features
        - PCA scores (PC1, PC2, ...)
        - Cluster (1..K)

Outputs
-------
Folder: results_clustering_morphology/

- cluster_summary_with_morphology.xlsx
    One row per cluster.
    Columns:
        - Cluster
        - n_patients
        - patient_ids (comma-separated)
        - For each numeric feature:
              <feature>_mean
              <feature>_std

This script corresponds to Phase B5:
"Caracterização clínica e morfológica dos clusters ao nível do paciente".
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    print("=" * 70)
    print("SUMMARIZE CLUSTERS WITH MORPHOLOGY – START")
    print("=" * 70)

    # --------------------------------------------------------------
    # 1. Paths
    # --------------------------------------------------------------
    project_root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {project_root}")

    in_file = (
        project_root
        / "results_clustering_morphology"
        / "patient_features_with_morphology_clusters.xlsx"
    )
    out_dir = project_root / "results_clustering_morphology"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved in: {out_dir}")

    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    # --------------------------------------------------------------
    # 2. Load data
    # --------------------------------------------------------------
    df = pd.read_excel(in_file)
    print(f"[INFO] Loaded clustered patient table: {in_file.name}")
    print(f"[INFO] Table size (patients x cols): {df.shape}")

    required_cols = ["patient_id", "Cluster"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in input table.")

    df["patient_id"] = df["patient_id"].astype(str)
    df["Cluster"] = df["Cluster"].astype(int)

    # --------------------------------------------------------------
    # 3. Numeric features
    # --------------------------------------------------------------
    # We keep all numeric columns except 'Cluster' (used for grouping).
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Cluster" in numeric_cols:
        numeric_cols.remove("Cluster")

    print(
        f"[INFO] Number of numeric feature columns (excluding 'Cluster'): {len(numeric_cols)}"
    )
    print("[INFO] Numeric columns to summarize by cluster:")
    for c in numeric_cols:
        print(f"   - {c}")

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns available for cluster summarization.")

    # --------------------------------------------------------------
    # 4. Group by cluster
    # --------------------------------------------------------------
    print("\n[STEP] Grouping by cluster and computing descriptive statistics...")

    grouped = df.groupby("Cluster")

    # Cluster sizes and patient lists
    cluster_sizes = grouped.size()
    cluster_patient_ids = grouped["patient_id"].apply(
        lambda s: ", ".join(sorted(s.astype(str)))
    )

    # Means and stds for numeric features
    means = grouped[numeric_cols].mean()
    stds = grouped[numeric_cols].std(ddof=1)

    # --------------------------------------------------------------
    # 5. Build summary table
    # --------------------------------------------------------------
    summary_rows = []
    for cluster_label in sorted(df["Cluster"].unique()):
        row = {
            "Cluster": int(cluster_label),
            "n_patients": int(cluster_sizes.loc[cluster_label]),
            "patient_ids": cluster_patient_ids.loc[cluster_label],
        }

        for col in numeric_cols:
            mean_val = means.loc[cluster_label, col]
            std_val = stds.loc[cluster_label, col]

            row[f"{col}_mean"] = mean_val
            row[f"{col}_std"] = std_val

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("Cluster").reset_index(drop=True)

    # --------------------------------------------------------------
    # 6. Save Excel
    # --------------------------------------------------------------
    out_excel = out_dir / "cluster_summary_with_morphology.xlsx"
    summary_df.to_excel(out_excel, index=False)

    print(f"[INFO] Saved cluster summary table to: {out_excel}")

    # Quick console view of basic info
    print("\n[INFO] Cluster sizes:")
    for cluster_label in sorted(df["Cluster"].unique()):
        print(
            f"   - Cluster {cluster_label}: {cluster_sizes.loc[cluster_label]} patients"
        )

    print("\n" + "=" * 70)
    print("SUMMARIZE CLUSTERS WITH MORPHOLOGY – DONE")
    print("=" * 70)
    print(f"[SUMMARY] Input clustered table:   {in_file}")
    print(f"[SUMMARY] Output summary (Excel):  {out_excel}")
    print("=" * 70)


if __name__ == "__main__":
    main()
