"""
cluster_patients_pca.py

This script:

1. Loads "patient_basic_features.xlsx", which contains one row per patient
   with:
   - clinical information (patient_id, age, sex, incident/prevalent, etc.)
   - aggregated image features (mean of image-level features).

2. Separates clinical columns from feature columns.
3. Standardizes the feature columns (z-score).
4. Applies PCA (Principal Component Analysis) to reduce to 2 dimensions.
5. Applies a simple hierarchical clustering (AgglomerativeClustering)
   to group patients into a small number of clusters (e.g. 3).

6. Saves:
   - an Excel file "patient_basic_features_with_clusters.xlsx"
     containing all original columns + PCA coordinates + cluster labels.
   - a PNG figure "patient_pca_clusters.png" showing patients in PCA space,
     colored by cluster.

This is an EXPLORATORY analysis to help visualize possible groups of patients.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 70)
    print("CLUSTER PATIENTS WITH PCA – START")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Define project root and read the patient features file
    # ------------------------------------------------------------------
    root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {root}")

    path_patient_features = root / "patient_basic_features.xlsx"

    if not path_patient_features.exists():
        print(f"[ERROR] File not found: {path_patient_features}")
        print("        Please run 'aggregate_features_by_patient.py' first.")
        return

    print(f"[INFO] Loading patient features table: {path_patient_features.name}")
    df = pd.read_excel(path_patient_features)

    print(f"[INFO] Table size: {df.shape[0]} rows, {df.shape[1]} columns.")

    if df.shape[0] < 2:
        print(
            "[ERROR] Less than 2 patients in the table. Clustering is not meaningful."
        )
        return

    # ------------------------------------------------------------------
    # 2. Define clinical vs feature columns
    # ------------------------------------------------------------------
    print("\n[STEP] Defining clinical vs feature columns...\n")

    clinical_cols = [
        "patient_id",
        "time_in_culture_days",
        "sex",
        "age_years",
        "incident_prevalent",
    ]

    # All other numeric columns (besides clinical) will be considered features.
    # Alternatively, you can explicitly list them if you prefer.
    feature_cols = [col for col in df.columns if col not in clinical_cols]

    print(f"[INFO] Clinical columns: {clinical_cols}")
    print(f"[INFO] Feature columns:  {feature_cols}")

    # Check that feature columns are numeric
    df_features = df[feature_cols].copy()

    if df_features.isna().any().any():
        print("[WARNING] There are missing values (NaN) in the feature columns.")
        print("          They will be filled with the column mean for now.")
        df_features = df_features.fillna(df_features.mean())

    # ------------------------------------------------------------------
    # 3. Standardize features (z-score)
    # ------------------------------------------------------------------
    print("\n[STEP] Standardizing feature columns (z-score)...\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features.values)

    # ------------------------------------------------------------------
    # 4. Apply PCA to reduce to 2 dimensions
    # ------------------------------------------------------------------
    print("[STEP] Applying PCA (2 components)...\n")

    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_scaled)

    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    explained_var = pca.explained_variance_ratio_
    print(f"[INFO] PCA explained variance ratio: {explained_var}")
    print(
        f"[INFO] Total variance explained by first 2 components: "
        f"{explained_var.sum():.3f}"
    )

    # ------------------------------------------------------------------
    # 5. Cluster patients (hierarchical clustering)
    # ------------------------------------------------------------------
    print("\n[STEP] Clustering patients (AgglomerativeClustering)...\n")

    # Number of clusters is exploratory. We choose 3 here as a starting point.
    n_clusters = 3

    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    cluster_labels = clusterer.fit_predict(X_scaled)

    # For human readability, use cluster labels starting at 1 instead of 0
    df["cluster"] = cluster_labels + 1

    print("[INFO] Cluster assignments:")
    for pid, c in zip(df["patient_id"], df["cluster"]):
        print(f"  Patient {pid}: Cluster {c}")

    # ------------------------------------------------------------------
    # 6. Save updated patient table with PCA and clusters
    # ------------------------------------------------------------------
    output_excel = root / "patient_basic_features_with_clusters.xlsx"
    df.to_excel(output_excel, index=False)

    # ------------------------------------------------------------------
    # 7. Create a scatter plot in PCA space
    # ------------------------------------------------------------------
    print("\n[STEP] Creating PCA scatter plot...\n")

    plt.figure(figsize=(8, 6))

    # Choose a simple color map for clusters
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for c in sorted(df["cluster"].unique()):
        mask = df["cluster"] == c
        plt.scatter(
            df.loc[mask, "pca1"],
            df.loc[mask, "pca2"],
            label=f"Cluster {c}",
            s=80,
        )
        # Add patient IDs as text labels near the points
        for _, row in df.loc[mask].iterrows():
            plt.text(
                row["pca1"] + 0.02,
                row["pca2"] + 0.02,
                str(row["patient_id"]),
                fontsize=9,
            )

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Patients in PCA space (basic features)")
    plt.legend()
    plt.tight_layout()

    output_png = root / "patient_pca_clusters.png"
    plt.savefig(output_png, dpi=300)
    plt.close()

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    n_patients = df["patient_id"].nunique()

    print("\n" + "=" * 70)
    print("CLUSTER PATIENTS WITH PCA – DONE")
    print("=" * 70)
    print(f"[SUMMARY] Number of patients:                {n_patients}")
    print(f"[SUMMARY] PCA + clusters Excel saved as:     {output_excel}")
    print(f"[SUMMARY] PCA scatter plot image saved as:   {output_png}")
    print("=" * 70)


if __name__ == "__main__":
    main()
