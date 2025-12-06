"""
B3/B4 – PCA + hierarchical clustering using clinical + image-derived + morphology features.

Input
-----
- results_morphology_aggregation/patient_features_with_morphology.xlsx
    One row per patient.
    Contains:
        - patient_id
        - clinical features (age, sex, incident_prevalent, time_in_culture_days, etc.)
        - image-derived summary features
        - aggregated morphology features (mean/std of cell/texture metrics)

Outputs
-------
Folder: results_clustering_morphology/

- patient_features_with_morphology_clusters.xlsx
    Same as input +:
        - PC1, PC2 (scores from PCA)
        - Cluster (1..K)

- patient_pca_clusters_morphology.png
    Scatter plot of PC1 vs PC2, colored by cluster, annotated by patient_id.

- patient_dendrogram_morphology.png
    Hierarchical clustering dendrogram (Ward).

This script corresponds to Phase B3/B4:
"Redução de dimensionalidade + clustering ao nível do paciente,
incluindo as novas features morfológicas".
"""

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def main():
    print("=" * 70)
    print("CLUSTER PATIENTS WITH MORPHOLOGY – START")
    print("=" * 70)

    # --------------------------------------------------------------
    # 1. Paths
    # --------------------------------------------------------------
    project_root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {project_root}")

    in_file = (
        project_root
        / "results_morphology_aggregation"
        / "patient_features_with_morphology.xlsx"
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
    print(f"[INFO] Loaded patient feature table: {in_file.name}")
    print(f"[INFO] Table size (patients x cols): {df.shape}")

    if "patient_id" not in df.columns:
        raise KeyError(
            "Column 'patient_id' not found in patient_features_with_morphology.xlsx"
        )

    # Keep copy of identifiers
    id_cols = ["patient_id"]
    df["patient_id"] = df["patient_id"].astype(str)

    # --------------------------------------------------------------
    # 3. Select numeric features and handle missing values
    # --------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"[INFO] Number of numeric feature columns: {len(numeric_cols)}")

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for PCA/clustering.")

    print("[INFO] Numeric columns used for PCA + clustering:")
    for c in numeric_cols:
        print(f"   - {c}")

    # Create feature matrix
    X = df[numeric_cols].copy()

    # Fill missing values with column means (simple, transparent strategy)
    if X.isna().any().any():
        print("[WARNING] Missing values found in features. Filling with column means.")
        X = X.fillna(X.mean())

    # --------------------------------------------------------------
    # 4. Standardize features (z-score)
    # --------------------------------------------------------------
    print("\n[STEP] Standardizing features (z-score)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------------
    # 5. PCA for visualization and dimensionality reduction
    # --------------------------------------------------------------
    n_samples, n_features = X_scaled.shape
    max_components = min(5, n_samples, n_features)

    print(f"[STEP] Running PCA with up to {max_components} components...")
    pca = PCA(n_components=max_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    print("[INFO] PCA explained variance ratio:")
    for i, var in enumerate(explained_var, start=1):
        print(f"   PC{i}: {var:.3f}")

    # Store at least PC1 and PC2 for plotting
    df["PC1"] = X_pca[:, 0]
    if max_components >= 2:
        df["PC2"] = X_pca[:, 1]
    else:
        # Fallback: duplicate PC1 if only 1 component, just to avoid crashes in plotting
        df["PC2"] = X_pca[:, 0]

    # --------------------------------------------------------------
    # 6. Hierarchical clustering (Ward) on the first PCs
    # --------------------------------------------------------------
    print("\n[STEP] Hierarchical clustering (Ward) on PCA scores...")

    # Use first 3 PCs (or fewer if not available) for clustering
    n_pcs_for_cluster = min(3, max_components)
    X_cluster = X_pca[:, :n_pcs_for_cluster]

    print(f"[INFO] Using first {n_pcs_for_cluster} PC(s) for clustering.")

    Z = linkage(X_cluster, method="ward", metric="euclidean")

    # Choose number of clusters (consistent with previous analysis)
    n_clusters = 3
    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    df["Cluster"] = cluster_labels
    print(f"[INFO] Assigned clusters (1..{n_clusters}) to {n_samples} patients.")

    # --------------------------------------------------------------
    # 7. Save clustered table
    # --------------------------------------------------------------
    out_excel = out_dir / "patient_features_with_morphology_clusters.xlsx"
    df.to_excel(out_excel, index=False)
    print(f"[INFO] Saved clustered patient table to: {out_excel}")

    # --------------------------------------------------------------
    # 8. Plots: PCA scatter + dendrogram
    # --------------------------------------------------------------
    # 8.1 PCA scatter
    print("[STEP] Generating PCA scatter plot...")

    plt.figure(figsize=(6, 5))
    for cl in sorted(np.unique(cluster_labels)):
        mask = df["Cluster"] == cl
        plt.scatter(
            df.loc[mask, "PC1"], df.loc[mask, "PC2"], label=f"Cluster {cl}", alpha=0.8
        )

    # Annotate points with patient_id
    for _, row in df.iterrows():
        plt.text(
            row["PC1"],
            row["PC2"],
            str(row["patient_id"]),
            fontsize=8,
            ha="center",
            va="center",
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Patients – PCA with morphology features\nColored by cluster")
    plt.legend()
    plt.tight_layout()

    out_pca_png = out_dir / "patient_pca_clusters_morphology.png"
    plt.savefig(out_pca_png, dpi=300)
    plt.close()
    print(f"[INFO] Saved PCA scatter plot to: {out_pca_png}")

    # 8.2 Dendrogram
    print("[STEP] Generating dendrogram...")

    plt.figure(figsize=(8, 4))
    dendrogram(Z, labels=df["patient_id"].to_list(), leaf_rotation=90)
    plt.title("Hierarchical clustering dendrogram (Ward)\nUsing PCA scores")
    plt.xlabel("Patient")
    plt.ylabel("Distance")
    plt.tight_layout()

    out_dendro_png = out_dir / "patient_dendrogram_morphology.png"
    plt.savefig(out_dendro_png, dpi=300)
    plt.close()
    print(f"[INFO] Saved dendrogram to: {out_dendro_png}")

    # --------------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLUSTER PATIENTS WITH MORPHOLOGY – DONE")
    print("=" * 70)
    print(f"[SUMMARY] Input file:                     {in_file}")
    print(f"[SUMMARY] Number of patients:            {n_samples}")
    print(f"[SUMMARY] Number of numeric features:    {n_features}")
    print(f"[SUMMARY] PCA components used:           {max_components}")
    print(f"[SUMMARY] Clusters (maxclust):           {n_clusters}")
    print(f"[SUMMARY] Output clustered table:        {out_excel}")
    print(f"[SUMMARY] PCA plot:                      {out_pca_png}")
    print(f"[SUMMARY] Dendrogram:                    {out_dendro_png}")
    print("=" * 70)


if __name__ == "__main__":
    main()
