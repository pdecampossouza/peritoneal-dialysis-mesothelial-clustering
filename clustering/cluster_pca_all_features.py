"""
cluster_pca_all_features.py

This script performs PCA and hierarchical clustering using BOTH
clinical variables and image-derived features.

It:

1. Loads "patient_basic_features.xlsx", which contains one row per patient
   with:
   - clinical information (patient_id, age, sex, incident/prevalent, etc.)
   - aggregated image features (mean of image-level features).

2. Uses ALL selected variables (clinical + image-derived) as features.
3. Standardizes the feature columns (z-score).
4. Applies PCA (Principal Component Analysis) to reduce to 2 dimensions.
5. Applies hierarchical clustering (AgglomerativeClustering) to group patients
   into a small number of clusters (e.g. 3).

6. Saves the results in a dedicated folder "results_all_features":
   - Excel file "patient_features_with_clusters_all.xlsx"
     containing all original columns + PCA coordinates + cluster labels.
   - PNG figure "patient_pca_clusters_all.png" showing patients in PCA space,
     colored by cluster.
   - LaTeX table "tab_patient_clusters_all.tex" with patient + cluster info.
   - LaTeX figure snippet "fig_patient_pca_all.tex" referencing the PNG figure.

This is an EXPLORATORY analysis combining clinical and image-derived features.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def build_latex_cluster_table(df: pd.DataFrame) -> str:
    """
    Build LaTeX code for a table summarizing patient clusters (all features).

    The table includes:
    - Patient ID
    - Age
    - Sex (Male/Female)
    - Status (Incident/Prevalent)
    - Cluster
    """
    df_tab = df.copy()

    sex_map = {0: "Male", 1: "Female"}
    status_map = {0: "Incident", 1: "Prevalent"}

    df_tab["Sex"] = df_tab["sex"].map(sex_map)
    df_tab["Status"] = df_tab["incident_prevalent"].map(status_map)

    df_latex = df_tab[
        ["patient_id", "age_years", "Sex", "Status", "cluster_all"]
    ].rename(
        columns={
            "patient_id": "Patient ID",
            "age_years": "Age (years)",
            "cluster_all": "Cluster",
        }
    )

    tabular = df_latex.to_latex(
        index=False,
        escape=False,
    )

    latex_code = (
        r"""\begin{table}[htbp]
    \centering
    \caption{Unsupervised cluster assignment for each patient based on the combined set of clinical and image-derived features.}
    \label{tab:patient_clusters_all_features}
"""
        + tabular
        + r"""
\end{table}
"""
    )

    return latex_code


def build_latex_pca_figure_snippet() -> str:
    """
    Build LaTeX code for the PCA figure snippet (all features).
    """
    latex_code = r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\textwidth]{results_all_features/patient_pca_clusters_all.png}
    \caption{Patients represented in the first two principal components (PCA1 and PCA2) computed from both clinical variables and image-derived features. Each point corresponds to one patient and is coloured according to the unsupervised cluster assignment. Patient IDs are shown as labels next to the points.}
    \label{fig:patient_pca_clusters_all_features}
\end{figure}
"""
    return latex_code


def main():
    print("=" * 70)
    print("CLUSTER PATIENTS WITH PCA (ALL FEATURES) – START")
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

    # Create output folder for this analysis
    results_dir = root / "results_all_features"
    results_dir.mkdir(exist_ok=True)
    print(f"[INFO] Results will be saved in: {results_dir}")

    # ------------------------------------------------------------------
    # 2. Define which columns will be used as features
    # ------------------------------------------------------------------
    print("\n[STEP] Defining feature columns (clinical + image-derived)...\n")

    # Clinical variables to include as features
    clinical_feature_cols = [
        "time_in_culture_days",
        "sex",
        "age_years",
        "incident_prevalent",
    ]

    # Image-derived features to include
    image_feature_cols = [
        "mean_intensity",
        "std_intensity",
        "min_intensity",
        "max_intensity",
        "mean_sobel_edge",
        "std_sobel_edge",
    ]

    feature_cols = clinical_feature_cols + image_feature_cols
    print(f"[INFO] Feature columns used in this analysis: {feature_cols}")

    # Sanity check: make sure all columns exist
    for col in feature_cols:
        if col not in df.columns:
            print(f"[WARNING] Column '{col}' not found in the table.")

    # Use only existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Patient identifier column (not used as feature)
    id_col = "patient_id"

    # Extract features
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

    df["pca1_all"] = X_pca[:, 0]
    df["pca2_all"] = X_pca[:, 1]

    explained_var = pca.explained_variance_ratio_
    print(f"[INFO] PCA explained variance ratio (all features): {explained_var}")
    print(
        f"[INFO] Total variance explained by first 2 components: "
        f"{explained_var.sum():.3f}"
    )

    # ------------------------------------------------------------------
    # 5. Cluster patients (hierarchical clustering)
    # ------------------------------------------------------------------
    print("\n[STEP] Clustering patients (AgglomerativeClustering, all features)...\n")

    # Number of clusters is exploratory. We choose 3 here as a starting point.
    n_clusters = 3

    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    cluster_labels = clusterer.fit_predict(X_scaled)

    # For human readability, use cluster labels starting at 1 instead of 0
    df["cluster_all"] = cluster_labels + 1

    print("[INFO] Cluster assignments (all features):")
    for pid, c in zip(df[id_col], df["cluster_all"]):
        print(f"  Patient {pid}: Cluster {c}")

    # ------------------------------------------------------------------
    # 6. Save updated patient table with PCA and clusters (all features)
    # ------------------------------------------------------------------
    output_excel = results_dir / "patient_features_with_clusters_all.xlsx"
    df.to_excel(output_excel, index=False)

    # ------------------------------------------------------------------
    # 7. Create a scatter plot in PCA space (all features)
    # ------------------------------------------------------------------
    print("\n[STEP] Creating PCA scatter plot (all features)...\n")

    plt.figure(figsize=(8, 6))

    # Simple color list for up to 5 clusters
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for c in sorted(df["cluster_all"].unique()):
        mask = df["cluster_all"] == c
        plt.scatter(
            df.loc[mask, "pca1_all"],
            df.loc[mask, "pca2_all"],
            label=f"Cluster {c}",
            s=80,
            color=colors[(c - 1) % len(colors)],
        )
        # Add patient IDs as text labels near the points
        for _, row in df.loc[mask].iterrows():
            plt.text(
                row["pca1_all"] + 0.02,
                row["pca2_all"] + 0.02,
                str(row[id_col]),
                fontsize=9,
            )

    plt.xlabel("PCA 1 (all features)")
    plt.ylabel("PCA 2 (all features)")
    plt.title("Patients in PCA space (clinical + image-derived features)")
    plt.legend()
    plt.tight_layout()

    output_png = results_dir / "patient_pca_clusters_all.png"
    plt.savefig(output_png, dpi=300)
    plt.close()

    # ------------------------------------------------------------------
    # 8. Generate LaTeX table and figure snippet for this analysis
    # ------------------------------------------------------------------
    print("\n[STEP] Generating LaTeX table and figure snippet (all features)...\n")

    latex_table = build_latex_cluster_table(df)
    out_table_tex = results_dir / "tab_patient_clusters_all.tex"
    out_table_tex.write_text(latex_table, encoding="utf-8")

    latex_fig = build_latex_pca_figure_snippet()
    out_fig_tex = results_dir / "fig_patient_pca_all.tex"
    out_fig_tex.write_text(latex_fig, encoding="utf-8")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    n_patients = df[id_col].nunique()

    print("\n" + "=" * 70)
    print("CLUSTER PATIENTS WITH PCA (ALL FEATURES) – DONE")
    print("=" * 70)
    print(f"[SUMMARY] Number of patients:                          {n_patients}")
    print(f"[SUMMARY] Excel with clusters (all features) saved as: {output_excel}")
    print(f"[SUMMARY] PCA scatter plot (all features) saved as:    {output_png}")
    print(f"[SUMMARY] LaTeX table saved as:                        {out_table_tex}")
    print(f"[SUMMARY] LaTeX figure snippet saved as:               {out_fig_tex}")
    print("=" * 70)


if __name__ == "__main__":
    main()
