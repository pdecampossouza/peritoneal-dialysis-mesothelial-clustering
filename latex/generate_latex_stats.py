"""
generate_latex_stats.py

This script generates LaTeX tables and text snippets with basic statistics
of the image-derived features and PCA results.

Input:
    - patient_basic_features_with_clusters.xlsx
      (created by cluster_patients_pca.py)

Output:
    - tab_feature_overall.tex
    - tab_feature_by_cluster.tex
    - tab_pca_loadings.tex
    - pca_explained_variance.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 70)
    print("GENERATE LATEX STATS – START")
    print("=" * 70)

    root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {root}")

    path_patient_clusters = root / "patient_basic_features_with_clusters.xlsx"

    if not path_patient_clusters.exists():
        print(f"[ERROR] File not found: {path_patient_clusters}")
        print("        Please run 'cluster_patients_pca.py' first.")
        return

    print(f"[INFO] Loading table: {path_patient_clusters.name}")
    df = pd.read_excel(path_patient_clusters)
    print(f"[INFO] Table size: {df.shape[0]} rows, {df.shape[1]} columns.")

    # ------------------------------------------------------------------
    # 1. Identify clinical and feature columns
    # ------------------------------------------------------------------
    clinical_cols = [
        "patient_id",
        "time_in_culture_days",
        "sex",
        "age_years",
        "incident_prevalent",
        "cluster",
        "pca1",
        "pca2",
    ]

    feature_cols = [c for c in df.columns if c not in clinical_cols]

    print("\n[STEP] Columns detected:")
    print(f"  Feature columns:  {feature_cols}")
    print(f"  Clinical columns: {clinical_cols}")

    # Make sure we only keep existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    if len(feature_cols) == 0:
        print("[ERROR] No feature columns found. Aborting.")
        return

    # ------------------------------------------------------------------
    # 2. Overall mean and SD of each feature (all patients)
    # ------------------------------------------------------------------
    print("\n[STEP] Computing overall mean and SD of features...\n")

    df_feats = df[feature_cols].astype(float)
    overall_mean = df_feats.mean()
    overall_std = df_feats.std()

    df_overall = pd.DataFrame(
        {
            "Feature": overall_mean.index,
            "Mean": overall_mean.values,
            "SD": overall_std.values,
        }
    )

    # Format for LaTeX using pandas.to_latex
    latex_overall = (
        r"""\begin{table}[htbp]
    \centering
    \caption{Overall mean and standard deviation of image-derived features across all patients (n=9).}
    \label{tab:feature_overall}
"""
        + df_overall.to_latex(index=False, float_format="%.3f")
        + r"""
\end{table}
"""
    )

    out_overall = root / "tab_feature_overall.tex"
    out_overall.write_text(latex_overall, encoding="utf-8")
    print(f"[INFO] Saved overall feature table to: {out_overall}")

    # ------------------------------------------------------------------
    # 3. Mean ± SD of each feature by cluster
    # ------------------------------------------------------------------
    print("\n[STEP] Computing mean and SD of features by cluster...\n")

    # Group by cluster
    grouped = df.groupby("cluster")[feature_cols]

    means_by_cluster = grouped.mean()
    stds_by_cluster = grouped.std()

    # Build a table with rows = cluster, columns = features (mean ± SD)
    rows = []
    for cluster_label in sorted(df["cluster"].unique()):
        row = {"Cluster": int(cluster_label)}
        for feat in feature_cols:
            m = means_by_cluster.loc[cluster_label, feat]
            s = stds_by_cluster.loc[cluster_label, feat]
            row[feat] = f"{m:.3f} ± {s:.3f}"
        rows.append(row)

    df_cluster = pd.DataFrame(rows)

    latex_by_cluster = (
        r"""\begin{table}[htbp]
    \centering
    \caption{Mean $\pm$ standard deviation of image-derived features by unsupervised cluster.}
    \label{tab:feature_by_cluster}
"""
        + df_cluster.to_latex(index=False, escape=False)
        + r"""
\end{table}
"""
    )

    out_by_cluster = root / "tab_feature_by_cluster.tex"
    out_by_cluster.write_text(latex_by_cluster, encoding="utf-8")
    print(f"[INFO] Saved feature-by-cluster table to: {out_by_cluster}")

    # ------------------------------------------------------------------
    # 4. Recompute PCA (2 components) to get loadings and variance explained
    # ------------------------------------------------------------------
    print("\n[STEP] Recomputing PCA on feature space for LaTeX stats...\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_feats.values)

    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_scaled)

    var_ratio = pca.explained_variance_ratio_
    comp = pca.components_  # shape (2, n_features)

    # PCA loadings table: rows = features, columns = PC1 and PC2
    df_loadings = pd.DataFrame(
        {
            "Feature": feature_cols,
            "PC1_loading": comp[0, :],
            "PC2_loading": comp[1, :],
        }
    )

    latex_loadings = (
        r"""\begin{table}[htbp]
    \centering
    \caption{Loadings of image-derived features on the first two principal components.}
    \label{tab:pca_loadings}
"""
        + df_loadings.to_latex(index=False, float_format="%.3f")
        + r"""
\end{table}
"""
    )

    out_loadings = root / "tab_pca_loadings.tex"
    out_loadings.write_text(latex_loadings, encoding="utf-8")
    print(f"[INFO] Saved PCA loadings table to: {out_loadings}")

    # ------------------------------------------------------------------
    # 5. Text snippet with explained variance
    # ------------------------------------------------------------------
    print("\n[STEP] Generating LaTeX text snippet for explained variance...\n")

    pc1_var = var_ratio[0] * 100.0
    pc2_var = var_ratio[1] * 100.0
    total_var = (var_ratio[0] + var_ratio[1]) * 100.0

    text_variance = (
        r"The first two principal components explained "
        f"{pc1_var:.1f}\\% and {pc2_var:.1f}\\% of the total variance, "
        f"respectively (cumulative {total_var:.1f}\\%)."
    )

    out_var_tex = root / "pca_explained_variance.tex"
    out_var_tex.write_text(text_variance, encoding="utf-8")
    print(f"[INFO] Saved PCA explained variance snippet to: {out_var_tex}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATE LATEX STATS – DONE")
    print("=" * 70)
    print(f"[SUMMARY] LaTeX stats files saved in: {root}")
    print("  - tab_feature_overall.tex")
    print("  - tab_feature_by_cluster.tex")
    print("  - tab_pca_loadings.tex")
    print("  - pca_explained_variance.tex")
    print("=" * 70)


if __name__ == "__main__":
    main()
