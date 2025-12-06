"""
cluster_clinical_stats.py

This script generates clinical summaries by cluster for two types of
clustering results:

1) Image-only clustering:
   - Input: "patient_basic_features_with_clusters.xlsx"
   - Column with cluster labels: "cluster"

2) Clinical + image-derived features clustering (optional):
   - Input: "results_all_features/patient_features_with_clusters_all.xlsx"
   - Column with cluster labels: "cluster_all"

For each clustering variant, the script computes, per cluster:
    - Number of patients (N)
    - Mean ± SD of age (age_years)
    - Mean ± SD of time_in_culture_days
    - Number of males / females
    - Number of incident / prevalent patients

Outputs:
    - Folder "results_cluster_stats" in the project root, containing:
        * "tab_cluster_clinical_basic.tex"
        * "tab_cluster_clinical_all_features.tex" (if all-features file exists)
"""

from pathlib import Path

import pandas as pd


def summarize_by_cluster(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """
    Compute clinical summaries by cluster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing clinical variables and a cluster column.
    cluster_col : str
        Name of the column with cluster labels.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per cluster.
    """
    rows = []
    for cluster_id, group in df.groupby(cluster_col):
        n_patients = group.shape[0]

        age_mean = group["age_years"].mean()
        age_sd = group["age_years"].std()

        time_mean = group["time_in_culture_days"].mean()
        time_sd = group["time_in_culture_days"].std()

        n_male = (group["sex"] == 0).sum()
        n_female = (group["sex"] == 1).sum()

        n_incident = (group["incident_prevalent"] == 0).sum()
        n_prevalent = (group["incident_prevalent"] == 1).sum()

        row = {
            "Cluster": int(cluster_id),
            "N": int(n_patients),
            "Age_mean": age_mean,
            "Age_sd": age_sd,
            "TimeCulture_mean": time_mean,
            "TimeCulture_sd": time_sd,
            "N_male": int(n_male),
            "N_female": int(n_female),
            "N_incident": int(n_incident),
            "N_prevalent": int(n_prevalent),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)
    return summary_df


def summary_to_latex_table(summary_df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Convert the summary DataFrame into a LaTeX table.

    Age and time in culture are represented as "mean ± SD".
    """
    df = summary_df.copy()

    # Create formatted columns for Age and Time in culture
    df["Age (years)"] = df.apply(
        lambda r: (
            f"{r['Age_mean']:.1f} ± {r['Age_sd']:.1f}"
            if r["N"] > 1
            else f"{r['Age_mean']:.1f}"
        ),
        axis=1,
    )
    df["Time in culture (days)"] = df.apply(
        lambda r: (
            f"{r['TimeCulture_mean']:.1f} ± {r['TimeCulture_sd']:.1f}"
            if r["N"] > 1
            else f"{r['TimeCulture_mean']:.1f}"
        ),
        axis=1,
    )

    # Keep only the formatted / count columns
    df_latex = df[
        [
            "Cluster",
            "N",
            "Age (years)",
            "Time in culture (days)",
            "N_male",
            "N_female",
            "N_incident",
            "N_prevalent",
        ]
    ].rename(
        columns={
            "N": "N",
            "N_male": "Male",
            "N_female": "Female",
            "N_incident": "Incident",
            "N_prevalent": "Prevalent",
        }
    )

    tabular = df_latex.to_latex(index=False, escape=False)

    latex_code = (
        r"\begin{table}[htbp]" + "\n"
        r"    \centering" + "\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n" + tabular + "\n\\end{table}\n"
    )

    return latex_code


def main():
    print("=" * 70)
    print("CLUSTER CLINICAL STATS – START")
    print("=" * 70)

    root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {root}")

    # Output folder
    results_dir = root / "results_cluster_stats"
    results_dir.mkdir(exist_ok=True)
    print(f"[INFO] Results will be saved in: {results_dir}")

    # ------------------------------------------------------------------
    # 1) Image-only clustering
    # ------------------------------------------------------------------
    basic_path = root / "patient_basic_features_with_clusters.xlsx"

    if not basic_path.exists():
        print(f"[WARNING] File not found: {basic_path}")
        print("          Please run 'cluster_patients_pca.py' first.")
    else:
        print(f"\n[STEP] Processing image-only clustering: {basic_path.name}\n")
        df_basic = pd.read_excel(basic_path)

        if "cluster" not in df_basic.columns:
            print("[WARNING] Column 'cluster' not found in basic file. Skipping.")
        else:
            summary_basic = summarize_by_cluster(df_basic, "cluster")
            print("[INFO] Image-only clustering summary by cluster:")
            print(summary_basic)

            caption_basic = (
                "Clinical characteristics of patients by cluster, based on "
                "image-derived features only."
            )
            label_basic = "tab:cluster_clinical_basic"

            latex_basic = summary_to_latex_table(
                summary_basic, caption_basic, label_basic
            )
            out_basic_tex = results_dir / "tab_cluster_clinical_basic.tex"
            out_basic_tex.write_text(latex_basic, encoding="utf-8")
            print(f"[INFO] Saved LaTeX table (image-only) to: {out_basic_tex}")

    # ------------------------------------------------------------------
    # 2) Clinical + image-derived clustering
    # ------------------------------------------------------------------
    allfeat_path = (
        root / "results_all_features" / "patient_features_with_clusters_all.xlsx"
    )

    if not allfeat_path.exists():
        print(f"\n[WARNING] All-features clustering file not found: {allfeat_path}")
        print(
            "          If you want this summary, run 'cluster_pca_all_features.py' first."
        )
    else:
        print(
            f"\n[STEP] Processing clinical + image-derived clustering: {allfeat_path.name}\n"
        )
        df_all = pd.read_excel(allfeat_path)

        if "cluster_all" not in df_all.columns:
            print(
                "[WARNING] Column 'cluster_all' not found in all-features file. Skipping."
            )
        else:
            summary_all = summarize_by_cluster(df_all, "cluster_all")
            print("[INFO] All-features clustering summary by cluster:")
            print(summary_all)

            caption_all = (
                "Clinical characteristics of patients by cluster, based on "
                "the combined set of clinical and image-derived features."
            )
            label_all = "tab:cluster_clinical_all_features"

            latex_all = summary_to_latex_table(summary_all, caption_all, label_all)
            out_all_tex = results_dir / "tab_cluster_clinical_all_features.tex"
            out_all_tex.write_text(latex_all, encoding="utf-8")
            print(f"[INFO] Saved LaTeX table (all features) to: {out_all_tex}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLUSTER CLINICAL STATS – DONE")
    print("=" * 70)
    print(
        f"[SUMMARY] LaTeX clinical summary tables (if generated) are in: {results_dir}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
