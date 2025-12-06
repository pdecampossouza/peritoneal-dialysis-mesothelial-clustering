"""
generate_latex_outputs.py

This script creates small LaTeX snippet files (.tex) that can be directly
included in a manuscript using \\input{...}.

It assumes that the following files already exist in the project folder:
- data_patient.xlsx
- patient_basic_features_with_clusters.xlsx   (created by cluster_patients_pca.py)
- patient_pca_clusters.png                   (PCA plot saved by cluster_patients_pca.py)

The script generates:
1) tab_patient_characteristics.tex
   - Table with basic clinical and experimental information per patient.

2) tab_patient_clusters.tex
   - Table with patient ID, clinical variables, and cluster assignment.

3) fig_patient_pca.tex
   - LaTeX figure environment including the PCA plot of patients.

All LaTeX code is saved in separate .tex files so that they can be included
with \\input{...} inside the main .tex document.
"""

from pathlib import Path
import pandas as pd


def build_patient_characteristics_table(df_patient: pd.DataFrame) -> str:
    """
    Build LaTeX code for the patient characteristics table.

    The output is a full table environment (table + tabular + caption + label).
    """
    # Create a copy so we do not modify the original DataFrame
    df = df_patient.copy()

    # Map numeric codes to human-readable categories
    sex_map = {0: "Male", 1: "Female"}
    status_map = {0: "Incident", 1: "Prevalent"}

    df["Sex"] = df["sex"].map(sex_map)
    df["Status"] = df["incident_prevalent"].map(status_map)

    # Select and rename columns for the LaTeX table
    df_latex = df[
        ["patient_id", "time_in_culture_days", "Sex", "age_years", "Status"]
    ].rename(
        columns={
            "patient_id": "Patient ID",
            "time_in_culture_days": "Time in culture (days)",
            "age_years": "Age (years)",
        }
    )

    # Use pandas to_latex to generate the tabular part
    tabular = df_latex.to_latex(
        index=False,
        escape=False,  # allow special characters if needed
    )

    # Wrap in a table environment
    latex_code = (
        r"""\begin{table}[htbp]
    \centering
    \caption{Clinical and experimental characteristics of the patients included in the peritoneal dialysis study.}
    \label{tab:patient_characteristics}
"""
        + tabular
        + r"""
\end{table}
"""
    )

    return latex_code


def build_patient_clusters_table(df_clusters: pd.DataFrame) -> str:
    """
    Build LaTeX code for a table summarizing patient clusters.

    The table includes:
    - Patient ID
    - Age
    - Sex
    - Status (Incident/Prevalent)
    - Cluster
    """
    df = df_clusters.copy()

    sex_map = {0: "Male", 1: "Female"}
    status_map = {0: "Incident", 1: "Prevalent"}

    df["Sex"] = df["sex"].map(sex_map)
    df["Status"] = df["incident_prevalent"].map(status_map)

    df_latex = df[["patient_id", "age_years", "Sex", "Status", "cluster"]].rename(
        columns={
            "patient_id": "Patient ID",
            "age_years": "Age (years)",
            "cluster": "Cluster",
        }
    )

    tabular = df_latex.to_latex(
        index=False,
        escape=False,
    )

    latex_code = (
        r"""\begin{table}[htbp]
    \centering
    \caption{Unsupervised cluster assignment for each patient based on image-derived features.}
    \label{tab:patient_clusters}
"""
        + tabular
        + r"""
\end{table}
"""
    )

    return latex_code


def build_pca_figure_snippet() -> str:
    """
    Build LaTeX code for the PCA figure snippet.

    This assumes that 'patient_pca_clusters.png' is located in the same folder
    (or in a folder that is in the LaTeX graphics path).
    """
    latex_code = r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\textwidth]{patient_pca_clusters.png}
    \caption{Patients represented in the first two principal components (PCA1 and PCA2) of the image-derived features. Each point corresponds to one patient and is coloured according to the unsupervised cluster assignment. Patient IDs are shown as labels next to the points.}
    \label{fig:patient_pca_clusters}
\end{figure}
"""
    return latex_code


def main():
    print("=" * 70)
    print("GENERATE LATEX OUTPUTS – START")
    print("=" * 70)

    root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {root}")

    path_patient = root / "data_patient.xlsx"
    path_clusters = root / "patient_basic_features_with_clusters.xlsx"
    path_pca_png = root / "patient_pca_clusters.png"

    # ------------------------------------------------------------------
    # 1. Check the required input files
    # ------------------------------------------------------------------
    if not path_patient.exists():
        print(f"[ERROR] File not found: {path_patient}")
        print("        Please make sure 'data_patient.xlsx' exists.")
        return

    if not path_clusters.exists():
        print(f"[ERROR] File not found: {path_clusters}")
        print("        Please run 'cluster_patients_pca.py' first.")
        return

    if not path_pca_png.exists():
        print(f"[WARNING] PCA plot image not found: {path_pca_png}")
        print("          The LaTeX figure snippet will still be created,")
        print("          but LaTeX will not find the image unless it is generated.")
    else:
        print(f"[INFO] Found PCA plot image: {path_pca_png.name}")

    # ------------------------------------------------------------------
    # 2. Load the data
    # ------------------------------------------------------------------
    print("\n[STEP] Loading Excel files...\n")
    df_patient = pd.read_excel(path_patient)
    df_clusters = pd.read_excel(path_clusters)

    print(
        f"[INFO] data_patient: {df_patient.shape[0]} rows, {df_patient.shape[1]} columns."
    )
    print(
        f"[INFO] patient_basic_features_with_clusters: {df_clusters.shape[0]} rows, {df_clusters.shape[1]} columns."
    )

    # ------------------------------------------------------------------
    # 3. Build and save LaTeX tables
    # ------------------------------------------------------------------
    print("\n[STEP] Generating LaTeX tables...\n")

    # 3.1 Patient characteristics table
    latex_patients = build_patient_characteristics_table(df_patient)
    out_patients_tex = root / "tab_patient_characteristics.tex"
    out_patients_tex.write_text(latex_patients, encoding="utf-8")
    print(f"[INFO] Saved patient characteristics table to: {out_patients_tex}")

    # 3.2 Patient clusters table
    latex_clusters = build_patient_clusters_table(df_clusters)
    out_clusters_tex = root / "tab_patient_clusters.tex"
    out_clusters_tex.write_text(latex_clusters, encoding="utf-8")
    print(f"[INFO] Saved patient clusters table to: {out_clusters_tex}")

    # ------------------------------------------------------------------
    # 4. Build and save LaTeX figure snippet
    # ------------------------------------------------------------------
    print("\n[STEP] Generating LaTeX figure snippet for PCA plot...\n")

    latex_fig_pca = build_pca_figure_snippet()
    out_fig_tex = root / "fig_patient_pca.tex"
    out_fig_tex.write_text(latex_fig_pca, encoding="utf-8")
    print(f"[INFO] Saved PCA figure snippet to: {out_fig_tex}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATE LATEX OUTPUTS – DONE")
    print("=" * 70)
    print(f"[SUMMARY] LaTeX tables and figure snippet saved in: {root}")
    print("You can include them in your main .tex file using, for example:")
    print("  \\input{tab_patient_characteristics}")
    print("  \\input{tab_patient_clusters}")
    print("  \\input{fig_patient_pca}")
    print("=" * 70)


if __name__ == "__main__":
    main()
