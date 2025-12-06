import os
from pathlib import Path
from functools import reduce

import pandas as pd


def load_cluster_file(path: Path, method_name: str):
    """
    Tenta carregar um arquivo de clusters e devolver:
    - df reduzido (patient_id, info clínica básica, cluster_method)
    - nome da coluna de cluster usada
    Se não encontrar ou der problema, retorna None.
    """
    if not path.exists():
        print(f"[WARN] File for method '{method_name}' not found: {path}")
        return None

    print(f"[INFO] Loading cluster file for '{method_name}': {path}")
    df = pd.read_excel(path)

    # Detectar coluna de paciente
    patient_col_candidates = [
        c for c in df.columns if "patient" in c.lower() and "id" in c.lower()
    ]
    if not patient_col_candidates:
        raise ValueError(f"No patient_id-like column found in file: {path}")
    patient_col = patient_col_candidates[0]

    # Detectar colunas clínicas básicas (se existirem)
    clinical_cols = []
    for key in ["time_in_culture_days", "sex", "age_years", "incident_prevalent"]:
        if key in df.columns:
            clinical_cols.append(key)

    # Detectar coluna de cluster
    cluster_candidates = [c for c in df.columns if "cluster" in c.lower()]
    if not cluster_candidates:
        raise ValueError(f"No cluster column found in file: {path}")
    cluster_col_original = cluster_candidates[0]
    cluster_col_new = f"cluster_{method_name}"

    # Montar dataframe reduzido
    cols_to_keep = [patient_col] + clinical_cols + [cluster_col_original]
    df_small = df[cols_to_keep].copy()

    # Renomear colunas para padronizar
    rename_map = {patient_col: "patient_id", cluster_col_original: cluster_col_new}
    df_small = df_small.rename(columns=rename_map)

    # Remover duplicatas por paciente (se houver)
    df_small = df_small.drop_duplicates(subset=["patient_id"])

    print(
        f"[INFO] Loaded '{method_name}': {df_small.shape[0]} patients, cluster column = {cluster_col_new}"
    )
    return df_small


def main():
    print("======================================================================")
    print("SUMMARIZE PATIENT CLUSTERS – START")
    print("======================================================================")

    project_root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {project_root}")

    # Definição dos arquivos esperados (ajuste se tiver nomes diferentes)
    basic_path = (
        project_root / "results_clustering" / "patient_features_with_clusters.xlsx"
    )
    morph_path = (
        project_root
        / "results_clustering_morphology"
        / "patient_features_with_morphology_clusters.xlsx"
    )
    deep_path = (
        project_root
        / "results_clustering_deep"
        / "patient_deep_features_resnet50_clusters.xlsx"
    )

    results_folder = project_root / "results_summary"
    results_folder.mkdir(exist_ok=True)

    dfs = []

    # Carrega cada tipo de cluster se o arquivo existir
    for method_name, path in [
        ("basic", basic_path),
        ("morphology", morph_path),
        ("deep", deep_path),
    ]:
        df_m = load_cluster_file(path, method_name)
        if df_m is not None:
            dfs.append(df_m)

    if not dfs:
        print("[ERROR] No cluster files were loaded. Check file paths.")
        return

    # Faz merge sucessivo por patient_id
    def merge_two(left, right):
        return pd.merge(
            left,
            right,
            on=["patient_id"]
            + [
                col
                for col in left.columns
                if col in right.columns and col != "patient_id"
            ],
            how="outer",
        )

    merged = reduce(merge_two, dfs)

    # Ordenar por patient_id
    merged = merged.sort_values(by="patient_id").reset_index(drop=True)

    # Salvar Excel completo
    excel_path = results_folder / "patient_all_clusters_summary.xlsx"
    merged.to_excel(excel_path, index=False)
    print(f"[INFO] Saved combined cluster summary to: {excel_path}")

    # Montar tabela LaTeX com subset de colunas mais interpretável
    preferred_cols = [
        "patient_id",
        "time_in_culture_days",
        "sex",
        "age_years",
        "incident_prevalent",
        "cluster_basic",
        "cluster_morphology",
        "cluster_deep",
    ]
    cols_for_latex = [c for c in preferred_cols if c in merged.columns]
    latex_df = merged[cols_for_latex].copy()

    latex_path = results_folder / "patient_all_clusters_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_df.to_latex(index=False, float_format="%.3f", escape=True))
    print(f"[INFO] Saved LaTeX table to: {latex_path}")

    # Imprime um resumo simples de contagem por cluster para cada método
    for col in merged.columns:
        if col.startswith("cluster_"):
            print(f"\n[SUMMARY] Cluster counts for {col}:")
            print(merged[col].value_counts().sort_index())

    print("======================================================================")
    print("SUMMARIZE PATIENT CLUSTERS – DONE")
    print("======================================================================")


if __name__ == "__main__":
    main()
