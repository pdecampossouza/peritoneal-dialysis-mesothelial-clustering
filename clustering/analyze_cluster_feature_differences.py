import os
import numpy as np
import pandas as pd


def main():
    # === CONFIGURAÇÃO BÁSICA ===
    project_root = r"C:\Users\acer\Desktop\fnn-pso\Paper Diálise"
    clustering_folder = os.path.join(project_root, "results_clustering_morphology")

    input_file = os.path.join(
        clustering_folder, "patient_features_with_morphology_clusters.xlsx"
    )
    output_file = os.path.join(clustering_folder, "cluster_feature_signatures.xlsx")

    print("=" * 70)
    print("ANALYZE CLUSTER FEATURE DIFFERENCES – START")
    print("=" * 70)
    print(f"[INFO] Project root folder: {project_root}")
    print(f"[INFO] Loading patient feature + cluster table: {input_file}")

    df = pd.read_excel(input_file)

    if "cluster" not in df.columns:
        raise ValueError("Column 'cluster' not found in the input file.")

    # Identificar colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remover colunas que não fazem sentido na análise de assinatura
    cols_to_exclude = ["cluster"]
    # Se tiver PCs no arquivo, também tiramos (são derivados)
    cols_to_exclude += [c for c in numeric_cols if c.upper().startswith("PC")]
    numeric_cols = [c for c in numeric_cols if c not in cols_to_exclude]

    print(f"[INFO] Number of numeric feature columns considered: {len(numeric_cols)}")
    print("[INFO] Numeric columns:")
    for c in numeric_cols:
        print(f"   - {c}")

    # Calcular estatísticas globais
    global_mean = df[numeric_cols].mean()
    global_std = df[numeric_cols].std(ddof=0)

    # Evitar divisão por zero
    global_std_replaced = global_std.replace(0, np.nan)

    # Tabela longa com assinatura de cada cluster
    rows = []
    for cluster_id, df_c in df.groupby("cluster"):
        c_mean = df_c[numeric_cols].mean()
        c_std = df_c[numeric_cols].std(ddof=0)

        # z_effect = (mean_cluster - mean_global) / std_global
        z_effect = (c_mean - global_mean) / global_std_replaced

        for feat in numeric_cols:
            rows.append(
                {
                    "cluster": cluster_id,
                    "feature": feat,
                    "cluster_mean": c_mean[feat],
                    "cluster_std": c_std[feat],
                    "global_mean": global_mean[feat],
                    "global_std": global_std[feat],
                    "z_effect": z_effect[feat],
                    "abs_z_effect": abs(z_effect[feat]),
                }
            )

    df_long = pd.DataFrame(rows)

    # Ordenar dentro de cada cluster pelo |z_effect|
    df_long_sorted = df_long.sort_values(
        ["cluster", "abs_z_effect"], ascending=[True, False]
    )

    # Também podemos fazer uma visão "top 10 por cluster"
    top_k = 10
    df_top_per_cluster = (
        df_long_sorted.groupby("cluster").head(top_k).reset_index(drop=True)
    )

    print(f"[INFO] Saving full cluster feature signatures to: {output_file}")

    with pd.ExcelWriter(output_file) as writer:
        df_long_sorted.to_excel(writer, sheet_name="all_features", index=False)
        df_top_per_cluster.to_excel(
            writer, sheet_name=f"top{top_k}_per_cluster", index=False
        )

    print("=" * 70)
    print("ANALYZE CLUSTER FEATURE DIFFERENCES – DONE")
    print("=" * 70)
    print("[SUMMARY] Input file: ", input_file)
    print("[SUMMARY] Output file:", output_file)
    print("=" * 70)


if __name__ == "__main__":
    main()
