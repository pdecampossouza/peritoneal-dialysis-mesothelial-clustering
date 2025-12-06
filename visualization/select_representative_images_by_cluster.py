import os
import numpy as np
import pandas as pd


def main():
    # === CONFIGURAÇÃO ===
    project_root = r"C:\Users\acer\Desktop\fnn-pso\Paper Diálise"

    morphology_folder = project_root  # onde está image_morphology_features.xlsx
    clustering_folder = os.path.join(project_root, "results_clustering_morphology")

    image_feats_file = os.path.join(morphology_folder, "image_morphology_features.xlsx")
    patient_cluster_file = os.path.join(
        clustering_folder, "patient_features_with_morphology_clusters.xlsx"
    )

    output_file = os.path.join(
        clustering_folder, "representative_images_by_cluster.csv"
    )

    print("=" * 70)
    print("SELECT REPRESENTATIVE IMAGES BY CLUSTER – START")
    print("=" * 70)
    print(f"[INFO] Project root folder: {project_root}")
    print(f"[INFO] Loading image-level morphology features: {image_feats_file}")
    print(f"[INFO] Loading patient-level cluster table: {patient_cluster_file}")

    df_img = pd.read_excel(image_feats_file)
    df_pat = pd.read_excel(patient_cluster_file)

    # Ajuste aqui se o nome da coluna de paciente for diferente
    patient_col_img = "patient_id"
    patient_col_pat = "patient_id"

    if patient_col_img not in df_img.columns:
        raise ValueError(
            f"Column '{patient_col_img}' not found in image_morphology_features.xlsx"
        )

    if patient_col_pat not in df_pat.columns:
        raise ValueError(
            f"Column '{patient_col_pat}' not found in patient_features_with_morphology_clusters.xlsx"
        )

    if "cluster" not in df_pat.columns:
        raise ValueError(
            "Column 'cluster' not found in patient_features_with_morphology_clusters.xlsx"
        )

    # Merge para associar cada imagem a um cluster
    df_merged = df_img.merge(
        df_pat[[patient_col_pat, "cluster"]],
        left_on=patient_col_img,
        right_on=patient_col_pat,
        how="left",
    )

    # Identificar colunas numéricas de morfologia na tabela de imagens
    numeric_cols_img = df_merged.select_dtypes(include=[np.number]).columns.tolist()
    # remover cluster etc. se entrar
    cols_to_exclude = ["cluster"]
    numeric_cols_img = [c for c in numeric_cols_img if c not in cols_to_exclude]

    print(
        f"[INFO] Numeric image-level feature columns considered ({len(numeric_cols_img)}):"
    )
    for c in numeric_cols_img:
        print(f"   - {c}")

    # Padronizar (z-score) as features a nível de imagem
    img_means = df_merged[numeric_cols_img].mean()
    img_stds = df_merged[numeric_cols_img].std(ddof=0).replace(0, np.nan)

    df_z = (df_merged[numeric_cols_img] - img_means) / img_stds
    df_z.columns = [f"{c}_z" for c in numeric_cols_img]

    df_merged = pd.concat([df_merged, df_z], axis=1)

    # Vamos pegar, para cada cluster e para cada feature, as imagens mais extremas
    # Você pode ajustar esse K
    top_k = 3  # por cluster/feature/direção

    rows = []
    clusters = sorted(df_merged["cluster"].dropna().unique())
    print(f"[INFO] Clusters found in image table: {clusters}")

    for cluster_id in clusters:
        df_c = df_merged[df_merged["cluster"] == cluster_id]

        print(f"[INFO] Processing cluster {cluster_id} with {len(df_c)} images...")

        for feat in numeric_cols_img:
            z_col = f"{feat}_z"

            # top K (maiores z) = "altos valores da feature"
            df_top = df_c.sort_values(z_col, ascending=False).head(top_k)
            for _, row in df_top.iterrows():
                rows.append(
                    {
                        "cluster": cluster_id,
                        "patient_id": row[patient_col_img],
                        "feature": feat,
                        "direction": "high",
                        "feature_value": row[feat],
                        "z_value": row[z_col],
                        # Campos úteis para localizar a imagem
                        "well": row.get("well", None),
                        "mode_code": row.get("mode_code", None),
                        "image_name": row.get("image_name", None),  # caso exista
                    }
                )

            # bottom K (menores z) = "baixos valores da feature"
            df_bottom = df_c.sort_values(z_col, ascending=True).head(top_k)
            for _, row in df_bottom.iterrows():
                rows.append(
                    {
                        "cluster": cluster_id,
                        "patient_id": row[patient_col_img],
                        "feature": feat,
                        "direction": "low",
                        "feature_value": row[feat],
                        "z_value": row[z_col],
                        "well": row.get("well", None),
                        "mode_code": row.get("mode_code", None),
                        "image_name": row.get("image_name", None),
                    }
                )

    df_representative = pd.DataFrame(rows)

    print(f"[INFO] Saving representative images list to: {output_file}")
    df_representative.to_csv(output_file, index=False, sep=";")

    print("=" * 70)
    print("SELECT REPRESENTATIVE IMAGES BY CLUSTER – DONE")
    print("=" * 70)
    print("[SUMMARY] Output file:", output_file)
    print("=" * 70)


if __name__ == "__main__":
    main()
