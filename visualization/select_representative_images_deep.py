import os
from pathlib import Path

import numpy as np
import pandas as pd


def main(
    k_per_cluster: int = 5,
):
    print("======================================================================")
    print("SELECT REPRESENTATIVE IMAGES PER DEEP CLUSTER – START")
    print("======================================================================")

    project_root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {project_root}")

    # Caminhos dos arquivos de entrada
    deep_image_path = (
        project_root / "results_deep_features" / "image_deep_features_resnet50.xlsx"
    )
    meta_path = project_root / "data_image_with_patient.xlsx"
    cluster_patient_path = (
        project_root
        / "results_clustering_deep"
        / "patient_deep_features_resnet50_clusters.xlsx"
    )

    results_folder = project_root / "results_clustering_deep"
    results_folder.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Carregar tabelas
    # ------------------------------------------------------------------
    if not deep_image_path.exists():
        raise FileNotFoundError(
            f"Deep image features file not found: {deep_image_path}"
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"Image–patient metadata file not found: {meta_path}")
    if not cluster_patient_path.exists():
        raise FileNotFoundError(
            f"Patient deep clustering file not found: {cluster_patient_path}"
        )

    print(f"[INFO] Loading deep features (image-level) from: {deep_image_path}")
    df_deep = pd.read_excel(deep_image_path)
    print(f"[INFO] Deep feature table size (images x cols): {df_deep.shape}")

    print(f"[INFO] Loading image–patient metadata from: {meta_path}")
    df_meta = pd.read_excel(meta_path)
    print(f"[INFO] Metadata table size (rows x cols): {df_meta.shape}")

    print(f"[INFO] Loading patient deep clusters from: {cluster_patient_path}")
    df_pat = pd.read_excel(cluster_patient_path)
    print(f"[INFO] Patient cluster table size (rows x cols): {df_pat.shape}")

    # ------------------------------------------------------------------
    # 2) Juntar: deep features + metadata + cluster_deep
    # ------------------------------------------------------------------
    # Detectar colunas de ID em comum (deve ser filename e relative_path)
    id_cols = [
        c
        for c in ["filename", "relative_path"]
        if c in df_deep.columns and c in df_meta.columns
    ]
    if not id_cols:
        raise ValueError(
            "No common ID columns between deep table and metadata (expected: 'filename' and/or 'relative_path')."
        )

    print(f"[INFO] ID columns used for merge (deep + meta): {id_cols}")

    # Se df_deep já tiver coluna de paciente, removemos para evitar patient_id_x / patient_id_y
    patient_col_candidates_meta = [
        c for c in df_meta.columns if "patient" in c.lower() and "id" in c.lower()
    ]
    if not patient_col_candidates_meta:
        raise ValueError("No patient_id-like column found in metadata table.")
    patient_col = patient_col_candidates_meta[0]
    print(f"[INFO] Patient ID column in metadata: {patient_col}")

    if patient_col in df_deep.columns:
        print(
            f"[INFO] Dropping patient column from deep table to avoid duplicates: {patient_col}"
        )
        df_deep = df_deep.drop(columns=[patient_col])

    df_merged = pd.merge(df_deep, df_meta, on=id_cols, how="inner")
    print(f"[INFO] Deep + meta merged table size: {df_merged.shape}")

    # Detectar coluna de cluster_deep na tabela de pacientes
    cluster_cols = [c for c in df_pat.columns if "cluster" in c.lower()]
    if not cluster_cols:
        raise ValueError("No cluster column found in patient deep cluster file.")
    cluster_deep_col = cluster_cols[0]
    print(f"[INFO] Cluster column in patient table: {cluster_deep_col}")

    # Pequena tabela só com paciente + cluster
    df_pat_small = df_pat[[patient_col, cluster_deep_col]].copy()
    df_pat_small = df_pat_small.rename(columns={cluster_deep_col: "cluster_deep"})

    # Merge usando o mesmo nome de coluna de paciente que vem do metadata
    df_merged = pd.merge(df_merged, df_pat_small, on=patient_col, how="left")
    print(f"[INFO] Final merged table (image + meta + cluster): {df_merged.shape}")

    # Para facilitar, criamos também uma coluna padronizada 'patient_id' se ainda não existir
    if "patient_id" not in df_merged.columns:
        df_merged = df_merged.rename(columns={patient_col: "patient_id"})
        patient_col_export = "patient_id"
    else:
        patient_col_export = "patient_id"  # já existe
    print(f"[INFO] Patient column used for export: {patient_col_export}")

    if df_merged["cluster_deep"].isna().any():
        n_missing = df_merged["cluster_deep"].isna().sum()
        print(
            f"[WARN] {n_missing} images without cluster_deep (check consistency of patient IDs)."
        )

    # ------------------------------------------------------------------
    # 3) Selecionar colunas de features deep
    # ------------------------------------------------------------------
    numeric_cols = df_merged.select_dtypes(include=["number"]).columns.tolist()

    exclude_cols = [
        patient_col_export,
        "time_in_culture_days",
        "sex",
        "age_years",
        "incident_prevalent",
        "cluster_deep",
    ]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    if len(feature_cols) == 0:
        raise ValueError(
            "No numeric feature columns left after excluding clinical/ID variables."
        )

    print(f"[INFO] Number of deep feature columns used: {len(feature_cols)}")

    # Matriz de features
    X = df_merged[feature_cols].values.astype(float)

    # Standardização global (z-score)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_std = (X - mean) / std

    df_merged_features = df_merged.copy()
    df_merged_features["_feat_array"] = list(X_std)

    # ------------------------------------------------------------------
    # 4) Para cada cluster_deep, selecionar k imagens mais próximas do centróide
    # ------------------------------------------------------------------
    representative_rows = []

    clusters = sorted(df_merged_features["cluster_deep"].dropna().unique())
    print(f"[INFO] Deep clusters found: {clusters}")

    for cl in clusters:
        df_cl = df_merged_features[df_merged_features["cluster_deep"] == cl].copy()
        if df_cl.empty:
            continue

        feats = np.stack(
            df_cl["_feat_array"].values, axis=0
        )  # (n_images_cluster, n_features)
        centroid = feats.mean(axis=0, keepdims=True)  # (1, n_features)

        # Distância euclidiana ao centróide
        dists = np.linalg.norm(feats - centroid, axis=1)
        df_cl["distance_to_centroid"] = dists

        # Ordenar e pegar as k mais representativas
        df_cl_sorted = df_cl.sort_values(by="distance_to_centroid", ascending=True)
        df_topk = df_cl_sorted.head(k_per_cluster).copy()
        df_topk["cluster_deep"] = cl

        representative_rows.append(df_topk)

        print(f"[INFO] Cluster {cl}: selected {len(df_topk)} representative images.")

    if not representative_rows:
        print(
            "[ERROR] No representative rows were selected. Check cluster_deep assignments."
        )
        return

    df_representatives = pd.concat(representative_rows, ignore_index=True)

    # ------------------------------------------------------------------
    # 5) Selecionar colunas para exportar (ids + meta + cluster + distância)
    # ------------------------------------------------------------------
    cols_for_export = []

    # Ordem de interesse
    for col in [
        patient_col_export,
        "cluster_deep",
        "time_in_culture_days",
        "sex",
        "age_years",
        "incident_prevalent",
        "filename",
        "relative_path",
        "well",
        "zone",
        "mode_code",
        "mode_label",
        "distance_to_centroid",
    ]:
        if col in df_representatives.columns and col not in cols_for_export:
            cols_for_export.append(col)

    df_export = df_representatives[cols_for_export].sort_values(
        by=["cluster_deep", "distance_to_centroid", patient_col_export]
    )

    # ------------------------------------------------------------------
    # 6) Salvar Excel com as imagens representativas
    # ------------------------------------------------------------------
    out_path = results_folder / "cluster_deep_representative_images.xlsx"
    df_export.to_excel(out_path, index=False)
    print(f"[INFO] Saved representative images table to: {out_path}")

    print("======================================================================")
    print("SELECT REPRESENTATIVE IMAGES PER DEEP CLUSTER – DONE")
    print("======================================================================")


if __name__ == "__main__":
    main()
