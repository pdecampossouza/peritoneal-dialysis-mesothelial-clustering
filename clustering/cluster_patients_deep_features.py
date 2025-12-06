# cluster_patients_deep_features.py

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ==============================================================
# CONFIGURAÇÕES
# ==============================================================

PROJECT_ROOT = r"C:\Users\acer\Desktop\fnn-pso\Paper Diálise"

INPUT_PATIENT_DEEP = os.path.join(
    PROJECT_ROOT, "results_deep_aggregation", "patient_deep_features_resnet50.xlsx"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_clustering_deep")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_CLUSTERED_TABLE = os.path.join(
    RESULTS_DIR, "patient_deep_features_resnet50_clusters.xlsx"
)

PCA_SCATTER_FIG = os.path.join(RESULTS_DIR, "patient_pca_clusters_deep.png")

DENDRO_FIG = os.path.join(RESULTS_DIR, "patient_dendrogram_deep.png")

N_COMPONENTS_MAX = 5  # PCs a calcular
N_CLUSTERS = 3  # igual à morfologia, para comparação

# ==============================================================
# PIPELINE
# ==============================================================

print("=" * 70)
print("CLUSTER PATIENTS WITH DEEP FEATURES (ResNet-50) – START")
print("=" * 70)
print(f"[INFO] Project root folder:      {PROJECT_ROOT}")
print(f"[INFO] Input patient deep file:  {INPUT_PATIENT_DEEP}")
print(f"[INFO] Results folder:           {RESULTS_DIR}")

# --------------------------------------------------------------
# 1) Carregar tabela de features por paciente
# --------------------------------------------------------------
df = pd.read_excel(INPUT_PATIENT_DEEP)
print(f"[INFO] Table size (patients x cols): {df.shape}")

if "patient_id" not in df.columns:
    raise ValueError(
        "A coluna 'patient_id' não foi encontrada na tabela de deep features por paciente."
    )

id_col = "patient_id"

# Todas as colunas numéricas exceto o identificador
numeric_cols = [
    c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])
]

print(f"[INFO] Number of numeric deep feature columns: {len(numeric_cols)}")

X = df[numeric_cols].values
n_patients, n_features = X.shape
print(f"[INFO] Shape of feature matrix: (patients={n_patients}, features={n_features})")

# --------------------------------------------------------------
# 2) Standardização (z-score)
# --------------------------------------------------------------
print("[STEP] Standardizing features (z-score)...")
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# --------------------------------------------------------------
# 3) PCA
# --------------------------------------------------------------
n_components = min(N_COMPONENTS_MAX, n_patients - 1)
print(f"[STEP] Running PCA with up to {n_components} components...")
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_std)

print("[INFO] PCA explained variance ratio:")
for i, var in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"   PC{i}: {var:.3f}")

# --------------------------------------------------------------
# 4) Clustering hierárquico (Ward) nas primeiras PCs
# --------------------------------------------------------------
n_pc_for_clust = min(3, n_components)
print(f"[STEP] Hierarchical clustering (Ward) on first {n_pc_for_clust} PC(s)...")

Z = linkage(X_pca[:, :n_pc_for_clust], method="ward")

clusters = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")
df["cluster_deep"] = clusters

print(f"[INFO] Assigned clusters (1..{N_CLUSTERS}) to {n_patients} patients.")
print("[INFO] Cluster counts:")
print(df["cluster_deep"].value_counts().sort_index())

# --------------------------------------------------------------
# 5) Salvar tabela com clusters
# --------------------------------------------------------------
df.to_excel(OUTPUT_CLUSTERED_TABLE, index=False)
print(f"[INFO] Saved clustered patient deep feature table to: {OUTPUT_CLUSTERED_TABLE}")

# --------------------------------------------------------------
# 6) Gráfico PCA (PC1 x PC2)
# --------------------------------------------------------------
print("[STEP] Generating PCA scatter plot...")
plt.figure(figsize=(6, 5))
for cl in sorted(np.unique(clusters)):
    idx = clusters == cl
    plt.scatter(
        X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {cl}", s=80, edgecolor="k"
    )

for i, pid in enumerate(df[id_col].values):
    plt.text(X_pca[i, 0] + 0.02, X_pca[i, 1] + 0.02, str(pid), fontsize=8)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Patients – Deep Features (ResNet-50) – PCA + Clusters")
plt.legend()
plt.tight_layout()
plt.savefig(PCA_SCATTER_FIG, dpi=300)
plt.close()
print(f"[INFO] Saved PCA scatter plot to: {PCA_SCATTER_FIG}")

# --------------------------------------------------------------
# 7) Dendrograma
# --------------------------------------------------------------
print("[STEP] Generating dendrogram...")
plt.figure(figsize=(6, 4))
dendrogram(
    Z,
    labels=[str(pid) for pid in df[id_col].values],
    orientation="top",
)
plt.title("Patients – Deep Features (ResNet-50) – Dendrogram (Ward)")
plt.xlabel("Patient")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(DENDRO_FIG, dpi=300)
plt.close()
print(f"[INFO] Saved dendrogram to: {DENDRO_FIG}")

print("=" * 70)
print("CLUSTER PATIENTS WITH DEEP FEATURES – DONE")
print("=" * 70)
print(f"[SUMMARY] Input file:              {INPUT_PATIENT_DEEP}")
print(f"[SUMMARY] Number of patients:      {n_patients}")
print(f"[SUMMARY] Number of deep features: {n_features}")
print(f"[SUMMARY] PCA components used:     {n_components}")
print(f"[SUMMARY] Clusters (maxclust):     {N_CLUSTERS}")
print(f"[SUMMARY] Output clustered table:  {OUTPUT_CLUSTERED_TABLE}")
print(f"[SUMMARY] PCA plot:                {PCA_SCATTER_FIG}")
print(f"[SUMMARY] Dendrogram:              {DENDRO_FIG}")
print("=" * 70)
