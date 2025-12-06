# aggregate_deep_features_by_patient.py

import os
import pandas as pd

# ==============================================================
# CONFIGURAÇÕES
# ==============================================================
PROJECT_ROOT = r"C:\Users\acer\Desktop\fnn-pso\Paper Diálise"

DEEP_FEATURES_FILE = os.path.join(
    PROJECT_ROOT, "results_deep_features", "image_deep_features_resnet50.xlsx"
)
DATA_IMAGE_FILE = os.path.join(PROJECT_ROOT, "data_image_with_patient.xlsx")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_deep_aggregation")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_PATIENT_DEEP = os.path.join(RESULTS_DIR, "patient_deep_features_resnet50.xlsx")


def standardize_column_name(col):
    if isinstance(col, str):
        return col.strip()
    return col


print("=" * 70)
print("AGGREGATE DEEP FEATURES (IMAGE → PATIENT) – START")
print("=" * 70)

print(f"[INFO] Project root folder: {PROJECT_ROOT}")
print(f"[INFO] Deep features file: {DEEP_FEATURES_FILE}")
print(f"[INFO] Data-image file:    {DATA_IMAGE_FILE}")
print(f"[INFO] Results folder:     {RESULTS_DIR}")

# --------------------------------------------------------------
# 1) Carrega deep features por imagem
# --------------------------------------------------------------
deep_df = pd.read_excel(DEEP_FEATURES_FILE)
deep_df.columns = [standardize_column_name(c) for c in deep_df.columns]
print(f"[INFO] Deep feature table size: {deep_df.shape}")

id_cols = []
for cand in ["filename", "relative_path", "image_name", "image", "file"]:
    if cand in deep_df.columns:
        id_cols.append(cand)

if not id_cols:
    raise ValueError(
        "Nenhuma coluna de identificação (ex: 'filename', 'relative_path') "
        "foi encontrada na tabela de deep features."
    )

print(f"[INFO] ID columns found in deep features: {id_cols}")

# --------------------------------------------------------------
# 2) Carrega mapa imagem → paciente
# --------------------------------------------------------------
meta_df = pd.read_excel(DATA_IMAGE_FILE)
meta_df.columns = [standardize_column_name(c) for c in meta_df.columns]
print(f"[INFO] Image–patient table size: {meta_df.shape}")

meta_id_cols = [c for c in id_cols if c in meta_df.columns]
if not meta_id_cols:
    raise ValueError(
        "As colunas de identificação usadas em deep_df não existem em "
        "data_image_with_patient.xlsx."
    )

print(f"[INFO] Matching ID columns in metadata: {meta_id_cols}")

# --------------------------------------------------------------
# 3) Merge trazendo patient_id
# --------------------------------------------------------------
merge_cols = meta_id_cols

merged = pd.merge(
    deep_df,
    meta_df[["patient_id"] + merge_cols],
    on=merge_cols,
    how="inner",
    validate="many_to_one",
)

print(f"[INFO] Deep + patient merged table size: {merged.shape}")

# --- NOVO BLOCO: garante coluna 'patient_id' única ----------------
if "patient_id" not in merged.columns:
    candidates = [c for c in merged.columns if c.startswith("patient_id")]
    if not candidates:
        raise ValueError(
            "Após o merge não foi encontrada nenhuma coluna 'patient_id' "
            "nem variantes ('patient_id_x', 'patient_id_y')."
        )
    print(
        f"[WARN] 'patient_id' não encontrado diretamente. Usando coluna {candidates[0]} como referência."
    )
    merged["patient_id"] = merged[candidates[0]]

print("[INFO] Patients present in merged table:", sorted(merged["patient_id"].unique()))
# ------------------------------------------------------------------

# --------------------------------------------------------------
# 4) Agregar features profundas por paciente (média)
# --------------------------------------------------------------
exclude_cols = set(["patient_id"] + merge_cols)
numeric_cols = [
    c
    for c in merged.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(merged[c])
]

print(f"[INFO] Number of numeric deep feature columns: {len(numeric_cols)}")

group = merged.groupby("patient_id")[numeric_cols]
patient_deep_mean = group.mean().reset_index()

print("[INFO] Aggregated patient deep feature table size:", patient_deep_mean.shape)

# --------------------------------------------------------------
# 5) Salvar
# --------------------------------------------------------------
patient_deep_mean.to_excel(OUTPUT_PATIENT_DEEP, index=False)
print(f"[INFO] Saved patient-level deep features to: {OUTPUT_PATIENT_DEEP}")

print("=" * 70)
print("AGGREGATE DEEP FEATURES (IMAGE → PATIENT) – DONE")
print("=" * 70)
