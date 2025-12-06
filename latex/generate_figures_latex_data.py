import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_excel("image_basic_features.xlsx")

# -------------------------
# 1) TABLE PER PATIENT (mean + std) - ONLY NUMERIC COLUMNS
# -------------------------

# Seleciona apenas colunas numéricas (para evitar strings tipo 'AABBCCDDEE')
numeric_cols = df.select_dtypes(include="number").columns

patient_stats = df.groupby("patient_id")[numeric_cols].agg(["mean", "std"])

# export LaTeX
with open("patient_summary_table.tex", "w") as f:
    f.write(patient_stats.to_latex(float_format="%.3f"))

print("Generated patient_summary_table.tex")

# -------------------------
# 2) HEATMAP patient × well
# -------------------------

# Escolha da feature para o heatmap
# Se quiser usar Sobel, troque para: feature = "mean_sobel_edge"
feature = "mean_intensity"

heatmap_data = df.groupby(["patient_id", "well"])[feature].mean().unstack()

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="viridis")
plt.title(f"Heatmap of {feature} by Patient × Well")
plt.xlabel("Well")
plt.ylabel("Patient")

plt.savefig("patient_well_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print("Generated patient_well_heatmap.png")
