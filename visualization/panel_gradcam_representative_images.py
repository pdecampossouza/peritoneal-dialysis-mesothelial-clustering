import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def main():
    # ------------------------------------------------------------------
    # CONFIGURAÇÕES
    # ------------------------------------------------------------------
    project_root = r"C:\Users\acer\Desktop\fnn-pso\Paper Diálise"

    reps_file = os.path.join(
        project_root,
        "results_clustering_deep",
        "cluster_deep_representative_images.xlsx",
    )

    explanations_dir = os.path.join(
        project_root, "results_clustering_deep", "explanations"
    )

    output_panel = os.path.join(
        project_root,
        "results_clustering_deep",
        "panel_gradcam_representative_images.png",
    )

    top_k_per_cluster = 5  # número máximo de imagens por cluster no painel

    print("=" * 70)
    print("BUILD PANEL OF GRAD-CAM REPRESENTATIVE IMAGES – START")
    print("=" * 70)
    print("[INFO] Project root:", project_root)
    print("[INFO] Representative table:", reps_file)
    print("[INFO] Grad-CAM explanations folder:", explanations_dir)
    print("[INFO] Output panel:", output_panel)

    # ------------------------------------------------------------------
    # LEITURA DA TABELA DE IMAGENS REPRESENTANTES
    # ------------------------------------------------------------------
    df = pd.read_excel(reps_file)

    # assumindo que as colunas são algo como:
    # ['patient_id', 'cluster_deep', 'filename', 'relative_path', ...]
    if "cluster_deep" not in df.columns:
        raise ValueError(
            "Column 'cluster_deep' not found in representative images file."
        )

    if "filename" not in df.columns:
        raise ValueError("Column 'filename' not found in representative images file.")

    clusters = sorted(df["cluster_deep"].unique())
    print("[INFO] Clusters found in representative table:", clusters)

    n_clusters = len(clusters)
    n_rows = top_k_per_cluster

    print(f"[INFO] Panel shape: rows={n_rows}, cols={n_clusters}")

    # ------------------------------------------------------------------
    # CRIAR FIGURA
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(n_rows, n_clusters, figsize=(4 * n_clusters, 4 * n_rows))

    # Se n_rows == 1 ou n_clusters == 1, axes vira array 1D; normalizar:
    if n_rows == 1 and n_clusters == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_clusters == 1:
        axes = [[ax] for ax in axes]

    # ------------------------------------------------------------------
    # PREENCHER O PAINEL
    # ------------------------------------------------------------------
    for col_idx, c in enumerate(clusters):
        df_c = df[df["cluster_deep"] == c].copy()
        df_c = df_c.head(top_k_per_cluster)

        print(f"[INFO] Cluster {c}: {len(df_c)} representative images used in panel.")

        for row_idx in range(n_rows):
            ax = axes[row_idx][col_idx]
            ax.axis("off")

            if row_idx >= len(df_c):
                # sem imagem para esta posição → deixa em branco
                continue

            row = df_c.iloc[row_idx]
            filename = str(row["filename"])
            patient_id = row.get("patient_id", "NA")

            # remover extensão
            base_name, _ = os.path.splitext(filename)

            # construir nome do arquivo overlay
            overlay_name = f"cluster{c}_{base_name}_overlay.png"
            overlay_path = os.path.join(explanations_dir, overlay_name)

            if not os.path.isfile(overlay_path):
                print("[WARN] Overlay file not found, skipping:", overlay_path)
                continue

            try:
                img = Image.open(overlay_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(
                    f"Cluster {c} – Patient {patient_id}\n{filename}", fontsize=8
                )
            except Exception as e:
                print("[WARN] Could not load overlay image:", overlay_path, "—", e)

    plt.tight_layout()
    fig.savefig(output_panel, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[INFO] Panel saved to:", output_panel)
    print("=" * 70)
    print("BUILD PANEL OF GRAD-CAM REPRESENTATIVE IMAGES – DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
