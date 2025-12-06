import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    print("=" * 70)
    print("BUILD PANEL OF REPRESENTATIVE IMAGES – START")
    print("=" * 70)

    # Root = pasta do projeto (onde está este script)
    root = Path(__file__).resolve().parent

    # Arquivo com as imagens representativas por cluster (já gerado)
    rep_file = (
        root / "results_clustering_deep" / "cluster_deep_representative_images.xlsx"
    )

    # Saída: figura com painel
    out_fig = root / "results_clustering_deep" / "panel_representative_images.png"

    # Lê a tabela de imagens representativas
    df = pd.read_excel(rep_file)

    # Garante que os rótulos de cluster sejam inteiros "bonitos"
    clusters = sorted(int(c) for c in df["cluster_deep"].unique())
    n_clusters = len(clusters)

    # Definimos até 5 imagens por cluster (linhas do painel)
    max_images_per_cluster = 5
    n_rows = max_images_per_cluster
    n_cols = n_clusters

    print(f"[INFO] Clusters: {clusters}")
    print(f"[INFO] Panel shape: rows={n_rows}, cols={n_cols}")

    # Figura
    plt.figure(figsize=(4 * n_cols, 3.8 * n_rows))

    plot_index = 1
    for col_idx, cl in enumerate(clusters, start=1):
        # Ordena por distância ao centróide (mais representativas primeiro)
        df_cl = (
            df[df["cluster_deep"] == cl]
            .sort_values(by="distance_to_centroid")
            .head(max_images_per_cluster)
        )

        for row_idx, (_, row) in enumerate(df_cl.iterrows(), start=1):
            # relative_path já é relativo à raiz do projeto
            # Ex.: HPMCs_DialisePeritoneal\1056\Poço 1\1.A.1.jpg
            rel_path = str(row["relative_path"])
            img_path = root / rel_path

            try:
                img = mpimg.imread(img_path)
            except Exception as e:
                print(f"[WARN] Could not load image: {img_path} — {e}")
                continue

            # Posição do subplot (varre por colunas de clusters)
            plt.subplot(n_rows, n_cols, plot_index)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Cluster {cl}\n{row['filename']}", fontsize=8)

            plot_index += 1

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()

    print(f"[INFO] Panel saved to: {out_fig}")
    print("=" * 70)
    print("BUILD PANEL OF REPRESENTATIVE IMAGES – DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
