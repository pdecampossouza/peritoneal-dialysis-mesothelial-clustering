#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GRAD-CAM ON REPRESENTATIVE IMAGES (ResNet-50)

- Lê a tabela de imagens representativas por cluster profundo
- Aplica Grad-CAM usando a última camada convolucional da ResNet-50
- Salva as imagens overlay (imagem original + mapa de calor)
- Gera uma figura de legenda da colormap usada no Grad-CAM
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import models, transforms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2


# ----------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# ----------------------------------------------------------------------
PROJECT_ROOT = r"C:\Users\acer\Desktop\fnn-pso\Paper Diálise"

# Tabela de imagens representativas (já gerada pelo select_representative_images_deep.py)
REP_FILE = os.path.join(
    PROJECT_ROOT,
    "results_clustering_deep",
    "cluster_deep_representative_images.xlsx",
)

# Pasta com as imagens originais (usando os caminhos relativos da planilha)
IMAGE_ROOT = PROJECT_ROOT

# Pasta de saída para explicações Grad-CAM
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results_clustering_deep", "explanations")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# FUNÇÕES DE PRÉ-PROCESSAMENTO E PÓS-PROCESSAMENTO
# ----------------------------------------------------------------------
def load_image_for_model(path, device):
    """
    Carrega a imagem do disco e aplica o pré-processamento padrão
    da ResNet-50 (ImageNet).
    Retorna:
      - tensor [1,3,H,W] no device
      - imagem PIL RGB original (para overlay)
    """
    img = Image.open(path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # padrão ImageNet
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    tensor = transform(img).unsqueeze(0).to(device)
    return tensor, img


def apply_colormap_on_image(img_pil, cam_norm, alpha=0.4):
    """
    Aplica o mapa de calor Grad-CAM (cam_norm em [0,1]) sobre a imagem original.
    - img_pil: PIL.Image RGB
    - cam_norm: numpy array [H, W] em [0, 1]
    Retorna: imagem RGB (numpy) com overlay.
    """
    img = np.array(img_pil)

    # Redimensiona o CAM para o tamanho da imagem
    cam_resized = cv2.resize(cam_norm, (img.shape[1], img.shape[0]))

    # Converte para 0-255 uint8
    heatmap = np.uint8(255 * cam_resized)

    # Aplica colormap JET (azul -> vermelho)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Normaliza imagem original para float [0,1]
    img_float = img.astype(np.float32) / 255.0
    heatmap_float = heatmap_color.astype(np.float32) / 255.0

    overlay = (1 - alpha) * img_float + alpha * heatmap_float
    overlay = np.clip(overlay, 0, 1)

    overlay_uint8 = np.uint8(255 * overlay)
    return overlay_uint8


# ----------------------------------------------------------------------
# CLASSE GRAD-CAM
# ----------------------------------------------------------------------
class GradCAMResNet50(object):
    """
    Implementação simples de Grad-CAM para ResNet-50.
    Usa a saída de layer4 como mapa de ativação.
    """

    def __init__(self, device):
        self.device = device

        # Carrega modelo pré-treinado ImageNet
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(self.device)

        # Última camada convolucional (barriga da rede)
        self.target_layer = self.model.layer4

        # Armazenar ativação e gradiente
        self.activations = None
        self.gradients = None

        # Registra hooks
        def forward_hook(module, input, output):
            # output: [B, C, H, W]
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0]: [B, C, H, W]
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        # hook completo para evitar o warning de hook parcial
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor):
        """
        Gera o Grad-CAM para um tensor de entrada [1,3,224,224].
        Usa o logit da classe de maior probabilidade como alvo.
        Retorna um CAM normalizado em [0,1] como numpy [H,W].
        """

        # Forward
        self.activations = None
        self.gradients = None

        output = self.model(input_tensor)  # [1, 1000]

        # Classe alvo: argmax da predição
        target_class = output.argmax(dim=1).item()

        # Zera gradientes
        self.model.zero_grad()

        # Backward a partir do logit da classe alvo
        loss = output[0, target_class]
        loss.backward()

        # Ativações e gradientes: [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        gradients = self.gradients  # [1, C, H, W]

        if activations is None or gradients is None:
            raise RuntimeError("Ativações ou gradientes não capturados pelos hooks.")

        # Média global do gradiente em H e W -> pesos para cada canal
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Combinação linear: soma_c (w_c * A_c)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]

        # ReLU: só ativações positivas
        cam = F.relu(cam)

        # Remove dimensões extras
        cam = cam[0, 0].cpu().numpy()  # [H, W]

        # Normaliza para [0,1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam_norm = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam)

        return cam_norm


# ----------------------------------------------------------------------
# FUNÇÃO PARA GERAR LEGENDA DA COLORMAP DO GRAD-CAM
# ----------------------------------------------------------------------
def save_gradcam_colormap_legend(output_path, cmap_name="jet"):
    """
    Gera uma figura contendo apenas a barra de cores (colorbar)
    usada no Grad-CAM, com rótulos 'Low relevance' e 'High relevance'.

    - output_path: caminho do arquivo PNG a ser salvo.
    - cmap_name: nome da colormap (deve ser o mesmo usado no overlay).
    """
    fig, ax = plt.subplots(figsize=(4, 1.0))
    fig.subplots_adjust(bottom=0.5)

    # Cria um gradiente 2D [1, 256]
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # Mostra o gradiente
    ax.imshow(gradient, aspect="auto", cmap=cmap_name)
    ax.set_axis_off()

    # Adiciona uma colorbar embaixo
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_name),
        ax=ax,
        orientation="horizontal",
        fraction=0.8,
        pad=0.2,
    )
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["Low relevance", "High relevance"])

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("GRAD-CAM ON REPRESENTATIVE IMAGES – START")
    print("=" * 70)

    print(f"[INFO] Loaded representative file: {REP_FILE}")
    print(f"[INFO] Saving explanations to: {OUTPUT_DIR}")

    df = pd.read_excel(REP_FILE)

    # Espera-se que a tabela tenha colunas:
    #   - 'cluster_deep' (cluster)
    #   - 'filename'
    #   - 'relative_path'
    #   - (opcional) 'patient_id' etc.
    if "cluster_deep" not in df.columns:
        raise ValueError(
            "Coluna 'cluster_deep' não encontrada na tabela de representantes."
        )
    if "filename" not in df.columns or "relative_path" not in df.columns:
        raise ValueError("Colunas 'filename' e 'relative_path' são obrigatórias.")

    clusters = sorted(df["cluster_deep"].unique())
    print(f"[INFO] Deep clusters found: {clusters}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    gradcam = GradCAMResNet50(device=device)

    # Loop sobre cada linha (imagem representativa)
    for idx, row in df.iterrows():
        cluster_id = int(row["cluster_deep"])
        filename = str(row["filename"])
        rel_path = str(row["relative_path"])

        img_path = os.path.join(IMAGE_ROOT, rel_path)

        if not os.path.isfile(img_path):
            print(f"[WARN] Image file not found, skipping: {img_path}")
            continue

        try:
            # Carrega imagem e prepara tensor
            input_tensor, img_pil = load_image_for_model(img_path, device)

            # Gera Grad-CAM
            cam_norm = gradcam.generate_cam(input_tensor)

            # Cria overlay
            overlay = apply_colormap_on_image(img_pil, cam_norm, alpha=0.4)

            # Nome do arquivo de saída
            base_name = os.path.splitext(filename)[0]
            out_name = f"cluster{cluster_id}_{base_name}_overlay.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            # Salva usando PIL
            Image.fromarray(overlay).save(out_path)

            print(f"[INFO] Saved CAM for: cluster {cluster_id} — {filename}")

        except Exception as e:
            print(f"[ERROR] Failed Grad-CAM for image {img_path}: {e}")

    # Gera figura de legenda da colormap do Grad-CAM
    legend_path = os.path.join(OUTPUT_DIR, "gradcam_color_legend.png")
    save_gradcam_colormap_legend(legend_path, cmap_name="jet")
    print(f"[INFO] Saved Grad-CAM colormap legend to: {legend_path}")

    print("=" * 70)
    print("GRAD-CAM ON REPRESENTATIVE IMAGES – DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
