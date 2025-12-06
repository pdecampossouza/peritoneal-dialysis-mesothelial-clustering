"""
extract_deep_image_features.py

Script to extract deep convolutional features from each image using a pretrained
ResNet-50 (ImageNet) model. For each image in 'data_image_with_patient.xlsx',
we compute a 2048-dimensional embedding from the penultimate layer and save
the result in an Excel file.

The output table has one row per image and includes:
    - patient_id
    - filename
    - relative_path
    - mode_code
    - well
    - zone
    - time_in_culture_days
    - sex
    - age_years
    - incident_prevalent
    - deep_feat_000 ... deep_feat_2047
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms


def get_device():
    """
    Select computation device. Prefer GPU if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU (no CUDA GPU available).")
    return device


def build_model(device):
    """
    Build a ResNet-50 model pretrained on ImageNet and return a feature
    extractor that outputs a 2048-dimensional embedding per image.
    """
    print("[STEP] Loading pretrained ResNet-50 (ImageNet)...")
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
    except AttributeError:
        backbone = models.resnet50(pretrained=True)

    backbone.eval()

    # Remove final classification layer
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

    feature_extractor.to(device)
    for p in feature_extractor.parameters():
        p.requires_grad = False

    print("[INFO] Feature extractor output dimension: 2048")
    return feature_extractor


def build_transform():
    """
    Image preprocessing consistent with ImageNet-trained ResNet-50.
    """
    print("[STEP] Building image transform pipeline...")
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform


def load_image(path, transform):
    """
    Load an image from disk and apply the preprocessing transform.
    Always convert to RGB to be compatible with ImageNet models.
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_t = transform(img)
    return img_t


def extract_features_for_images(df, root, model, transform, device, batch_size=8):
    """
    Iterate over the rows of df, load each image, and compute deep features.

    Expects at least:
        - patient_id
        - filename
        - relative_path
        - mode_code
        - well
        - zone
        - time_in_culture_days
        - sex
        - age_years
        - incident_prevalent
    """
    n = df.shape[0]
    print(f"[INFO] Number of images to process: {n}")

    features_list = []

    start_idx = 0
    while start_idx < n:
        end_idx = min(start_idx + batch_size, n)
        batch_df = df.iloc[start_idx:end_idx]

        batch_tensors = []
        meta_batch = []

        for _, row in batch_df.iterrows():
            img_rel = row["relative_path"]
            img_path = root / img_rel

            if not img_path.exists():
                print(f"[WARNING] Image not found, skipping: {img_path}")
                continue

            try:
                img_t = load_image(img_path, transform)
            except Exception as e:
                print(f"[WARNING] Failed to load {img_path}: {e}")
                continue

            batch_tensors.append(img_t)

            meta_batch.append(
                {
                    "patient_id": row.get("patient_id", None),
                    "filename": row.get("filename", None),
                    "relative_path": img_rel,
                    "mode_code": row.get("mode_code", None),
                    "mode_label": row.get("mode_label", None),
                    "well": row.get("well", None),
                    "zone": row.get("zone", None),
                    "time_in_culture_days": row.get("time_in_culture_days", None),
                    "sex": row.get("sex", None),
                    "age_years": row.get("age_years", None),
                    "incident_prevalent": row.get("incident_prevalent", None),
                }
            )

        if not batch_tensors:
            start_idx = end_idx
            continue

        batch_tensor = torch.stack(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            feats = model(batch_tensor)
            feats = feats.view(feats.size(0), -1)  # (B, 2048)

        feats_np = feats.cpu().numpy()

        for meta, vec in zip(meta_batch, feats_np):
            row_dict = dict(meta)
            for i, val in enumerate(vec):
                row_dict[f"deep_feat_{i:03d}"] = float(val)
            features_list.append(row_dict)

        start_idx = end_idx
        print(
            f"[INFO] Processed images {start_idx:4d} / {n:4d} "
            f"({100.0 * start_idx / n:5.1f}%)"
        )

    return features_list


def main():
    print("=" * 70)
    print("EXTRACT DEEP IMAGE FEATURES (ResNet-50) – START")
    print("=" * 70)

    root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {root}")

    # Ajuste aqui se o nome do ficheiro for diferente
    path_merged = root / "data_image_with_patient.xlsx"
    if not path_merged.exists():
        print(f"[ERROR] File not found: {path_merged}")
        print("        Check the image–patient table filename.")
        return

    print(f"[INFO] Loading image table: {path_merged.name}")
    df = pd.read_excel(path_merged)

    required_cols = [
        "patient_id",
        "filename",
        "relative_path",
        "mode_code",
        "mode_label",
        "well",
        "zone",
        "time_in_culture_days",
        "sex",
        "age_years",
        "incident_prevalent",
    ]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Required column missing in table: '{col}'")
            return

    n_total = df.shape[0]
    print(f"[INFO] Total number of images in table: {n_total}")

    # Se quiser, podemos filtrar só BW:
    # df_use = df[df["mode_code"] == 2].copy()
    df_use = df.copy()
    n_use = df_use.shape[0]
    print(f"[INFO] Number of images selected for deep features: {n_use}")

    out_dir = root / "results_deep_features"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "image_deep_features_resnet50.xlsx"
    print(f"[INFO] Results will be saved in: {out_dir}")

    device = get_device()
    model = build_model(device)
    transform = build_transform()

    print("[STEP] Extracting deep features for all images...")
    features_list = extract_features_for_images(
        df_use,
        root=root,
        model=model,
        transform=transform,
        device=device,
        batch_size=8,
    )

    if not features_list:
        print("[ERROR] No features were extracted. Check warnings above.")
        return

    df_feats = pd.DataFrame(features_list)
    print(f"[INFO] Deep feature table size: {df_feats.shape}")

    df_feats.to_excel(out_path, index=False)
    print(f"[INFO] Saved deep features to: {out_path}")

    print("=" * 70)
    print("EXTRACT DEEP IMAGE FEATURES (ResNet-50) – DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
