# 🧬 Peritoneal Dialysis – Mesothelial Cell Imaging & Unsupervised Patient Clustering

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Made with ❤️ by NOVA IMS](https://img.shields.io/badge/Made%20with-%F0%9F%92%96%20by%20NOVA%20IMS-orange)](https://novaims.unl.pt)

---

## 🧭 Overview

This repository contains the **full pipeline**, source code, metadata, and documentation used in the analysis of **mesothelial cell images from peritoneal dialysis (PD) patients**.

It supports the complete workflow used in the experiments, covering:

- dataset structuring  
- image preprocessing  
- morphological, intensity, texture, and deep-feature extraction  
- patient-level aggregation  
- PCA & hierarchical clustering  
- representative image selection  
- Grad-CAM interpretability  
- LaTeX-ready tables and figures  

The goal is to ensure **full reproducibility**, transparency, and scientific rigor.

---

## 📁 Repository Structure

```
peritoneal-dialysis-mesothelial-clustering/
│
├── data/
│   ├── raw/                    # Raw microscopy images (NOT uploaded here)
│   ├── intermediate/           # Extracted features by image/patient
│   └── results/                # PCA projections, dendrograms, representative panels
│
├── src/
│   ├── preprocessing/          # Preprocessing + feature extraction
│   ├── aggregation/            # Patient-level aggregation scripts
│   ├── clustering/             # PCA + hierarchical clustering
│   ├── visualization/          # Representative images, Grad-CAM
│   └── latex/                  # Export tables & figures for the text
│   └── figures/                # Figures generated in the experiments
│
├── README.md
├── LICENSE
└── .gitignore
```
## 📁 Dataset Structure

The raw microscopy dataset follows a strict hierarchical organization defined by the laboratory workflow.  
Each patient corresponds to one cultured sample, and each sample is imaged in **five wells** and **five spatial zones** (A–E), in **two acquisition modes** (RGB and BW).

The filesystem structure is:

```
HPMCs_DialisePeritoneal/
│
├── 1056/                        # Patient ID
│   ├── Poço 1/
│   │   ├── 1.A.1.jpg            # Well 1 – Zone A – Mode 1 (RGB)
│   │   ├── 1.A.2.jpg            # Well 1 – Zone A – Mode 2 (BW)
│   │   ├── 1.B.1.jpg
│   │   ├── 1.B.2.jpg
│   │   ├── 1.C.1.jpg
│   │   ├── 1.C.2.jpg
│   │   ├── 1.D.1.jpg
│   │   ├── 1.D.2.jpg
│   │   ├── 1.E.1.jpg
│   │   ├── 1.E.2.jpg
│   │   ...
│   ├── Poço 2/
│   ├── Poço 3/
│   ├── Poço 4/
│   └── Poço 5/
│
├── 1059/
├── 1060/
├── 1062/
├── 1065/
├── 1066/
├── 1067/
├── 1068/
└── 1069/
```

### ✔ Meaning of the naming convention

Each filename follows the laboratory’s acquisition protocol:

```
{well}.{zone}.{mode}.jpg
```

Where:

- **well** ∈ {1, 2, 3, 4, 5}  
- **zone** ∈ {A, B, C, D, E}  
- **mode**  
  - **1 → RGB** (colour image of the same field)  
  - **2 → BW** (high-contrast grayscale for cell-body and nuclear visibility)

Thus, each well contributes:

- **5 Zones × 2 Modes = 10 images per well**
- **5 Wells × 10 = 50 images per patient**

All analyses implemented in the project assume and validate this structure.

---

## 🔍 How This Structure Is Loaded in the Pipeline

Every feature-extraction script relies on this organization.  
All scripts begin by scanning the directory recursively and parsing the identifiers directly from filenames:

```
patient_id / well / zone / mode
```

For example, in:

```
1060/Poço 3/3.C.2.jpg
```

We extract:

| Component     | Meaning                     |
|--------------|-----------------------------|
| 1060         | Patient                     |
| Poço 3       | Well                        |
| 3            | Well ID (redundancy check)  |
| C            | Spatial zone                |
| 2            | BW acquisition              |

This guarantees reproducibility and allows your code to:

- validate file structure  
- detect missing images  
- join with clinical metadata  
- aggregate by well, zone, and patient  
- compute patient-level summaries  
- cluster in meaningful hierarchical units  

---

## 📌 Why This Structure Must Stay Private

The images belong to real patients and contain identifiable biological material.  
Thus:

- **Raw images cannot be committed to GitHub**  
- **Repository must be PRIVATE**  
- Only extracted numerical features are uploaded (safe & anonymised)
---

## ⚙️ Installation

```bash
conda create -n pdmesothelial python=3.10
conda activate pdmesothelial
pip install -r requirements.txt
```

---

## ▶️ Pipeline Execution

### **1. Build dataset index**

```bash
python src/preprocessing/build_data_image.py
python src/preprocessing/prepare_dataset_overview.py
```

### **2. Extract features**

```bash
python src/preprocessing/extract_basic_image_features.py
python src/preprocessing/extract_morphology_features.py
python src/preprocessing/extract_deep_image_features.py
```

### **3. Aggregate patient-level descriptors**

```bash
python src/aggregation/aggregate_features_by_patient.py
python src/aggregation/aggregate_morphology_features.py
python src/aggregation/aggregate_deep_features_by_patient.py
```

### **4. Perform clustering**

```bash
python src/clustering/cluster_patients_pca.py
python src/clustering/cluster_patients_with_morphology.py
python src/clustering/cluster_patients_deep_features.py
python src/clustering/cluster_pca_all_features.py
python src/clustering/analyze_cluster_feature_differences.py
python src/clustering/cluster_clinical_stats.py
python src/clustering/summarize_clusters_with_morphology.py
python src/clustering/summarize_patient_clusters.py
```

### **5. Generate representative panels & Grad-CAM**

```bash
python src/visualization/select_representative_images_by_cluster.py
python src/visualization/panel_representative_images.py
python src/visualization/gradcam_representative_images.py
python src/visualization/panel_gradcam_representative_images.py
python src/visualization/select_representative_images_deep.py
```

### **6. Export LaTeX tables**

```bash
python src/latex/generate_latex_outputs.py
python src/latex/generate_latex_stats.py
python src/latex/generate_figures_latex_data.py
```

---

## 🧠 Features Extracted

### **Basic Image Features (Intensity + Edges)**
- mean, std, min, max intensity  
- Sobel edge magnitude  

### **Morphology Features**
- cell count & density  
- mean & std of:
  - area  
  - eccentricity  
  - aspect ratio  
  - solidity  

### **GLCM Texture**
- contrast  
- homogeneity  
- energy  
- correlation  

### **Deep Learning (CNN / ResNet-50)**
- 2048-d feature vector per image  
- 4096D patient vector (mean + std)

---

## 📊 Outputs Included

- PCA plots (all feature sets)
- clustering dendrograms
- representative image panels
- Grad-CAM maps
- aggregated tables (`.tex`) used in the dissertation
- 
## 🧬 PCA Clustering (Image-only features)

<img src="figures/patient_pca_clusters_morphology.png" width="650">

## 🌳 Dendrogram (Ward Linkage)

<img src="figures/patient_dendrogram_morphology.png" width="750">

## 📸 Representative Images

<img src="figures/panel_representative_images.png" width="850">

## 🔥 Grad-CAM Interpretability

<img src="figures/panel_gradcam_representative_images.png" width="850">


---

## 🔒 Privacy Notice

This repository is intended to be **private**, as raw human cell images cannot be publicly distributed.  
Only processed features (without identifiable content) should be uploaded.

---

## 📘 Citation

```bibtex
@misc{souza2025pdmesothelial,
  author    = {Paulo Vitor de Campos Souza},
  title     = {Peritoneal Dialysis Mesothelial Cell Imaging: Feature Extraction and Unsupervised Patient Clustering},
  year      = {2025},
}
```

---

## 📬 Contact

Paulo Vitor de Campos Souza  
NOVA Information Management School (NOVA IMS)  
Email: psouza@novaims.unl.pt

Luísa Alexandra Teixeira Santos  
NOVA Medical School  
Email: luisa.santos@nms.unl.pt

Sofia Pereira  
NOVA Medical School  
Email: sofia.pereira@nms.unl.pt

---

*“Understanding cellular morphology to uncover patient-level patterns in peritoneal dialysis.”*
