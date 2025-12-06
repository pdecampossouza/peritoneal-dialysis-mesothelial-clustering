# Peritoneal Dialysis Microscopy Pipeline – Project Documentation

This repository contains a complete computational pipeline for organizing,
processing, extracting features, analyzing, and visualizing microscopy images
of peritoneal dialysis (PD) patient–derived cultured cells.

---

## Overview of the Pipeline

The workflow transforms raw microscopy images into interpretable patient-level
profiles using:

1. Data organization and QC  
2. Extraction of handcrafted features (intensity, morphology, texture)  
3. Deep feature extraction with ResNet-50  
4. Patient-level aggregation  
5. PCA for dimensionality reduction  
6. Clustering (handcrafted, multimodal, deep features)  
7. Visualization + LaTeX figure and table generation  
8. Cluster-level biological interpretation  

---

## Repository Structure

The main scripts include:

- Feature extraction  
- Aggregation  
- PCA  
- Clustering  
- Grad-CAM  
- Representative image selection  
- LaTeX output generation  

---

## Data Organization

**prepare_dataset_overview.py**  
Indexes all images and merges metadata. Performs QC checks to ensure dataset completeness.

---

## Handcrafted Feature Extraction

**extract_basic_image_features.py**  
Computes intensity, contrast, Sobel edges, masking.

**extract_morphology_features.py**  
Segment cells using Gaussian smoothing, Otsu thresholding, connected components. Extracts area, eccentricity, solidity, aspect ratio, cell count, density.

**extract_deep_image_features.py**  
Uses ResNet‑50 to extract 2048‑dimensional embeddings and saves them per image.

---

## Aggregation

**aggregate_features_by_patient.py**  
Aggregates intensity and edge features.

**aggregate_morphology_features.py**  
Aggregates morphological features.

**aggregate_deep_features_by_patient.py**  
Aggregates deep features (mean and std of all 2048 dimensions).

---

## Dimensionality Reduction

**cluster_patients_pca.py**  
Runs PCA on handcrafted features and produces PC plots.

**cluster_pca_all_features.py**  
Runs PCA including clinical data.

**cluster_patients_deep_features.py**  
PCA on deep embeddings.

---

## Clustering

**cluster_patients_with_morphology.py**  
Clusters using handcrafted features.

**cluster_patients_deep_features.py**  
Clusters using deep features.

**cluster_clinical_stats.py**  
Computes differences across clusters in clinical variables.

**summarize_patient_clusters.py**  
Generates human‑readable summaries.

---

## Feature Interpretability

**analyze_cluster_feature_differences.py**  
Computes cluster‑wise vs global feature deviation using z‑effect. Produces tables of top discriminatory features.

---

## Visualization

**panel_representative_images.py**  
Extracts representative raw images per cluster.

**panel_gradcam_representative_images.py**  
Applies Grad‑CAM overlays for interpretability.

---

## LaTeX Output Generation

**generate_figures_latex_data.py**  
Plots heatmaps, PCA projections, dendrograms.

**generate_latex_stats.py**  
Formats statistical tables for LaTeX.

**generate_latex_outputs.py**  
Produces complete figure and table bundles.

---

## Pipeline Execution Example

```
python prepare_dataset_overview.py
python extract_basic_image_features.py
python extract_morphology_features.py
python extract_deep_image_features.py
python aggregate_features_by_patient.py
python aggregate_morphology_features.py
python aggregate_deep_features_by_patient.py
python cluster_patients_pca.py
python cluster_patients_with_morphology.py
python cluster_patients_deep_features.py
python analyze_cluster_feature_differences.py
python generate_latex_outputs.py
```

---

## Interpretation Notes for Clinicians

- Clusters reflect **consistent morphological phenotypes**, not disease severity.
- Deep learning identifies **subtle morphological signatures** beyond human visual inspection.
- Z‑effect values highlight the **features that define each cluster**.
- Representative images and Grad‑CAM maps offer **visual explanations**.

---

## Citation

Please cite:
 

---

This README was automatically constructed from the full pipeline contents.
