# Spectral-Preconditioning-and-Adaptive-Head-Selection-1

# Spectral Preconditioning and Adaptive Head Selection for Text Classification

This repository contains the reference implementation for the paper

> **Spectral Preconditioning and Adaptive Head Selection for Text Classification**  
> Tirthankar Ghosh, Debashis Chatterjee (Visva–Bharati University)

The code implements a **spectral–first, adaptive–head** pipeline for text classification, and compares it against a more conventional **Deep SVD + Conv1D CNN** baseline. Both the **simulation study** and the **real–data experiment on 20 Newsgroups** are implemented in a single Python script.

---

## 1. Overview

Given a sparse TF–IDF term–document matrix, the pipeline:

1. Applies **truncated SVD** to obtain low–rank spectral document embeddings.
2. Trains and cross–validates multiple classifier heads on the same embeddings:
   - Multinomial logistic regression  
   - Linear SVM  
   - Random Forest  
3. Selects the **best (rank, head)** combination via macro–F1 on validation folds.
4. Compares this against a fixed **Conv1D CNN** head on the same SVD features.
5. Saves all **metrics, plots, and tables** to disk and archives them as ZIP files.

Two experiments are run:

- **Synthetic simulation** under a Dirichlet–multinomial topic model.
- **Real data** using the classic **20 Newsgroups** dataset from `scikit-learn`.

---

## 2. Repository Structure

After running the script once, the repository will look roughly like:

```text
.
├── spectral_adaptive_text_experiments.ipynb   # main script (sim + real data)
├── README.md
├── simulation_results_spectral_adaptive/
│   ├── baseline_deep_svd_cnn_training_curves.png
│   ├── baseline_deep_svd_cnn_confusion_matrix.png
│   ├── proposed_best_head_confusion_matrix.png
│   ├── comparison_barplot_accuracy_macroF1.png
│   ├── cv_results_spectral_adaptive.csv
│   ├── summary_comparison_baseline_vs_proposed.csv
│   └── simulation_results_spectral_first_vs_baseline.zip
└── realdata_20newsgroups_results/
    ├── class_distribution_train_test.png
    ├── svd_explained_variance.png
    ├── svd_2d_embedding_sample.png
    ├── baseline_deep_svd_cnn_training_curves.png
    ├── baseline_deep_svd_cnn_confusion_matrix.png
    ├── proposed_best_head_realdata_confusion_matrix.png
    ├── comparison_realdata_barplot_accuracy_macroF1.png
    ├── cv_results_realdata_spectral_adaptive.csv
    ├── summary_realdata_baseline_vs_proposed.csv
    └── realdata_20newsgroups_results.zip
