# WiDsDatathon
````markdown
# WiDS Datathon 2025 – ADHD & Sex Prediction from fMRI and Metadata

This project was developed for the 2025 [WiDS Datathon](https://www.widsconference.org/datathon.html), focusing on predicting ADHD diagnosis and sex from fMRI functional connectome data and socio-demographic, emotional, and parenting metadata.

## 🔍 Overview

We implement and compare machine learning models to classify:
- **ADHD diagnosis** (binary)
- **Sex** (binary)

The core model is a **Graph Neural Network (GNN)** that integrates fMRI-derived functional connectomes with tabular metadata. We also benchmark other models including:
- XGBoost
- Fully Connected Neural Networks
- BrainNetCNN (on connectomes only)
- Connectome-CNN

## 🧠 Data

Data is from the [Healthy Brain Network](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/) (HBN) dataset, provided by the datathon organizers. It includes:
- Functional brain connectivity matrices (fMRI)
- Socio-demographic and behavioral metadata

## ⚙️ Requirements

- Python ≥ 3.9  
- PyTorch, PyTorch Geometric  
- Scikit-learn, XGBoost, NumPy, Pandas  
- (Optional) BrainGB for GNN baselines  

Install dependencies:
```bash
pip install -r requirements.txt
````

## 🚀 Usage

Run the main training script:

```bash
python main.py
```

You can adjust architecture and data options in `config.yaml` or within each model's script.

## 📁 Project Structure

```
.
├── data/                 # Processed datasets and splits
├── models/               # GNN, CNN, and XGBoost model definitions
├── GNN_helpers.py        # Core GCN implementation and training loop
├── utils/                # Preprocessing, metrics, and plotting
├── main.py               # Entry point
└── README.md
```

## ✅ Features

* Integrates tabular and graph data
* Modular design for easy model swapping
* Supports training/validation/testing splits with reproducibility

## 📌 TODO

* Hyperparameter tuning
* Model ensembling
* Interpretation (e.g., saliency maps, SHAP for metadata)


```
```

