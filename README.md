
# WiDS Datathon 2025 – ADHD & Sex Prediction from fMRI and Metadata

This project was developed for the 2025 [WiDS Datathon]([https://www.widsconference.org/datathon.html](https://www.kaggle.com/competitions/widsdatathon2025/overview)) and for my CS 211 Final Project, focusing on predicting ADHD diagnosis and sex from fMRI functional connectome data and socio-demographic, emotional, and parenting metadata.

## 🔍 Overview

I implemented machine learning models to classify:
- **ADHD diagnosis** (binary)
- **Sex** (binary)

I tried out multiple machine learning models:
- XGBoost
- Fully Connected Neural Networks
- Graph Convolutional Neural Network (on connectomes only)

## 🧠 Data

Data is from the [Healthy Brain Network]([(https://childmind.org/)]) (HBN) dataset, provided by the datathon organizers. It includes:
- Functional brain connectivity matrices (fMRI)
- Socio-demographic and behavioral metadata

```
kaggle competitions download -c widsdatathon2025
```
## ⚙️ Requirements

- Python ≥ 3.9  
- PyTorch, PyTorch Geometric  
- Scikit-learn, XGBoost, NumPy, Pandas  

Install dependencies:
```bash
pip install -r requirements.txt
````

## 🚀 Usage

Run the main training for **Graph Convolutional Neural Network** script:

```bash
python main.py
```


## 📁 Project Structure

```
.
├── data/                 # Processed datasets and splits
├── models/               # GNN model definitions
├── GNN_helpers.py        # Core GCN implementation and training loop
├── utils/                # Preprocessing, metrics, and plotting
├── main.py               # Entry point
└── README.md
```

## ✅ Features

* Easy training and testing for GCN model with class and method definitions
* Includes specialized model GCN and flexible model XGboost for comparisons

## 📌 TODO

* Add MLP layer to GCN to include metadata in the model
* Clean up XGboost pipeline


