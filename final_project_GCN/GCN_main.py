import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, global_mean_pool
from GCN_helpers import *

# === Load data ===
data_path = "/Users/Haley/Desktop/WiDs Datathon/widsdatathon2025/"
connectome_train = pd.read_csv(f"{data_path}TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
targets_train = pd.read_excel(f"{data_path}TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
connectome_test = pd.read_csv(f"{data_path}TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")

# === Convert to matrices ===
connectivity_matrices = unflatten_matrices(connectome_train)
connectivity_matrices_test = unflatten_matrices(connectome_test)

# === Plot option to visualize ===
plot_average_connectome(connectivity_matrices, targets_train, adhd=0, sex=1)
plot_grouped_connectomes(connectivity_matrices, targets_train)

# === Prep for training ===
train_data = data_to_tensor(connectivity_matrices,targets_train)
test_data = data_to_tensor_test(connectivity_matrices_test)

# === Load model ===
model = GCN()
# model.load_state_dict(torch.load("old_model.pt")) # reload old model for more training 

# === Train ===
trainer = Trainer(model, train_data)
trainer.train(epochs=100)
trainer.plot_losses()
# trainer.save_model('model_GNN.pt')

# === Test ===
tester = Tester(trainer.model, test_data, connectome_test)
tester.evaluate()
tester.get_submission_df
# tester.save_submission("submission_GNN.csv")
