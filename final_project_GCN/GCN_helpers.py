"""
GNN_helpers.py
This file contains functions for implementation of a simple Graph Convolutional Neural Network (GCN) to predict ADHD diagnosis and sex.
Runs with functional connectome matrices from WiDs Datathon
Includes data preprocessing, visualization, training, and testing 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, global_mean_pool

def flatten_to_square_matrix(flattened_fcm, size=200):
    if not isinstance(flattened_fcm, (np.ndarray, list)):
        raise TypeError("Input must be a NumPy array or list.")
    # Ensure the length of the flattened matrix corresponds to the upper triangular part of a matrix
    num_elements = len(flattened_fcm)
    expected_elements = (size * (size - 1)) // 2
    if num_elements != expected_elements:
        raise ValueError(f"Flattened matrix size mismatch. Expected {expected_elements} elements, got {num_elements}")

    # Initialize a square matrix (size x size) filled with zeros
    matrix = np.zeros((size, size))

    # Extract the upper triangular indices (i, j) where i < j
    indices = np.triu_indices(size, k=1)  # k=1 excludes diagonal (i != j)

    # Assign the flattened values to the upper triangular part of the matrix
    matrix[indices] = flattened_fcm
    matrix.T[indices] = flattened_fcm  # Symmetric part: Copy to the lower triangle

    return matrix

def unflatten_matrices(connectomes):
    connectivity_matrices = []
    for i in range(len(connectomes)):
        flattened_fcm = connectomes.iloc[i, 1:].values  # Skip the participant_id column
        fcm_matrix = flatten_to_square_matrix(flattened_fcm)
        connectivity_matrices.append(fcm_matrix)
    connectivity_matrices = np.array(connectivity_matrices)
    return connectivity_matrices
    # print("Train Connectomes:",connectivity_matrices.shape)  # Should print (N, 200, 200), where N is the number of participant

def plot_average_connectome(connectivity_matrices, targets_df, adhd, sex, title=None, cmap="coolwarm"):
    # Ensure tensor format
    if not isinstance(connectivity_matrices, torch.Tensor):
        connectivity_matrices = torch.tensor(connectivity_matrices, dtype=torch.float32)

    # Build mask and apply
    mask = (targets_df["ADHD_Outcome"] == adhd) & (targets_df["Sex_F"] == sex)
    selected_matrices = connectivity_matrices[mask.values]
    
    if selected_matrices.shape[0] == 0:
        print(f"No data found for ADHD={adhd}, Sex={sex}.")
        return
    # Average connectome
    avg_matrix = torch.mean(selected_matrices, dim=0)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(avg_matrix, cmap=cmap)
    plt.title(title or f"Average Connectome: {'ADHD' if adhd else 'Control'} {'Female' if sex else 'Male'}", fontsize=14)
    plt.xlabel("Brain Region")
    plt.ylabel("Brain Region")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def data_to_tensor(connectivity_matrices,targets_train):
    if len(connectivity_matrices) != len(targets_train):
        raise ValueError("Number of connectomes and labels must match.")
    connectivity_matrices = torch.tensor(connectivity_matrices).float()
    # labels = torch.tensor(targets_train).float()

    num_nodes = connectivity_matrices.shape[1]

    data_list = []

    for i in range(len(connectivity_matrices)):
        matrix = connectivity_matrices[i]

        edge_index = (matrix > 0).nonzero(as_tuple=False).t() # identifies connections
        edge_attr = matrix[edge_index[0], edge_index[1]] # stores weights

        # Node features: use identity matrix (or you could use others)
        x = torch.eye(num_nodes)

        # Use ADHD and Sex as labels
        ADHD = targets_train.iloc[i, 1] # ADHD
        SEX = targets_train.iloc[i, 2]  # shape (2,) → [ADHD, Sex_F]
        y=torch.tensor([ADHD, SEX],dtype=torch.float)
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(graph_data)

    return data_list

def plot_grouped_connectomes(connectivity_matrices, targets_df):
    """
    Plots average connectomes for 4 groups:
    1. Control Male
    2. Control Female
    3. ADHD Male
    4. ADHD Female
    """
    groups = [
        {"adhd": 0, "sex": 0, "title": "Control Male"},
        {"adhd": 0, "sex": 1, "title": "Control Female"},
        {"adhd": 1, "sex": 0, "title": "ADHD Male"},
        {"adhd": 1, "sex": 1, "title": "ADHD Female"},
    ]
    
    # Convert to tensor if needed
    if isinstance(connectivity_matrices, np.ndarray):
        connectivity_matrices = torch.tensor(connectivity_matrices)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, group in enumerate(groups):
        mask = (
            (targets_df["ADHD_Outcome"] == group["adhd"]) & 
            (targets_df["Sex_F"] == group["sex"])
        )

        selected = connectivity_matrices[mask]
        if selected.shape[0] == 0:
            avg_matrix = torch.zeros((200, 200))  # fallback if empty
            title = group["title"] + " (no data)"
        else:
            avg_matrix = torch.mean(selected, dim=0)
            title = group["title"]

        im = axes[i].imshow(avg_matrix, cmap='coolwarm')
        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel("Brain Region")
        axes[i].set_ylabel("Brain Region")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=200, out_channels=64)
        self.conv2 = GCNConv(in_channels=64, out_channels=2)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x                                 )
        x = self.conv2(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)
        return x.view(-1)
    
def data_to_tensor_test(connectivity_matrices):
    connectivity_matrices = torch.tensor(connectivity_matrices).float()
    num_nodes = connectivity_matrices.shape[1]

    data_list = []

    for i in range(len(connectivity_matrices)):
        matrix = connectivity_matrices[i]
        edge_index = (matrix > 0).nonzero(as_tuple=False).t()
        edge_attr = matrix[edge_index[0], edge_index[1]]
        x = torch.eye(num_nodes)
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(graph_data)
    return data_list

class Trainer:
    def __init__(self, model, train_data, batch_size=8, lr=0.01):
        self.model = model
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.losses = []

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            self.optimizer.zero_grad()
            out = self.model(data)  # Assuming model returns shape (batch_size, 2)
            loss = self.criterion(out, data.y)  # data.y should also be (batch_size, 2)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def train(self, epochs=100, log_interval=10):
        for epoch in range(epochs):
            loss = self.train_one_epoch()
            self.losses.append(loss)
            if epoch % log_interval == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
        return self.losses

    def plot_losses(self):
        plt.figure(figsize=(6, 3))
        plt.plot(range(1, len(self.losses) + 1), self.losses, marker='o', linestyle='-', color='b')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_model(self, path='model.pt'):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

class Tester:
    def __init__(self, model, test_data, connectome_test, batch_size=8):
        self.model = model
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        self.connectome_test = connectome_test
        self.predicted_rows = []

    def evaluate(self):
        self.model.eval()
        self.predicted_rows = []

        with torch.no_grad():
            for data in self.test_loader:
                out = self.model(data)
                out = out.reshape(-1, 2)
                preds = (torch.sigmoid(out) > 0.5).int()
                self.predicted_rows.extend(preds.tolist())

    def get_submission_df(self, expected_rows=304):
        df = pd.DataFrame(self.predicted_rows, columns=["ADHD_Outcome", "Sex_F"])
        assert df.shape[0] == expected_rows, f"Expected {expected_rows} rows, got {df.shape[0]}"
        df["participant_id"] = self.connectome_test["participant_id"].values
        return df
    
    def preview_submission(self, n=5):
        df = self.get_submission_df()
        print(df.head(n))

    def save_submission(self, path="submission4.csv"):
        df = self.get_submission_df()
        df.to_csv(path, index=False)
        print(f"Submission saved to {path}")

