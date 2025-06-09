import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, global_mean_pool

# === Load TRAIN data ===
train_path = "/Users/Haley/Desktop/WiDs Datathon/widsdatathon2025/TRAIN_NEW"
connectome_train = pd.read_csv(f"{train_path}/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
quant_meta_train = pd.read_excel(f"{train_path}/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
cat_meta_train = pd.read_excel(f"{train_path}/TRAIN_CATEGORICAL_METADATA_new.xlsx")
targets_train = pd.read_excel(f"{train_path}/TRAINING_SOLUTIONS.xlsx")


# Check shapes
print("Train Connectome:", connectome_train.shape)
print("Train Quantitative metadata:", quant_meta_train.shape)
print("Train Categorical metadata:", cat_meta_train.shape)
print("Train Targets:", targets_train.shape)

# === Load TEST data ===
test_path = "/Users/Haley/Desktop/WiDs Datathon/widsdatathon2025/TEST"
connectome_test = pd.read_csv(f"{test_path}/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
quant_meta_test = pd.read_excel(f"{test_path}/TEST_QUANTITATIVE_METADATA.xlsx")
cat_meta_test = pd.read_excel(f"{test_path}/TEST_CATEGORICAL.xlsx")


# Check shapes
print("Test Connectome:", connectome_test.shape)
print("Test Quantitative metadata:", quant_meta_test.shape)
print("Test Categorical metadata:", cat_meta_test.shape)

def flatten_to_square_matrix(flattened_fcm, size=200):
    # Ensure the length of the flattened matrix corresponds to the upper triangular part of a 192x192 matrix
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

# Example for the first participant
flattened_fcm = connectome_train.iloc[0, 1:].values  # Skip the participant_id column
fcm_matrix = flatten_to_square_matrix(flattened_fcm)
print(fcm_matrix.shape)  # Should print (200, 200)

# Assuming train_FCM is a pandas DataFrame with participant IDs and flattened FCMs
connectivity_matrices = []

for i in range(len(connectome_train)):
    flattened_fcm = connectome_train.iloc[i, 1:].values  # Skip the participant_id column
    fcm_matrix = flatten_to_square_matrix(flattened_fcm)
    connectivity_matrices.append(fcm_matrix)

x = np.array(connectivity_matrices)
print("Train Connectomes:",connectivity_matrices.shape)  # Should print (N, 200, 200), where N is the number of participants


connectivity_matrices_test = []

for i in range(len(connectome_test)):
    flattened_fcm = connectome_test.iloc[i, 1:].values  # Skip the participant_id column
    fcm_matrix = flatten_to_square_matrix(flattened_fcm)
    connectivity_matrices_test.append(fcm_matrix)

connectivity_matrices_test = np.array(connectivity_matrices_test)
print("Test Connectomes:",connectivity_matrices_test.shape) 

def label_to_str(label_row):
    adhd_str = "ADHD" if label_row[1] == 1 else "Control"
    gender_str = "Female" if label_row[2] == 1 else "Male"
    return f"{adhd_str}, {gender_str}"

fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

# Plot first connectome
img1 = axes[0].imshow(connectivity_matrices[3], cmap='coolwarm')
# axes[0].set_title(label_to_str(targets_train[1]), fontsize=14)
axes[0].set_xlabel('Brain Region', fontsize=12)
axes[0].set_ylabel('Brain Region', fontsize=12)
fig.colorbar(img1, ax=axes[0], fraction=0.046, pad=0.04)

# Plot second connectome
img2 = axes[1].imshow(connectivity_matrices[2], cmap='coolwarm')
# axes[1].set_title(label_to_str(targets_train[2]), fontsize=14)
axes[1].set_xlabel('Brain Region', fontsize=12)
axes[1].set_ylabel('Brain Region', fontsize=12)
fig.colorbar(img2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

import torch
import matplotlib.pyplot as plt

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

plot_grouped_connectomes(connectivity_matrices, targets_train)

def data_to_tensor(connectivity_matrices,targets_train):

    connectivity_matrices = torch.tensor(connectivity_matrices).float()
    # labels = torch.tensor(targets_train).float()

    num_nodes = connectivity_matrices.shape[1]

    data_list = []

    for i in range(len(connectivity_matrices)):
        matrix = connectivity_matrices[i]

        # Create edges from non-zero entries
        edge_index = (matrix > 0).nonzero(as_tuple=False).t()
        edge_attr = matrix[edge_index[0], edge_index[1]]

        # Node features: use identity matrix (or you could use others)
        x = torch.eye(num_nodes)

        # Use ADHD and Sex as labels
        ADHD = targets_train.iloc[i, 1] # ADHD
        SEX = targets_train.iloc[i, 2]  # shape (2,) → [ADHD, Sex_F]
        y=torch.tensor([ADHD, SEX],dtype=torch.float)
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(graph_data)

    return data_list

train_data = data_to_tensor(connectivity_matrices,targets_train)
train_data[:5]

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


from torch_geometric.data import DataLoader

# Initialize model, loss function, and optimizer
# model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

# Define DataLoader
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

# Training loop
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        # out = out.reshape(8, 2)
                # shape: (batch_size, 2)
        loss = criterion(out, data.y)    # target y shape must also be (batch_size, 2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# # Run training
# losses = []
# for epoch in range(100):
#     loss = train()
#     losses.append(loss)
#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Loss: {loss:.4f}')

# # Plot loss
# plt.figure(figsize=(6, 3))
# plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b')
# plt.title('Training Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Continue training and adding to losses
for epoch in range(len(losses), len(losses) + 50):
    loss = train()
    losses.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


import matplotlib.pyplot as plt

epochs = list(range(1, len(losses) + 1))

plt.figure(figsize=(6, 3))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


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

test_data = data_to_tensor_test(connectivity_matrices_test)

import torch
import pandas as pd
from torch_geometric.loader import DataLoader

# 1. Set up DataLoader
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# 2. Switch model to evaluation mode
model.eval()

# 3. Collect predictions
predicted_rows = []

with torch.no_grad():
    for data in test_loader:
        out = model(data)  # Shape: [batch_size, 2]
        out = out.reshape(8, 2)
        # print(out.shape)
        preds = (torch.sigmoid(out) > 0.5).int()  # Binarize directly
        predicted_rows.extend(preds.tolist())     # Convert batch predictions to list

# 4. Build DataFrame
submission_df = pd.DataFrame(predicted_rows, columns=["ADHD_Outcome", "Sex_F"])

# 5. Check shape
assert submission_df.shape == (304, 2), f"Expected 304 rows, got {submission_df.shape[0]}"

# 6. Save to CSV

# Optional: Save to Parquet if needed
# submission_df.to_parquet("submission.parquet", index=False)
