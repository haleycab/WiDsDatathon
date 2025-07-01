import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

def create_test_graphs(connectome_test):
    connectivity_matrices_test = []

    for i in range(len(connectome_test)):
        flattened_fcm = connectome_test.iloc[i, 1:].values  # Skip the participant_id column
        fcm_matrix = flatten_to_square_matrix(flattened_fcm)
        connectivity_matrices_test.append(fcm_matrix)

    connectivity_matrices_test = np.array(connectivity_matrices_test)
    return connectivity_matrices_test

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

class ConnectomeProcessor:
    def __init__(self, size=200):
        self.size = size

    def flatten_to_square_matrix(self, flattened_fcm):
        expected_elements = (self.size * (self.size - 1)) // 2
        if len(flattened_fcm) != expected_elements:
            raise ValueError(f"Expected {expected_elements} elements, got {len(flattened_fcm)}")
        matrix = np.zeros((self.size, self.size))
        indices = np.triu_indices(self.size, k=1)
        matrix[indices] = flattened_fcm
        matrix.T[indices] = flattened_fcm
        return matrix

class ConnectomeGraph:
    def __init__(self, matrix, adhd=None, sex=None):
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Matrix must be a NumPy array")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")
        self.matrix = torch.tensor(matrix).float()
        self.adhd = adhd
        self.sex = sex

    def to_graph_data(self):
        edge_index = (self.matrix > 0).nonzero(as_tuple=False).t()
        edge_attr = self.matrix[edge_index[0], edge_index[1]]
        x = torch.eye(self.matrix.size(0))
        y = None
        if self.adhd is not None and self.sex is not None:
            y = torch.tensor([self.adhd, self.sex], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

class ConnectomeModel(torch.nn.Module):
    def __init__(self, input_channels=200, hidden_channels=64, output_channels=2):
        super(ConnectomeModel, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return global_mean_pool(x, batch)

class ModelWrapper:
    def __init__(self, model_path=None):
        self.model = ConnectomeModel()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, data_loader):
        predictions = []
        with torch.no_grad():
            for data in data_loader:
                out = self.model(data)
                out = out.view(-1, 2)
                probs = torch.sigmoid(out)
                pred = (probs > 0.5).int()
                predictions.extend(pred.tolist())
        return predictions

    def train_model(self, train_loader, epochs=100, lr=0.01):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            total_loss = 0
            for data in train_loader:
                optimizer.zero_grad()
                out = self.model(data)
                target = data.y.view(-1, 2).float()  # make sure target is same shape
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # assert out.shape == target.shape, f"Output shape {out.shape} != target shape {target.shape}"
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

class Visualizer:
    def __init__(self, cmap='coolwarm'):
        self.cmap = cmap

    def plot_matrix(self, matrix, title='Connectome'):
        plt.figure(figsize=(6, 5))
        plt.imshow(matrix, cmap=self.cmap)
        plt.title(title, fontsize=14)
        plt.xlabel("Brain Region")
        plt.ylabel("Brain Region")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    def plot_grouped_averages(self, matrices, labels_df):
        groups = [
            {"adhd": 0, "sex": 0, "title": "Control Male"},
            {"adhd": 0, "sex": 1, "title": "Control Female"},
            {"adhd": 1, "sex": 0, "title": "ADHD Male"},
            {"adhd": 1, "sex": 1, "title": "ADHD Female"},
        ]

        matrices = torch.tensor(matrices).float()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, group in enumerate(groups):
            mask = (labels_df["ADHD_Outcome"] == group["adhd"]) & (labels_df["Sex_F"] == group["sex"])
            selected = matrices[mask.values]
            avg_matrix = torch.mean(selected, dim=0) if selected.shape[0] > 0 else torch.zeros((200, 200))
            im = axes[i].imshow(avg_matrix, cmap=self.cmap)
            axes[i].set_title(group["title"], fontsize=14)
            axes[i].set_xlabel("Brain Region")
            axes[i].set_ylabel("Brain Region")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    try:
        example_flat = np.random.rand(19900)
        matrix = processor.flatten_to_square_matrix(example_flat)
        graph = ConnectomeGraph(matrix, adhd=1, sex=0)
        data = graph.to_graph_data()
        print(data)
        visualizer = Visualizer()
        visualizer.plot_matrix(matrix, title="Example Connectome")
    except Exception as e:
        print("Error:", e)
