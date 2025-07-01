import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
\
class Connectome:
    def __init__(self, matrix, adhd=None, sex=None):
        self._matrix = matrix
        self._adhd = adhd
        self._sex = sex

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        if isinstance(value, torch.Tensor):
            self._matrix = value
        else:
            raise ValueError("Matrix must be a torch.Tensor")

    @property
    def adhd(self):
        return self._adhd

    @adhd.setter
    def adhd(self, value):
        if value in [0, 1, None]:
            self._adhd = value
        else:
            raise ValueError("ADHD must be 0, 1, or None")

    @property
    def sex(self):
        return self._sex

    @sex.setter
    def sex(self, value):
        if value in [0, 1, None]:
            self._sex = value
        else:
            raise ValueError("Sex must be 0, 1, or None")

    def to_graph(self):
        edge_index = (self.matrix > 0).nonzero(as_tuple=False).t()
        edge_attr = self.matrix[edge_index[0], edge_index[1]]
        x = torch.eye(self.matrix.size(0))
        y = torch.tensor([self.adhd, self.sex], dtype=torch.float) if self.adhd is not None else None
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def plot_connectome(self, cmap='coolwarm'):
        plt.figure(figsize=(6, 5))
        plt.imshow(self.matrix, cmap=cmap)
        title = f"Connectome (ADHD: {self.adhd}, Sex_F: {self.sex})"
        plt.title(title, fontsize=14)
        plt.xlabel("Brain Region")
        plt.ylabel("Brain Region")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()
