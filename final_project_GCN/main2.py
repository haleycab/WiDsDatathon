import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from ConnectomeModel import ConnectomeProcessor, ConnectomeGraph, ConnectomeModel, Visualizer

# === Load Data ===
train_path = "./TRAIN_NEW"
test_path = "./TEST"

print("Loading data...")
connectome_test = pd.read_csv(f"{test_path}/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
labels_df = pd.read_excel(f"{train_path}/TRAINING_SOLUTIONS.xlsx")

# === Process Test Data ===
processor = ConnectomeProcessor(size=200)
matrices = []

print("Processing connectomes...")
for i in range(len(connectome_test)):
    flattened = connectome_test.iloc[i, 1:].values  # skip participant_id
    try:
        matrix = processor.flatten_to_square_matrix(flattened)
        matrices.append(matrix)
    except ValueError as e:
        print(f"Skipping index {i}: {e}")

# === Create Graphs ===
test_graphs = [ConnectomeGraph(m).to_graph_data() for m in matrices]
test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)

# === Load Trained Model ===
print("Loading model...")
model = ConnectomeModel()
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))

# === Run Predictions ===
print("Running predictions...")
preds = model.predict(test_loader)
submission_df = pd.DataFrame(preds, columns=["ADHD_Outcome", "Sex_F"])
submission_df.to_csv("submission.csv", index=False)
print("Saved predictions to submission.csv")

# === Optional: Visualize a Sample ===
print("Visualizing one sample connectome...")
visualizer = Visualizer()
visualizer.plot_matrix(matrices[0], title="Sample Connectome")
