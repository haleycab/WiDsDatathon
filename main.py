import torch
import pandas as pd
import numpy as np
from connectome_data import Connectome

def flatten_to_square_matrix(flattened_fcm, size=200):
    matrix = np.zeros((size, size))
    indices = np.triu_indices(size, k=1)
    matrix[indices] = flattened_fcm
    matrix.T[indices] = flattened_fcm
    return torch.tensor(matrix, dtype=torch.float)

def main():
    train_path = "/Users/Haley/Desktop/WiDs Datathon/widsdatathon2025/TRAIN_NEW"
    connectome_train = pd.read_csv(f"{train_path}/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
    targets_train = pd.read_excel(f"{train_path}/TRAINING_SOLUTIONS.xlsx")

    # One subject
    index = 21
    flattened = connectome_train.iloc[index, 1:].values
    matrix = flatten_to_square_matrix(flattened)

    adhd = targets_train.iloc[index]["ADHD_Outcome"]
    sex = targets_train.iloc[index]["Sex_F"]

    sample = Connectome(matrix, adhd=int(adhd), sex=int(sex))

    print("ADHD:", sample.adhd)
    print("Sex:", sample.sex)

    # Clear labels to simulate "unknown"
    sample.adhd = None
    sample.sex = None

    # Predict using saved model
    sample.predict_labels("Exam2/model.pt")

    # Plot with predicted labels
    sample.plot_connectome()


    print("ADHD:", sample.adhd)
    print("Sex:", sample.sex)

    graph = sample.to_graph()
    print("Graph object created from real data:")
    print(graph)

    sample.plot_connectome()

if __name__ == "__main__":
    main()
