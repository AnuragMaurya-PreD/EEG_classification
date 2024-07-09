import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
class MindBigDataDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, 0].values
        self.features = self.data.iloc[:, 1:].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.labels[idx]
        features = self.features[idx].reshape(5, 256)
        return torch.tensor(features), torch.tensor(label)

