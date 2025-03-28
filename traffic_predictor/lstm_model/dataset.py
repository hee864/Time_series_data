import torch
from torch.utils.data import Dataset
import numpy as np

class TrafficDataset(Dataset):
    def __init__(self, series, input_size=24, pred_size=1):
        self.series = series
        self.input_size = input_size
        self.pred_size = pred_size
        self.samples = self._create_sequences()

        
        print("시리즈 길이:", len(self.series))
        print("생성된 시퀀스 개수:", len(self.samples))


    def _create_sequences(self):
        X, y = [], []
        for i in range(len(self.series) - self.input_size - self.pred_size + 1):
            x_seq = self.series[i:i + self.input_size]
            y_seq = self.series[i + self.input_size:i + self.input_size + self.pred_size]
            X.append(x_seq)
            y.append(y_seq)
        return list(zip(X, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
