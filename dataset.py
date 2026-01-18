import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class IoTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_client_data(root_dir, client_name, batch_size=32):
    root_dir = os.path.abspath(root_dir)
    client_dir = os.path.join(root_dir, client_name)
    if not os.path.isdir(client_dir):
        raise RuntimeError(f"Client dir not found: {client_dir}")
    
    data = []
    labels = []

    # ---------benign traffic------------
    benign_path = os.path.join(client_dir, "benign_traffic.csv")
    if not os.path.isfile(benign_path):
        raise RuntimeError(f"Missing benign CSV: {benign_path}")
    benign_df = pd.read_csv(benign_path)
    data.append(benign_df.values)
    labels.extend([0] * len(benign_df))

    #-----------gafgyt attacks----------
    gafgyt_dir = os.path.join(client_dir, "gafgyt_attacks")
    for f in os.listdir(gafgyt_dir):
        df = pd.read_csv(os.path.join(gafgyt_dir, f))
        data.append(df.values)
        labels.extend([1] * len(df))

    #----------mirai attacks---------
    mirai_dir = os.path.join(client_dir, "mirai_attacks")
    for f in os.listdir(mirai_dir):
        df = pd.read_csv(os.path.join(mirai_dir, f))
        data.append(df.values)
        labels.extend([2] * len(df))

    X = pd.concat([pd.DataFrame(d) for d in data]).values
    y = labels

    dataset = IoTDataset(X,y)

    train_size = int(0.8*len(dataset))
    test_size = len(dataset) -train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1]
    num_classes = 3

    return train_loader, test_loader, input_dim, num_classes

