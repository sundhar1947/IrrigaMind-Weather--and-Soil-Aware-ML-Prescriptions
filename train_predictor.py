# file: train_predictor.py
# Supervised PyTorch model to predict irrigation need (mm) from soil+weather features

import os, json, math, time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# 1) Dataset
class IrrigationDataset(Dataset):
    def __init__(self, csv_path: str, feature_cols: List[str], target_col: str, normalize=True):
        df = pd.read_csv(csv_path)
        self.X = df[feature_cols].astype(np.float32).values
        self.y = df[target_col].astype(np.float32).values
        if normalize:
            self.x_mean = self.X.mean(axis=0)
            self.x_std = self.X.std(axis=0) + 1e-6
            self.X = (self.X - self.x_mean) / self.x_std
        else:
            self.x_mean = np.zeros(self.X.shape[1], dtype=np.float32)
            self.x_std = np.ones(self.X.shape[1], dtype=np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# 2) Model
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=[128, 64, 32], dropout=0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x).squeeze(-1)

# 3) Training loop
def train(model, train_loader, val_loader, epochs=60, lr=1e-3, wd=1e-4, device="cpu"):
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(val_loader.dataset)
        sched.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), "predictor_best.pt")
            with open("scaler.json", "w") as f:
                json.dump({"mean": train_loader.dataset.x_mean.tolist(),
                           "std": train_loader.dataset.x_std.tolist()}, f)
        print(f"Epoch {ep+1:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")

    return history

# 4) Example main
if __name__ == "__main__":
    # CSV schema example:
    # timestamp,soil_moisture,air_temp,air_humidity,wind_speed,solar_rad,pressure,et0_mm,forecast_rain_mm,last_irrig_mm,target_mm
    csv_path = "data/irrigation_samples.csv"
    feature_cols = [
        "soil_moisture","air_temp","air_humidity","wind_speed",
        "solar_rad","pressure","et0_mm","forecast_rain_mm","last_irrig_mm"
    ]
    target_col = "target_mm"  # irrigation needed today (mm)

    ds = IrrigationDataset(csv_path, feature_cols, target_col, normalize=True)
    n_total = len(ds)
    n_val = max(64, int(0.2*n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = MLP(in_dim=len(feature_cols))
    history = train(model, train_loader, val_loader, epochs=80, lr=1e-3, wd=1e-4, device="cpu")

    # After training: convert to TorchScript for edge
    model.load_state_dict(torch.load("predictor_best.pt", map_location="cpu"))
    model.eval()
    example = torch.randn(1, len(feature_cols))
    traced = torch.jit.trace(model, example)
    traced.save("predictor_best.ts")
