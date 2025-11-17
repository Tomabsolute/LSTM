# LSTM forecasting on AAPL_data.csv (PyTorch version)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path(r"AAPL_data.csv")
OUT_DIR = Path("./outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_series(date_col="date", price_col="close"):
    df = pd.read_csv(DATA_PATH)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    s = df.set_index(date_col)[price_col].astype(float)
    return s

def make_supervised(values, window, horizon=1):
    X, y = [], []
    for i in range(len(values) - window - horizon + 1):
        X.append(values[i:i+window])
        y.append(values[i+window:i+window+horizon])
    return np.array(X), np.array(y).reshape(-1, horizon)

def split_series(values, test_ratio=0.2, min_test=30):
    n = len(values)
    t = max(int(n*test_ratio), min_test)
    return values[:-t], values[-t:]

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/(y_true+1e-9)))*100
    return mae, rmse, mape

# ====== PyTorch LSTM 定义 ======
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True  # (batch, seq_len, feat)
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # 取最后一个时间步
        last_hidden = out[:, -1, :]  # (batch, hidden_size)
        out = self.fc(last_hidden)   # (batch, horizon)
        return out

def lstm_torch(train_vals, test_vals, window, horizon=1,
               epochs=20, batch=32, units=64, dropout=0.0,Loss=[]):
    # dropout 参数暂不使用（可以在 LSTM/Linear 后加 nn.Dropout）
    scaler = MinMaxScaler()
    all_vals = np.concatenate([train_vals, test_vals])
    scaled = scaler.fit_transform(all_vals.reshape(-1,1)).flatten()
    train_scaled = scaled[:len(train_vals)]
    test_scaled = scaled[len(train_vals):]

    # 构造监督学习数据
    X_train, y_train = make_supervised(train_scaled, window, horizon)

    # 转成张量并放到 device
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)  # (N, window, 1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)               # (N, horizon)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True)

    # 模型、损失、优化器
    model = LSTMPredictor(input_size=1, hidden_size=units, horizon=horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        print(f"Epoch {ep+1}/{epochs} - loss: {epoch_loss/len(dataset):.6f}")
    Loss.append(loss)


    # 预测：自回归滚动预测，与原 TF 版本逻辑一致
    model.eval()
    history = list(train_scaled[-window:])
    preds = []
    with torch.no_grad():
        for t in range(len(test_scaled)):
            x = np.array(history[-window:]).reshape(1, window, 1)
            x_t = torch.tensor(x, dtype=torch.float32).to(device)
            yhat = model(x_t).cpu().numpy().flatten()  # horizon=1 -> 取 [0]
            preds.append(yhat[0])
            # 用真实 test_scaled[t] 继续往前滚
            history.append(test_scaled[t])

    preds = np.array(preds).reshape(-1,1)
    inv_preds = scaler.inverse_transform(preds).flatten()
    inv_test = scaler.inverse_transform(test_scaled.reshape(-1,1)).flatten()
    return inv_test, inv_preds

def main(m):
    s = load_series()
    train, test = split_series(s.values.astype(float), 0.2, 30)
    Loss = []

    for i in range(m):
        n = int(input("请输入学习天数"))
        y_true, y_pred = lstm_torch(train, test,
                                    window=n, horizon=1,
                                    epochs=200, batch=32,
                                    units=64, dropout=0.1,Loss=Loss)
        MAE, RMSE, MAPE = metrics(y_true, y_pred)
        print(f"LSTM (PyTorch) -> MAE={MAE:.4f} RMSE={RMSE:.4f} MAPE={MAPE:.2f}%")

        plt.figure()
        plt.plot(range(len(y_true)), y_true, label="Test")
        plt.plot(range(len(y_pred)), y_pred, label=f"LSTM Forecast$_{{{n}}}$")
        plt.title("LSTM Forecast vs Actual (PyTorch)")
        plt.xlabel("Time steps"); plt.ylabel("close")
        plt.legend(); plt.tight_layout()
        plt.savefig(OUT_DIR / f"lstm_forecast_torch_{i}.png", dpi=600)
        plt.close()
    
    print(Loss)

if __name__ == "__main__":
    main(3)