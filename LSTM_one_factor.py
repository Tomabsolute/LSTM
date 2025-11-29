# LSTM forecasting on AAPL_data.csv (PyTorch, with other-stock forecast)
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
OUT_DIR = Path("./outputs1"); OUT_DIR.mkdir(parents=True, exist_ok=True)

OTHER_DATA_PATHS = [
    Path("GOOGL_data.csv"),
    Path("NVDA_data.csv"),
]

def load_series_from_path(path, date_col="date", price_col="close"):
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    s = df.set_index(date_col)[price_col].astype(float)
    return s

def load_series(date_col="date", price_col="close"):
    return load_series_from_path(DATA_PATH, date_col, price_col)

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

# ====== LSTM 模型 ======
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        out = self.fc(last_hidden)
        return out

# ====== 1. 训练 + 在 AAPL 测试集上预测（可选择返回 model）======
def lstm_torch(
    train_vals,
    test_vals,
    window,
    horizon=1,
    epochs=200,
    batch=32,
    units=64,
    dropout=0.0,
    return_model=False,     # <<< 新增：是否返回训练好的模型
):
    scaler = MinMaxScaler()
    all_vals = np.concatenate([train_vals, test_vals])
    scaled = scaler.fit_transform(all_vals.reshape(-1,1)).flatten()
    train_scaled = scaled[:len(train_vals)]
    test_scaled = scaled[len(train_vals):]

    X_train, y_train = make_supervised(train_scaled, window, horizon)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True)

    model = LSTMPredictor(input_size=1, hidden_size=units, horizon=horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

    # 在 AAPL 的 test 上滚动预测
    model.eval()
    history = list(train_scaled[-window:])
    preds = []
    with torch.no_grad():
        for t in range(len(test_scaled)):
            x = np.array(history[-window:]).reshape(1, window, 1)
            x_t = torch.tensor(x, dtype=torch.float32).to(device)
            yhat = model(x_t).cpu().numpy().flatten()
            preds.append(yhat[0])
            history.append(test_scaled[t])

    preds = np.array(preds).reshape(-1,1)
    inv_preds = scaler.inverse_transform(preds).flatten()
    inv_test = scaler.inverse_transform(test_scaled.reshape(-1,1)).flatten()

    if return_model:
        return inv_test, inv_preds, model
    else:
        return inv_test, inv_preds

# ====== 2. 用训练好的 model 在“其他股票”上预测 ======
def forecast_other_stock(model, window, other_vals):
    """
    model: 在 AAPL 上训练好的 LSTMPredictor
    window: 训练时用的学习天数（窗口长度）
    other_vals: 其他股票的价格序列 (np.array 一维)
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(other_vals.reshape(-1,1)).flatten()

    if len(scaled) <= window:
        raise ValueError("其他股票数据长度必须大于 window")

    history = list(scaled[:window])
    preds = []

    model.eval()
    with torch.no_grad():
        for t in range(window, len(scaled)):
            x = np.array(history[-window:]).reshape(1, window, 1)
            x_t = torch.tensor(x, dtype=torch.float32).to(device)
            yhat = model(x_t).cpu().numpy().flatten()
            preds.append(yhat[0])
            # 用真实值做 teacher forcing
            history.append(scaled[t])

    preds = np.array(preds).reshape(-1,1)
    inv_preds = scaler.inverse_transform(preds).flatten()
    true_part = other_vals[window:]  # 与 preds 对齐的真实价格

    return true_part, inv_preds

def main(m):
    s = load_series()
    train, test = split_series(s.values.astype(float), 0.2, 30)

    first_model = None
    first_window = None

    plt.figure()
    for i in range(m):
        n = int(input("请输入学习天数"))
        # 第一次训练时顺便把模型保存下来
        if i == 0:
            y_true, y_pred, model = lstm_torch(
                train, test,
                window=n, horizon=1,
                epochs=200, batch=32,
                units=64, dropout=0.1,
                return_model=True      # <<< 第一次要模型
            )
            first_model = model
            first_window = n
            plt.plot(range(len(y_true)), y_true, label="AAPL Test")
        else:
            y_true, y_pred = lstm_torch(
                train, test,
                window=n, horizon=1,
                epochs=200, batch=32,
                units=64, dropout=0.1,
                return_model=False
            )

        MAE, RMSE, MAPE = metrics(y_true, y_pred)
        print(f"AAPL LSTM (PyTorch, window={n}) -> MAE={MAE:.4f} RMSE={RMSE:.4f} MAPE={MAPE:.2f}%")

        plt.plot(range(len(y_pred)), y_pred, label=f"AAPL Forecast$_{{{n}}}$")

    plt.title("LSTM Forecast vs Actual on AAPL (PyTorch)")
    plt.xlabel("Time steps"); plt.ylabel("close")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "lstm_forecast_torch_AAPL.png", dpi=600)

    # ====== 3. 用第一次训练好的模型，在其他股票上做预测 ======
    if first_model is not None and first_window is not None and OTHER_DATA_PATHS:
        for other_path in OTHER_DATA_PATHS:
            if not other_path.exists():
                print(f"[警告] 找不到文件: {other_path}, 跳过。")
                continue
            s_other = load_series_from_path(other_path)
            other_vals = s_other.values.astype(float)

            y_true_o, y_pred_o = forecast_other_stock(first_model, first_window, other_vals)
            MAE_o, RMSE_o, MAPE_o = metrics(y_true_o, y_pred_o)
            print(f"{other_path.name} 预测结果 -> MAE={MAE_o:.4f} RMSE={RMSE_o:.4f} MAPE={MAPE_o:.2f}%")

            plt.figure()
            plt.plot(range(len(y_true_o)), y_true_o, label=f"{other_path.stem} True")
            plt.plot(range(len(y_pred_o)), y_pred_o, label=f"{other_path.stem} Forecast")
            plt.title(f"LSTM Forecast on {other_path.stem} (using AAPL-trained model)")
            plt.xlabel("Time steps"); plt.ylabel("close")
            plt.legend(); plt.tight_layout()
            out_name = f"lstm_forecast_torch_{other_path.stem}.png"
            plt.savefig(OUT_DIR / out_name, dpi=600)

if __name__ == "__main__":
    main(3)
