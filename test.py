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

DATA_PATH = Path("stock.csv")      
OUT_DIR = Path("./outputs1"); OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_NAME = "AAPL"
OTHER_NAMES = ["AAL", "ABBV"]     

FEATURES = ["open", "high", "low", "close", "volume"]
TARGET = "close"

# ====== 数据加载与处理 ======
def load_all(date_col="date", name_col="Name"):
    df = pd.read_csv(DATA_PATH)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values([name_col, date_col])
    return df

def get_series_by_name(df, name, date_col="date", name_col="Name"):
    sub = df[df[name_col] == name].copy()
    if sub.empty:
        raise ValueError(f"在 CSV 中找不到股票: {name}")
    sub = sub.sort_values(date_col)
    return sub.set_index(date_col)

def split_series(values, test_ratio=0.2, min_test=30):
    n = len(values)
    t = max(int(n * test_ratio), min_test)
    return values[:-t], values[-t:]

def make_supervised_multi(X_feat, y_vals, window, horizon=1):
    """
    X_feat: (T, n_features)
    y_vals: (T,) 目标是一维 close
    """
    X_list, y_list = [], []
    T = len(y_vals)
    for i in range(T - window - horizon + 1):
        X_list.append(X_feat[i:i+window, :])
        y_list.append(y_vals[i+window:i+window+horizon])
    X = np.stack(X_list)                 # (N, window, n_features)
    y = np.array(y_list).reshape(-1, horizon)   # (N, horizon)
    return X, y

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    return mae, rmse, mape

# ====== LSTM 模型 ======
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]     # (batch, hidden_size)
        out = self.fc(last_hidden)      # (batch, horizon)
        return out

# ====== 训练 + 在同一股票 test 上预测 ======
def lstm_torch_multi(
    X_train_feat, y_train_vals,
    X_test_feat, y_test_vals,
    window,
    horizon=1,
    epochs=200,
    batch=32,
    units=64,
    return_model=False
):
    """
    使多因子模型和单因子模型风格一致：
    - 训练：固定窗口 -> 下一日 close
    - 测试：从 train 尾部取一个 window，滚动预测 test 段，
            每一步用“真实的 test 特征”推进窗口（teacher forcing）
    """
    assert horizon == 1, "当前实现只支持 horizon=1"

    n_features = X_train_feat.shape[1]

    # ===== 1. 联合归一化 train+test =====
    all_feat = np.vstack([X_train_feat, X_test_feat])
    all_target = np.concatenate([y_train_vals, y_test_vals])

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    feat_scaled = scaler_X.fit_transform(all_feat)
    target_scaled = scaler_y.fit_transform(all_target.reshape(-1, 1)).flatten()

    train_len = len(X_train_feat)
    X_train_scaled = feat_scaled[:train_len]
    X_test_scaled = feat_scaled[train_len:]
    y_train_scaled = target_scaled[:train_len]
    y_test_scaled = target_scaled[train_len:]

    # ===== 2. 构造训练样本：滑动窗口 -> 下一天 =====
    X_list, y_list = [], []
    for i in range(len(y_train_scaled) - window):
        X_list.append(X_train_scaled[i:i+window, :])   # (window, n_features)
        y_list.append(y_train_scaled[i+window])        # 标的是第 window+1 天的 close

    X_train_seq = np.array(X_list)                    # (N, window, n_features)
    y_train_seq = np.array(y_list).reshape(-1, 1)     # (N, 1)

    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True)

    # ===== 3. 定义模型 =====
    model = LSTMPredictor(input_size=n_features, hidden_size=units, horizon=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ===== 4. 训练 =====
    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)          # (batch, 1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        print(f"Epoch {ep+1}/{epochs} - loss: {epoch_loss/len(dataset):.6f}")

    # ===== 5. 在 test 段做“滚动预测”（与单因子风格一致）=====
    model.eval()
    preds_scaled = []

    # 初始 history：取 train 段最后 window 天的特征
    history_feat = list(X_train_scaled[-window:])   # 长度 = window

    with torch.no_grad():
        for t in range(len(X_test_scaled)):
            # 用 history 中最近 window 天作为模型输入
            x = np.array(history_feat[-window:]).reshape(1, window, n_features)
            x_t = torch.tensor(x, dtype=torch.float32).to(device)
            yhat_scaled = model(x_t).cpu().numpy().flatten()[0]
            preds_scaled.append(yhat_scaled)

            # 用真实的 test 特征推进窗口（teacher forcing）
            history_feat.append(X_test_scaled[t])

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)

    # y_true：直接用原始尺度的 y_test_vals
    inv_preds = scaler_y.inverse_transform(preds_scaled).flatten()
    inv_test = y_test_vals[-len(inv_preds):]   # 对齐（正常情况长度相等）

    if return_model:
        return inv_test, inv_preds, model, scaler_X, scaler_y
    else:
        return inv_test, inv_preds
    
# ====== 用训练好的模型在其它股票上预测 ======
def forecast_other_stock_multi(model, scaler_X, scaler_y, df_stock, window, horizon=1):
    """
    df_stock: 某一只股票的数据，index 为 date，包含 FEATURES 和 TARGET
    使用在 AAPL 上训练好的多因子模型，在该股票上做滚动预测：
    - 起始窗口：该股票前 window 天的真实特征
    - 每一步：用 history 的 window 天特征预测下一日 close
             然后用“真实特征”推进 history（teacher forcing）
    """
    assert horizon == 1, "当前实现只支持 horizon=1"

    feat = df_stock[FEATURES].values.astype(float)   # (T, n_features)
    target = df_stock[TARGET].values.astype(float)   # (T,)

    # 用训练时的 scaler 归一化
    feat_scaled = scaler_X.transform(feat)
    target_scaled = scaler_y.transform(target.reshape(-1, 1)).flatten()

    n_features = feat_scaled.shape[1]
    T = len(feat_scaled)
    if T <= window:
        raise ValueError("该股票数据长度不足以构造窗口")

    # 初始 history：前 window 天真实特征
    history_feat = list(feat_scaled[:window])
    preds_scaled = []

    model.eval()
    with torch.no_grad():
        for t in range(window, T):
            x = np.array(history_feat[-window:]).reshape(1, window, n_features)
            x_t = torch.tensor(x, dtype=torch.float32).to(device)
            yhat_scaled = model(x_t).cpu().numpy().flatten()[0]
            preds_scaled.append(yhat_scaled)

            # teacher forcing：用真实特征推进
            history_feat.append(feat_scaled[t])

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    # 真实 close（与预测对齐，从第 window+1 天开始）
    y_true_scaled = target_scaled[window:]

    inv_preds = scaler_y.inverse_transform(preds_scaled).flatten()
    inv_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    return inv_true, inv_preds

def main(m):
    df_all = load_all()
    # 训练股票
    df_train_stock = get_series_by_name(df_all, TRAIN_NAME)
    if len(df_train_stock) < 100:
        raise ValueError(f"{TRAIN_NAME} 数据太少")

    # 特征与目标
    feat_vals = df_train_stock[FEATURES].values.astype(float)
    target_vals = df_train_stock[TARGET].values.astype(float)

    # 只按时间切 train/test
    n = len(target_vals)
    t = max(int(n * 0.2), 30)
    X_train_feat = feat_vals[:-t]
    X_test_feat  = feat_vals[-t:]
    y_train_vals = target_vals[:-t]
    y_test_vals  = target_vals[-t:]

    plt.figure()
    first_model = None
    first_scaler_X = None
    first_scaler_y = None
    first_window = None

    for i in range(m):
        n_window = int(input("请输入学习天数 window: "))

        if i == 0:
            y_true, y_pred, model, scaler_X, scaler_y = lstm_torch_multi(
                X_train_feat, y_train_vals,
                X_test_feat, y_test_vals,
                window=n_window,
                horizon=1,
                epochs=200,
                batch=32,
                units=64,
                return_model=True
            )
            first_model = model
            first_scaler_X = scaler_X
            first_scaler_y = scaler_y
            first_window = n_window
            plt.plot(range(len(y_true)), y_true, label=f"{TRAIN_NAME} Test True")
        else:
            y_true, y_pred = lstm_torch_multi(
                X_train_feat, y_train_vals,
                X_test_feat, y_test_vals,
                window=n_window,
                horizon=1,
                epochs=200,
                batch=32,
                units=64,
                return_model=False
            )

        MAE, RMSE, MAPE = metrics(y_true, y_pred)
        print(f"{TRAIN_NAME} LSTM (multi-factor, window={n_window}) -> "
              f"MAE={MAE:.4f} RMSE={RMSE:.4f} MAPE={MAPE:.2f}%")

        plt.plot(range(len(y_pred)), y_pred, label=f"{TRAIN_NAME} Forecast_{n_window}")

    plt.title(f"LSTM Forecast vs Actual on {TRAIN_NAME} (multi-factor)")
    plt.xlabel("Time steps"); plt.ylabel("close")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / f"lstm_multi_{TRAIN_NAME}_{n_window}.png", dpi=600)

    # 用第一次训练的模型在其它股票上预测
    if first_model is not None and OTHER_NAMES:
        for name in OTHER_NAMES:
            try:
                df_other = get_series_by_name(df_all, name)
            except ValueError as e:
                print(e)
                continue

            if len(df_other) <= first_window + 5:
                print(f"{name} 数据太少，跳过。")
                continue

            y_true_o, y_pred_o = forecast_other_stock_multi(
                first_model, first_scaler_X, first_scaler_y,
                df_other, window=first_window, horizon=1
            )
            MAE_o, RMSE_o, MAPE_o = metrics(y_true_o, y_pred_o)
            print(f"{name} (multi-factor, use {TRAIN_NAME}-trained model) -> "
                  f"MAE={MAE_o:.4f} RMSE={RMSE_o:.4f} MAPE={MAPE_o:.2f}%")

            plt.figure()
            plt.plot(range(len(y_true_o)), y_true_o, label=f"{name} True")
            plt.plot(range(len(y_pred_o)), y_pred_o, label=f"{name} Forecast")
            plt.title(f"LSTM multi-factor forecast on {name} (trained on {TRAIN_NAME})")
            plt.xlabel("Time steps"); plt.ylabel("close")
            plt.legend(); plt.tight_layout()
            plt.savefig(OUT_DIR / f"lstm_multi_{name}_{n_window}.png", dpi=600)

if __name__ == "__main__":
    main(1)
