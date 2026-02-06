# 完整独立版 analyze_feature_corr.py
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# 复制utils里的核心函数（避免依赖问题）
def clean_numeric_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def evaluate_ic(y_true, y_pred) -> float:
    y_true = clean_numeric_array(y_true)
    y_pred = clean_numeric_array(y_pred)
    min_len = min(len(y_true), len(y_pred))
    if min_len < 2:
        return 0.0
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    ic, p_value = pearsonr(y_true, y_pred)
    return float(ic) if not np.isnan(ic) else 0.0

def load_day_data(day: str, data_path: str = "./data") -> dict:
    day = str(day)
    day_path = os.path.join(data_path, day)
    required = ["A.csv", "B.csv", "C.csv", "D.csv", "E.csv"]
    out = {}
    for fname in required:
        fp = os.path.join(day_path, fname)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"缺失文件：{fp}")
        try:
            df = pd.read_csv(fp, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding="gbk")
        for col in ["LastPrice", "TradeBuyVolume", "TradeSellVolume", "Return5min"]:
            if col not in df.columns:
                df[col] = 0.0
        out[fname.split(".")[0]] = df
    return out

def get_day_folders(data_path: str = "./data") -> list:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据目录不存在：{data_path}")
    days = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.strip().isdigit()]
    days.sort(key=lambda x: int(x))
    return days

def calculate_batch_features(df_E, df_A, df_B, df_C, df_D) -> tuple[np.ndarray, tuple]:
    # 简化版特征计算（和你utils里的维度一致，16维）
    n = len(df_E)
    feat = np.zeros((n, 16), dtype=np.float64)
    feat_idx = 0

    # 基础特征（3维）
    e_p = df_E["LastPrice"].values
    e_buy = df_E["TradeBuyVolume"].values
    e_sell = df_E["TradeSellVolume"].values
    e_vol = e_buy + e_sell
    e_order_flow = e_buy - e_sell

    feat[:, feat_idx] = e_vol / (e_vol + 1e-6); feat_idx +=1
    e_p_ma5 = pd.Series(e_p).rolling(window=5, min_periods=1, closed='left').mean().values
    feat[:, feat_idx] = (e_p - e_p_ma5) / (e_p_ma5 + 1e-6); feat_idx +=1
    feat[:, feat_idx] = e_order_flow / (e_vol + 1e-6); feat_idx +=1

    # 差分特征（3维）
    feat[:, feat_idx] = np.diff(e_vol, prepend=e_vol[0]); feat_idx +=1
    feat[:, feat_idx] = np.diff(e_p, prepend=e_p[0]); feat_idx +=1
    feat[:, feat_idx] = (e_p - np.roll(e_p, 1)) / (np.roll(e_p, 1) + 1e-6); feat_idx +=1

    # 其他特征（1维）
    feat[:, feat_idx] = np.ones(n); feat_idx +=1

    # Alpha特征（6维）
    a_p = df_A["LastPrice"].values
    b_p = df_B["LastPrice"].values
    c_p = df_C["LastPrice"].values
    d_p = df_D["LastPrice"].values
    feat[:, feat_idx] = e_order_flow - (df_A["TradeBuyVolume"].values - df_A["TradeSellVolume"].values); feat_idx +=1
    feat[:, feat_idx] = (e_p - np.roll(e_p,1)) - (a_p - np.roll(a_p,1)); feat_idx +=1
    feat[:, feat_idx] = (a_p - np.roll(a_p,1)) / (np.roll(a_p,1)+1e-6); feat_idx +=1
    feat[:, feat_idx] = (b_p - np.roll(b_p,1)) / (np.roll(b_p,1)+1e-6); feat_idx +=1
    feat[:, feat_idx] = (c_p - np.roll(c_p,1)) / (np.roll(c_p,1)+1e-6); feat_idx +=1
    feat[:, feat_idx] = (d_p - np.roll(d_p,1)) / (np.roll(d_p,1)+1e-6); feat_idx +=1

    # 新增有效特征（3维）
    feat[:, feat_idx] = (e_buy - e_sell) / (e_buy + e_sell + 1e-6); feat_idx +=1
    feat[:, feat_idx] = pd.Series(e_vol).rolling(window=10, min_periods=1).std().values; feat_idx +=1
    feat[:, feat_idx] = ((e_p - np.roll(e_p,1)) / (np.roll(e_p,1)+1e-6)) * (e_vol - np.roll(e_vol,1)); feat_idx +=1

    feat = clean_numeric_array(feat)
    return feat, ()

# 核心分析逻辑
def analyze_feature_correlation():
    print("开始分析特征相关性...")
    # 加载训练数据（排除测试集5）
    days = [d for d in get_day_folders("./data") if d != '5']
    if not days:
        print("错误：没有找到训练交易日！")
        return
    
    X_list, y_list = [], []
    for d in days:
        print(f"加载交易日 {d} 数据...")
        data = load_day_data(d)
        X, _ = calculate_batch_features(data["E"], data["A"], data["B"], data["C"], data["D"])
        y = data["E"]["Return5min"].values.astype(np.float32)
        y_shifted = np.roll(y, -1)
        y_shifted[-1] = np.nan
        X_list.append(X)
        y_list.append(y_shifted)
    
    # 合并清洗
    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)
    valid_mask = ~(np.isnan(y_all) | np.isnan(X_all).any(axis=1))
    X_all = X_all[valid_mask]
    y_all = y_all[valid_mask]
    
    print(f"有效数据量：特征矩阵 {X_all.shape}，标签 {y_all.shape}")
    if len(X_all) == 0:
        print("错误：无有效数据！")
        return
    
    # 定义16个特征的名称（对应维度）
    feature_names = [
        "vol_ratio", "price_ma5_rel", "order_flow_ratio",  # 基础特征
        "vol_diff", "price_diff", "price_ret",            # 差分特征
        "time_period",                                    # 其他特征
        "order_flow_rsi", "return_residual",              # Alpha特征
        "A_1min_return", "B_1min_return", 
        "C_1min_return", "D_1min_return",
        "order_imbalance", "vol_volatility",              # 新增有效特征
        "price_vol_trend"
    ]
    
    # 计算每个特征的IC
    print("\n===== 各特征与标签的相关性（IC）=====")
    feat_ic = []
    for i in range(X_all.shape[1]):
        ic = evaluate_ic(y_all, X_all[:, i])
        feat_ic.append({
            "特征名称": feature_names[i],
            "IC值": round(ic, 4),
            "相关性方向": "正向" if ic > 0 else "反向"
        })
    
    # 输出结果（按IC绝对值排序）
    df = pd.DataFrame(feat_ic)
    df["IC绝对值"] = df["IC值"].abs()
    df = df.sort_values("IC绝对值", ascending=False)
    print(df[["特征名称", "IC值", "相关性方向"]])
    return df

if __name__ == "__main__":
    try:
        analyze_feature_correlation()
    except Exception as e:
        print(f"\n运行出错：{type(e).__name__} - {e}")