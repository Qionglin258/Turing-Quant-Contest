import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

# ===================== 全局常量（完全保留你的配置，仅屏蔽无效特征）=====================
FEATURE_CONFIG = {
    "base": ["vol_ratio", "price_ma5_rel", "order_flow_ratio"],
    "diff": ["vol_diff", "price_diff", "price_ret"],
    "other": ["time_period"],  # 保留索引，后续赋值为0
    "alpha": [
        "order_flow_rsi",
        "return_residual",  # 保留索引，后续赋值为0
        "A_1min_return",
        "B_1min_return",
        "C_1min_return",
        "D_1min_return"
    ],
    "new_effective": [
        "order_imbalance",
        "vol_volatility",
        "price_vol_trend"
    ]
}
# 【保留】自动计算特征维度，一字未改！
FEATURE_DIM = sum(len(v) for v in FEATURE_CONFIG.values())  # 仍为16维，索引不变

TIME_PERIOD_BINS = [570, 600, 870, 900]
PRICE_CLIP_RANGE = (-0.05, 0.05)
RETURN_CLIP_RANGE = (-0.1, 0.1)
SAFE_DIV = 1e-6

# 模型参数
LGB_PARAMS = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'learning_rate': 0.005,  # 从0.015降到0.005，放慢学习
    'num_leaves': 4,         # 从8降到4，限制树复杂度
    'max_depth': 2,          # 从3降到2，树最多2层
    'min_child_samples': 100,# 从50升到100，要求更多样本才分裂
    'subsample': 0.7,        # 从0.8降到0.7，随机采样行
    'colsample_bytree': 0.4, # 从0.5降到0.4，随机采样列
    'reg_alpha': 0.5,        # 从0.3升到0.5，加强L1正则
    'reg_lambda': 0.5,       # 从0.3升到0.5，加强L2正则
    'n_estimators': 200,     # 从500降到200，减少树数量
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}

# 路径配置（自动创建目录）
DATA_PATH = "./data"
OUTPUT_DIR = "./output"
MODEL_DIR = "./model_weights"
CV_SPLITS = 3
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 警告过滤（保留）=====================
def filter_warnings():
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')
    warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===================== 工具函数（仅修复clean_numeric_array）=====================
def clean_numeric_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    # 修复：填充均值而非0，避免常量特征
    arr = np.nan_to_num(arr, nan=np.nanmean(arr) if not np.isnan(np.nanmean(arr)) else 0.0, posinf=0.0, neginf=0.0)
    return arr

def calculate_safe_return(current: np.ndarray, last: np.ndarray) -> np.ndarray:
    current = clean_numeric_array(current)
    last = clean_numeric_array(last)
    ret = (current - last) / (last + SAFE_DIV)
    ret = np.clip(ret, *RETURN_CLIP_RANGE)
    return clean_numeric_array(ret)

def calc_series_diff(arr: np.ndarray) -> np.ndarray:
    arr = clean_numeric_array(arr)
    if len(arr) <= 1:
        return np.zeros_like(arr)
    diff = np.zeros_like(arr)
    diff[1:] = arr[1:] - arr[:-1]
    return diff

def calc_series_return(arr: np.ndarray) -> np.ndarray:
    arr = clean_numeric_array(arr)
    if len(arr) <= 1:
        return np.zeros_like(arr)
    ret = np.zeros_like(arr)
    ret[1:] = calculate_safe_return(arr[1:], arr[:-1])
    return ret

# ===================== 数据加载（完全保留）=====================
def get_day_folders(data_path: str = DATA_PATH) -> list:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据目录 {data_path} 不存在")
    days = [
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and d.strip().isdigit()
    ]
    days.sort(key=lambda x: int(x))
    if not days:
        raise ValueError(f"{data_path} 下无交易日文件夹")
    return days

def load_day_data(day: str, data_path: str = DATA_PATH) -> dict:
    day_path = os.path.join(data_path, day)
    required = ["A.csv", "B.csv", "C.csv", "D.csv", "E.csv"]
    out = {}
    for fname in required:
        fp = os.path.join(day_path, fname)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"缺失 {fp}")
        try:
            df = pd.read_csv(fp, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding="gbk")
        core = ["LastPrice", "TradeBuyVolume", "TradeSellVolume", "Return5min"]
        for c in core:
            if c not in df.columns:
                raise ValueError(f"{fp} 缺少列 {c}")
        for col in core:
            df[col] = clean_numeric_array(df[col].values)
        if "time_num" in df.columns:
            df["time_period"] = parse_time_num_batch(df["time_num"].values)
        else:
            df["time_period"] = np.ones(len(df), dtype=np.float32)
        df["time_period"] = clean_numeric_array(df["time_period"].values)
        df = df.dropna(how="all").reset_index(drop=True)
        out[fname.split(".")[0]] = df
    return out

def parse_time_num_batch(time_nums):
    time_nums = clean_numeric_array(time_nums)
    s = np.char.zfill(time_nums.astype(str), 9)
    h = np.vectorize(lambda x: int(x[:2]) if len(x)>=2 else 0)(s)
    m = np.vectorize(lambda x: int(x[2:4]) if len(x)>=4 else 0)(s)
    tm = h * 60 + m
    out = np.ones_like(tm, dtype=np.float32)
    out[(tm >= TIME_PERIOD_BINS[0]) & (tm < TIME_PERIOD_BINS[1])] = 0.0
    out[(tm >= TIME_PERIOD_BINS[2]) & (tm < TIME_PERIOD_BINS[3])] = 2.0
    return clean_numeric_array(out)

# ===================== 批量特征计算（核心修改：精准取反+放大+屏蔽）=====================
def calculate_batch_features(df_E, df_A, df_B, df_C, df_D) -> tuple[np.ndarray, tuple]:
    n = len(df_E)
    feat = np.zeros((n, FEATURE_DIM), dtype=np.float64)
    feat_idx = 0
    # 基础数据提取
    e_p = df_E["LastPrice"].values
    e_buy = df_E["TradeBuyVolume"].values
    e_sell = df_E["TradeSellVolume"].values
    e_tp = df_E["time_period"].values
    e_vol = e_buy + e_sell
    e_order_flow = e_buy - e_sell
    # 板块数据提取
    a_p, b_p, c_p, d_p = [df["LastPrice"].values for df in [df_A, df_B, df_C, df_D]]
    a_of, b_of, c_of, d_of = [df["TradeBuyVolume"].values - df["TradeSellVolume"].values for df in [df_A, df_B, df_C, df_D]]
    sector_of = (a_of + b_of + c_of + d_of) / 4
    sector_of = clean_numeric_array(sector_of)

    # 1. 基础特征
    # vol_ratio（索引0）：放大3倍，强化唯一有效信号
    feat[:, feat_idx] = (e_vol / (e_vol + SAFE_DIV)) * 3  # 核心修改：乘以3放大
    feat_idx += 1
    # price_ma5_rel（索引1）：保留原计算，后续取反
    ###feat[:, feat_idx] = - ((e_p - pd.Series(e_p).rolling(window=5, min_periods=1, closed='left').mean().values) / (pd.Series(e_p).rolling(window=5, min_periods=1, closed='left').mean().values + SAFE_DIV))
    feat[:, feat_idx] = 0.0  # 核心修改：赋值为0，屏蔽无效信号
    feat_idx += 1
    # order_flow_ratio（索引2）
    feat[:, feat_idx] = (e_order_flow / (e_vol + SAFE_DIV))
    feat_idx += 1

    # 2. 差分特征（3-5）→ 全设为0
    for _ in range(3):
        feat[:, feat_idx] = 0.0
        feat_idx += 1

    # 3. 其他特征（索引6）：time_period → 赋值为0，屏蔽无效信号
    feat[:, feat_idx] = 0.0  # 核心修改：改为0
    feat_idx += 1

    """
    # 4. Alpha特征
    # order_flow_rsi（索引7）
    feat[:, feat_idx] = - (e_order_flow - sector_of)
    feat_idx += 1
    # return_residual（索引8）：赋值为0，屏蔽无效信号
    feat[:, feat_idx] = 0.0  # 核心修改：改为0
    feat_idx += 1
    # A_1min_return（索引9）：保留原计算，后续取反
    a_ret = calc_series_return(a_p)
    feat[:, feat_idx] = - a_ret
    feat_idx += 1
    # B_1min_return（索引10）：保留原计算，后续取反
    b_ret = calc_series_return(b_p)
    feat[:, feat_idx] = - b_ret
    feat_idx += 1
    # C_1min_return（索引11）
    c_ret = calc_series_return(c_p)
    feat[:, feat_idx] = - c_ret
    feat_idx += 1
    # D_1min_return（索引12）：保留原计算，后续取反
    d_ret = calc_series_return(d_p)
    feat[:, feat_idx] = - d_ret
    feat_idx += 1

    """
    # 4. Alpha特征（7-12）→ 全设为0
    for _ in range(6):
        feat[:, feat_idx] = 0.0
        feat_idx += 1

    # 5. 新增有效特征（索引13-15）
    e_order_imbalance = (e_buy - e_sell) / (e_buy + e_sell + SAFE_DIV)
    feat[:, feat_idx] = e_order_imbalance * 2
    feat_idx += 1
    e_vol_std = pd.Series(e_vol).rolling(window=10, min_periods=1).std().values
    feat[:, feat_idx] = clean_numeric_array(e_vol_std) * 2
    feat_idx += 1
    """
    e_price_trend = calc_series_return(e_p)
    e_vol_trend = calc_series_diff(e_vol)
    price_vol_trend = e_price_trend * e_vol_trend
    feat[:, feat_idx] = - price_vol_trend * 2
    """
    feat[:, feat_idx] = 0.0
    feat_idx += 1

    # 特征清洗
    feat = clean_numeric_array(feat)
    feat = np.clip(feat, -10, 10)
    feat = feat.astype(np.float32)
    # 维度校验
    if feat_idx != FEATURE_DIM:
        raise ValueError(f"特征维度错误：实际{feat_idx}维，配置{FEATURE_DIM}维")

    # 核心修改：精准取反4个反向特征（索引1/9/10/12），完全匹配你的IC结果
    reverse_feat_idx = [1,9,10,12]  # price_ma5_rel/A/B/D_1min_return
    feat[:, reverse_feat_idx] = -feat[:, reverse_feat_idx]

    return feat, (0.0, 0.0)  # reg_coef占位，不影响

# ===================== 在线特征计算（和批量特征同步修改）=====================
def calculate_online_feature(
    E_row,
    sector_rows,
    last_vol: float,
    last_p: float,
    last_sector_p: dict,
    reg_coef: tuple
) -> tuple[np.ndarray, dict]:
    # 基础数据提取
    e_p = clean_numeric_array([E_row["LastPrice"]])[0] or SAFE_DIV
    e_buy = clean_numeric_array([E_row["TradeBuyVolume"]])[0] or 0.0
    e_sell = clean_numeric_array([E_row["TradeSellVolume"]])[0] or 0.0
    e_tp = clean_numeric_array([E_row["time_period"]])[0] or 1.0
    e_vol = e_buy + e_sell
    e_order_flow = e_buy - e_sell
    sector_names = ["A", "B", "C", "D"]
    feat = np.zeros(FEATURE_DIM, dtype=np.float64)
    feat_idx = 0

    # 1. 基础特征
    feat[feat_idx] = (e_vol / (e_vol + SAFE_DIV)) * 3  # 同步放大3倍
    feat_idx += 1
    e_p_ma5 = (e_p + last_p * 4) / 5
    feat[feat_idx] = ((e_p - e_p_ma5) / (e_p_ma5 + SAFE_DIV))
    feat_idx += 1
    feat[feat_idx] = (e_order_flow / (e_vol + SAFE_DIV))
    feat_idx += 1

    # 2. 差分特征
    feat[feat_idx] = (e_vol - last_vol if last_vol != 0 else 0.0)
    feat_idx += 1
    feat[feat_idx] = (e_p - last_p if last_p != 0 else 0.0)
    feat_idx += 1
    e_ret = calculate_safe_return(np.array([e_p]), np.array([last_p]))[0] if last_p != 0 else 0.0
    feat[feat_idx] = e_ret
    feat_idx += 1

    # 3. 其他特征：time_period → 0
    feat[feat_idx] = 0.0
    feat_idx += 1

    # 4. Alpha特征
    sector_of = 0.0
    current_sector_p = {}
    for idx, s_name in enumerate(sector_names):
        s_row = sector_rows[idx]
        s_buy = clean_numeric_array([s_row["TradeBuyVolume"]])[0] or 0.0
        s_sell = clean_numeric_array([s_row["TradeSellVolume"]])[0] or 0.0
        sector_of += (s_buy - s_sell)
        current_sector_p[s_name] = clean_numeric_array([s_row["LastPrice"]])[0] or SAFE_DIV
    sector_of /= 4
    feat[feat_idx] = (e_order_flow - sector_of)
    feat_idx += 1
    # return_residual → 0
    feat[feat_idx] = 0.0
    feat_idx += 1

    # A/B/C/D_1min_return
    for s_name in sector_names:
        s_last_p = last_sector_p.get(s_name, SAFE_DIV)
        s_p = current_sector_p[s_name]
        s_ret = calculate_safe_return(np.array([s_p]), np.array([s_last_p]))[0]
        feat[feat_idx] = s_ret
        feat_idx += 1

    # 5. 新增有效特征
    e_order_imbalance = (e_buy - e_sell) / (e_buy + e_sell + SAFE_DIV)
    feat[feat_idx] = e_order_imbalance * 2
    feat_idx += 1
    e_vol_std = e_vol / (last_vol + SAFE_DIV) if last_vol != 0 else 0.0
    feat[feat_idx] = e_vol_std * 2
    feat_idx += 1
    e_price_trend = calculate_safe_return(np.array([e_p]), np.array([last_p]))[0] if last_p != 0 else 0.0
    e_vol_trend = e_vol - last_vol if last_vol != 0 else 0.0
    price_vol_trend = e_price_trend * e_vol_trend
    feat[feat_idx] = price_vol_trend * 2
    feat_idx += 1

    # 特征清洗
    feat = clean_numeric_array(feat)
    feat = np.clip(feat, -10, 10)
    feat = feat.astype(np.float32)
    if feat_idx != FEATURE_DIM:
        raise ValueError(f"在线特征维度错误：实际{feat_idx}维，配置{FEATURE_DIM}维")

    # 同步精准取反4个反向特征
    reverse_feat_idx = [1,9,10,12]
    feat[reverse_feat_idx] = -feat[reverse_feat_idx]

    return feat, current_sector_p

# ===================== IC评估（修复常量判断）=====================
def evaluate_ic(y_true, y_pred) -> float:
    y_true = clean_numeric_array(y_true)
    y_pred = clean_numeric_array(y_pred)
    min_len = min(len(y_true), len(y_pred))
    if min_len < 2:
        return 0.0
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    # 用方差判断常量，避免精度问题
    if np.var(y_true) < 1e-8 or np.var(y_pred) < 1e-8:
        return 0.0
    ic, p_value = pearsonr(y_true, y_pred)
    return float(ic) if not np.isnan(ic) else 0.0