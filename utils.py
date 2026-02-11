import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

# ===================== 全局常量（适配0.05s/Tick）=====================
FEATURE_CONFIG = {
    "core_positive": ["vol_ratio", "weighted_sector_ret", "vol_speed"]  # 仅保留3个正向因子
}
FEATURE_DIM = sum(len(v) for v in FEATURE_CONFIG.values()) # 3维特征

# 适配0.05s/Tick的窗口参数
TICK_PER_SECOND = 20  # 1秒=20个Tick
TICK_PER_MINUTE = TICK_PER_SECOND * 60  # 1分钟=120个Tick
TICK_PER_5MIN = TICK_PER_MINUTE * 5     # 5分钟=600个Tick

# 因子计算窗口（核心修正）
VOL_VOLATILITY_WINDOW = TICK_PER_MINUTE  # 120个Tick（1min）
PRICE_VOL_SPEED_WINDOW = TICK_PER_5MIN   # 600个Tick（5min）
DYNAMIC_WEIGHT_WINDOW = TICK_PER_5MIN // 2  # 300个Tick（2.5min）

# 其他常量（不变）
TIME_PERIOD_BINS = [570, 600, 870, 900]
PRICE_CLIP_RANGE = (-0.05, 0.05)
RETURN_CLIP_RANGE = (-0.1, 0.1)
SAFE_DIV = 1e-6

# LGB参数（小幅放宽，适配有效信号）
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,    # 从0.005→0.01，适配强信号
    'num_leaves': 6,          # 从4→6，轻微提升拟合
    'max_depth': 3,           # 从2→3
    'min_child_samples': 50,  # 从100→50
    'subsample': 0.7,
    'colsample_bytree': 0.6,  # 从0.4→0.6，保留更多有效因子
    'reg_alpha': 0.3,         # 从0.5→0.3，适度降低正则
    'reg_lambda': 0.3,
    'n_estimators': 300,      # 从200→300
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}

# 路径配置
DATA_PATH = "./data"
DATA_DIR = DATA_PATH
OUTPUT_DIR = "./output"
MODEL_DIR = "./model_weights"
CV_SPLITS = 3
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def calculate_dynamic_weight(
    e_return: np.ndarray,
    a_return: np.ndarray,
    b_return: np.ndarray,
    c_return: np.ndarray,
    d_return: np.ndarray
) -> tuple:
    """适配0.05s/Tick：滚动窗口300个Tick（2.5min）"""
    window = DYNAMIC_WEIGHT_WINDOW  # 300个Tick
    e_return = clean_numeric_array(e_return)
    a_return = clean_numeric_array(a_return)
    b_return = clean_numeric_array(b_return)
    c_return = clean_numeric_array(c_return)
    d_return = clean_numeric_array(d_return)
    
    w_a = np.zeros_like(e_return)
    w_b = np.zeros_like(e_return)
    w_c = np.zeros_like(e_return)
    w_d = np.zeros_like(e_return)
    
    for i in range(window, len(e_return)):
        e_win = e_return[i-window:i]
        a_win = a_return[i-window:i]
        b_win = b_return[i-window:i]
        c_win = c_return[i-window:i]
        d_win = d_return[i-window:i]
        
        # 计算IC（相关系数）
        if np.var(a_win) > 1e-8 and np.var(e_win) > 1e-8:
            w_a[i] = abs(pearsonr(e_win, a_win)[0])
        if np.var(b_win) > 1e-8 and np.var(e_win) > 1e-8:
            w_b[i] = abs(pearsonr(e_win, b_win)[0])
        if np.var(c_win) > 1e-8 and np.var(e_win) > 1e-8:
            w_c[i] = abs(pearsonr(e_win, c_win)[0])
        if np.var(d_win) > 1e-8 and np.var(e_win) > 1e-8:
            w_d[i] = abs(pearsonr(e_win, d_win)[0])
    
    # 权重归一化
    total_w = w_a + w_b + w_c + w_d + SAFE_DIV
    w_a = w_a / total_w
    w_b = w_b / total_w
    w_c = w_c / total_w
    w_d = w_d / total_w
    
    return w_a, w_b, w_c, w_d

# ===================== 原有工具函数（完全保留）=====================
def filter_warnings():
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')
    warnings.filterwarnings('ignore', category=RuntimeWarning)

def clean_numeric_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
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

def evaluate_ic(y_true, y_pred) -> float:
    y_true = clean_numeric_array(y_true)
    y_pred = clean_numeric_array(y_pred)
    min_len = min(len(y_true), len(y_pred))
    if min_len < 2:
        return 0.0
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    if np.var(y_true) < 1e-8 or np.var(y_pred) < 1e-8:
        return 0.0
    ic, p_value = pearsonr(y_true, y_pred)
    return float(ic) if not np.isnan(ic) else 0.0

# ===================== 批量特征计算 =====================
def calculate_batch_features(df_E, df_A, df_B, df_C, df_D) -> tuple[np.ndarray, tuple]:
    n = len(df_E)
    feat = np.zeros((n, FEATURE_DIM), dtype=np.float64)
    feat_idx = 0
    
    # 基础数据提取
    e_p = df_E["LastPrice"].values
    e_buy = df_E["TradeBuyVolume"].values
    e_sell = df_E["TradeSellVolume"].values
    e_vol = e_buy + e_sell
    e_return = df_E["Return5min"].values
    
    # 板块数据提取
    a_p, b_p, c_p, d_p = [df["LastPrice"].values for df in [df_A, df_B, df_C, df_D]]
    a_ret = calc_series_return(a_p)
    b_ret = calc_series_return(b_p)
    c_ret = calc_series_return(c_p)
    d_ret = calc_series_return(d_p)
    
    # 1. vol_ratio（放大30倍）
    feat[:, feat_idx] = (e_vol / (e_vol + SAFE_DIV)) * 30
    feat_idx += 1
    
    # 2. weighted_sector_ret（放大15倍）
    w_a, w_b, w_c, w_d = calculate_dynamic_weight(e_return, a_ret, b_ret, c_ret, d_ret)
    weighted_sector_ret = w_a * a_ret + w_b * b_ret + w_c * c_ret + w_d * d_ret
    weighted_sector_ret = clean_numeric_array(weighted_sector_ret)
    feat[:, feat_idx] = - weighted_sector_ret * 15
    feat_idx += 1
    
    # 3. vol_speed（放大150倍）
    def calc_speed(arr, window):
        arr = clean_numeric_array(arr)
        speed = np.zeros_like(arr)
        for i in range(window, len(arr)):
            x = np.arange(window).reshape(-1, 1)
            y = arr[i-window:i]
            if np.var(y) < 1e-8:
                speed[i] = 0.0
            else:
                lr = LinearRegression().fit(x, y)
                speed[i] = lr.coef_[0]
        return speed
    
    vol_speed = calc_speed(e_vol, window=PRICE_VOL_SPEED_WINDOW)
    feat[:, feat_idx] = clean_numeric_array(vol_speed) * 150
    feat_idx += 1
    
    # 特征清洗
    feat = clean_numeric_array(feat)
    feat = np.clip(feat, -10, 10)
    feat = feat.astype(np.float32)
    
    if feat_idx != FEATURE_DIM:
        raise ValueError(f"特征维度错误：实际{feat_idx}维，配置{FEATURE_DIM}维")
    
    return feat, (0.0, 0.0)

# ===================== 在线特征计算（补充price_speed+vol_speed）=====================
def calculate_online_feature(
    E_row,
    sector_rows,
    last_vol: float,
    last_p: float,
    last_sector_p: dict,
    reg_coef: tuple
) -> tuple[np.ndarray, dict]:
    """
    在线特征计算（适配0.05s/Tick + 仅保留3个正向因子）
    参数：
        E_row: E股当前Tick行数据
        sector_rows: [A/B/C/D股当前Tick行数据]
        last_vol: E股上一Tick成交量
        last_p: E股上一Tick价格
        last_sector_p: 上一Tick A/B/C/D股价格字典
        reg_coef: 预留回归系数（无实际作用）
    返回：
        feat: 3维特征数组
        current_sector_p: 当前Tick A/B/C/D股价格字典
    """
    # 初始化全局变量（避免重复定义）
    global sector_weight, tick_count, history_data
    if 'sector_weight' not in globals():
        # 初始等权
        sector_weight = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        tick_count = 0
        # 历史数据缓存（适配0.05s/Tick，保留足够窗口）
        history_data = {
            "e_return": [], "a_ret": [], "b_ret": [], "c_ret": [], "d_ret": [],
            "e_p": [], "e_vol": []
        }

    tick_count += 1

    # ===================== 1. 基础数据提取与清洗 =====================
    # E股当前Tick数据
    e_p = clean_numeric_array([E_row["LastPrice"]])[0] or SAFE_DIV
    e_buy = clean_numeric_array([E_row["TradeBuyVolume"]])[0] or 0.0
    e_sell = clean_numeric_array([E_row["TradeSellVolume"]])[0] or 0.0
    e_vol = e_buy + e_sell
    e_return = clean_numeric_array([E_row["Return5min"]])[0] if "Return5min" in E_row else 0.0

    # 记录E股历史数据（保留5min=600个Tick）
    history_data["e_p"].append(e_p)
    history_data["e_vol"].append(e_vol)
    history_data["e_return"].append(e_return)
    # 窗口截断（避免内存溢出）
    history_data["e_p"] = history_data["e_p"][-PRICE_VOL_SPEED_WINDOW:]  # 600个Tick
    history_data["e_vol"] = history_data["e_vol"][-PRICE_VOL_SPEED_WINDOW:]
    history_data["e_return"] = history_data["e_return"][-DYNAMIC_WEIGHT_WINDOW:]  # 300个Tick

    # ===================== 2. 板块数据（A/B/C/D）处理 =====================
    sector_names = ["A", "B", "C", "D"]
    current_sector_p = {}  # 当前A/B/C/D价格
    sector_rets = {}       # 当前A/B/C/D 1min收益率

    for idx, s_name in enumerate(sector_names):
        # 提取当前板块股数据
        s_row = sector_rows[idx]
        s_p = clean_numeric_array([s_row["LastPrice"]])[0] or SAFE_DIV
        current_sector_p[s_name] = s_p

        # 计算1min收益率（基于上一Tick价格）
        last_s_p = last_sector_p.get(s_name, SAFE_DIV)
        s_ret = calculate_safe_return(np.array([s_p]), np.array([last_s_p]))[0]
        sector_rets[s_name] = s_ret

        # 记录板块历史收益率（用于动态权重计算）
        history_data[f"{s_name}_ret"].append(s_ret)
        history_data[f"{s_name}_ret"] = history_data[f"{s_name}_ret"][-DYNAMIC_WEIGHT_WINDOW:]  # 300个Tick

    # ===================== 3. 动态权重更新（每300个Tick更新一次） =====================
    if (tick_count % DYNAMIC_WEIGHT_WINDOW == 0) and (len(history_data["e_return"]) >= DYNAMIC_WEIGHT_WINDOW):
        # 提取历史数据数组
        e_return_arr = np.array(history_data["e_return"])
        a_ret_arr = np.array(history_data["a_ret"])
        b_ret_arr = np.array(history_data["b_ret"])
        c_ret_arr = np.array(history_data["c_ret"])
        d_ret_arr = np.array(history_data["d_ret"])

        # 计算动态权重
        w_a, w_b, w_c, w_d = calculate_dynamic_weight(e_return_arr, a_ret_arr, b_ret_arr, c_ret_arr, d_ret_arr)
        # 取平均权重（在线场景简化）
        sector_weight = {
            "A": np.mean(w_a) if len(w_a) > 0 else 0.25,
            "B": np.mean(w_b) if len(w_b) > 0 else 0.25,
            "C": np.mean(w_c) if len(w_c) > 0 else 0.25,
            "D": np.mean(w_d) if len(w_d) > 0 else 0.25
        }

    # ===================== 4. 特征计算（仅3个正向因子） =====================
    # 初始化3维特征数组
    feat = np.zeros(FEATURE_DIM, dtype=np.float64)
    feat_idx = 0

    # 因子1：vol_ratio（成交量比率，放大30倍）
    feat[feat_idx] = (e_vol / (e_vol + SAFE_DIV)) * 30
    feat_idx += 1

    # 因子2：weighted_sector_ret（动态权重板块收益率，放大15倍）
    weighted_sector_ret = (
        sector_weight["A"] * sector_rets["A"] +
        sector_weight["B"] * sector_rets["B"] +
        sector_weight["C"] * sector_rets["C"] +
        sector_weight["D"] * sector_rets["D"]
    )
    feat[feat_idx] = - weighted_sector_ret * 15  # 保留原有取反逻辑
    feat_idx += 1

    # 因子3：vol_speed（成交量涨速，5min窗口，放大150倍）
    def calc_online_speed(arr):
        """在线涨速计算（线性回归斜率）"""
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr)).reshape(-1, 1)
        y = np.array(arr)
        if np.var(y) < 1e-8:
            return 0.0
        lr = LinearRegression().fit(x, y)
        return lr.coef_[0]

    vol_speed = calc_online_speed(history_data["e_vol"]) * 150
    feat[feat_idx] = vol_speed
    feat_idx += 1

    # ===================== 5. 特征清洗与校验 =====================
    # 数据清洗（去极值、填NaN）
    feat = clean_numeric_array(feat)
    feat = np.clip(feat, -10, 10)  # 防止极端值
    feat = feat.astype(np.float32)

    # 维度校验（避免线上报错）
    if feat_idx != FEATURE_DIM:
        raise ValueError(f"在线特征维度错误：实际{feat_idx}维，配置{FEATURE_DIM}维")

    return feat, current_sector_p