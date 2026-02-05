import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ===================== 全局常量（唯一入口）=====================
# 特征维度：原有7维 + 预留扩展位（新增特征只需修改这里和特征计算函数）
FEATURE_CONFIG = {
    "base": ["vol_ratio1", "price_ratio", "vol_ratio2"],
    "diff": ["vol_diff", "price_diff", "vol_diff2"],
    "other": ["time_period"],
    # 新增特征直接加在对应分类下，例如：
    # "base": ["vol_ratio1", "price_ratio", "vol_ratio2", "new_base_feat"],
    # "custom": ["new_custom_feat1", "new_custom_feat2"]
}
# 自动计算总特征维度（无需手动改数字）
FEATURE_DIM = sum(len(v) for v in FEATURE_CONFIG.values())

TIME_PERIOD_BINS = [570, 600, 870, 900]
PRICE_CLIP_RANGE = (-0.05, 0.05)

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.08,
    'num_leaves': 20,
    'max_depth': 6,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
    'n_estimators': 100,
    'verbose': -1,
    'n_jobs': -1,
}

DATA_PATH = "./data"
MODEL_DIR = "./model_weights"
CV_SPLITS = 3

# ===================== 统一警告过滤 =====================
def filter_warnings():
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')

# ===================== 交易日文件夹 =====================
def get_day_folders(data_path: str = DATA_PATH) -> list:
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        raise FileNotFoundError(f"数据目录 {data_path} 不存在，已创建，请放入数据")
    days = [
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and d.strip().isdigit()
    ]
    days.sort(key=lambda x: int(x))
    if not days:
        raise ValueError(f"{data_path} 下无数字命名交易日文件夹")
    return days

# ===================== 单天数据加载 =====================
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
                raise ValueError(f"{fp} 缺少必需列 {c}")
        df["LastPrice"] = df["LastPrice"].astype(np.float32)
        df["TradeBuyVolume"] = df["TradeBuyVolume"].astype(np.int32)
        df["TradeSellVolume"] = df["TradeSellVolume"].astype(np.int32)
        df["Return5min"] = df["Return5min"].astype(np.float32)
        if "time_period" not in df.columns:
            df["time_period"] = 1.0
        else:
            df["time_period"] = df["time_period"].astype(np.float32)
        df = df.dropna(how="all").reset_index(drop=True)
        out[fname.split(".")[0]] = df
    return out

# ===================== 批量特征计算（训练阶段用）=====================
def calculate_batch_features(df_E, df_A, df_B, df_C, df_D) -> np.ndarray:
    """训练阶段批量计算特征（兼容原有逻辑）"""
    n = len(df_E)
    # 初始化特征矩阵（自动适配FEATURE_DIM）
    feat = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    
    # 基础数据提取
    p = df_E["LastPrice"].values
    buy = df_E["TradeBuyVolume"].values
    sell = df_E["TradeSellVolume"].values
    tp = df_E["time_period"].values
    vol = buy + sell
    safe = vol + 1e-8

    # 映射特征位置（按FEATURE_CONFIG顺序）
    feat_idx = 0
    # 1. 基础特征
    feat[:, feat_idx] = vol / safe  # vol_ratio1
    feat_idx += 1
    feat[:, feat_idx] = (p / (p + 1e-8)) - 1.0  # price_ratio
    feat_idx += 1
    feat[:, feat_idx] = vol / safe  # vol_ratio2
    feat_idx += 1

    # 2. 差分特征
    feat[1:, feat_idx] = vol[1:] - vol[:-1]  # vol_diff
    feat_idx += 1
    feat[1:, feat_idx] = p[1:] - p[:-1]      # price_diff
    feat_idx += 1
    feat[1:, feat_idx] = vol[1:] - vol[:-1]  # vol_diff2
    feat_idx += 1

    # 3. 其他特征
    feat[:, feat_idx] = tp  # time_period
    feat_idx += 1

    # 新增特征示例：直接在对应分类后加即可
    # feat[:, feat_idx] = xxx  # new_custom_feat1
    # feat_idx += 1

    # 异常值处理
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat

# ===================== 在线特征计算（预测阶段用）=====================
def calculate_online_feature(E_row, last_vol: float, last_p: float) -> np.ndarray:
    """
    在线预测单条Tick特征计算（抽离自MyModel）
    :param E_row: 当前Tick的E行数据（pd.Series）
    :param last_vol: 上一轮成交量（用于差分）
    :param last_p: 上一轮价格（用于差分）
    :return: 单条特征向量（1D np.array）
    """
    # 基础数据提取（带异常值处理）
    price_mean = 1e-8
    p = E_row["LastPrice"] if not np.isnan(E_row["LastPrice"]) else price_mean
    buy = E_row["TradeBuyVolume"] if not np.isnan(E_row["TradeBuyVolume"]) else 0
    sell = E_row["TradeSellVolume"] if not np.isnan(E_row["TradeSellVolume"]) else 0
    tp = E_row.get("time_period", 1.0)
    vol = buy + sell
    safe = vol + 1e-8

    # 初始化特征向量
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    feat_idx = 0

    # 1. 基础特征
    feat[feat_idx] = vol / safe  # vol_ratio1
    feat_idx += 1
    feat[feat_idx] = (p / (p + 1e-8)) - 1.0  # price_ratio
    feat_idx += 1
    feat[feat_idx] = vol / safe  # vol_ratio2
    feat_idx += 1

    # 2. 差分特征（用缓存的上一轮值）
    feat[feat_idx] = vol - last_vol  # vol_diff
    feat_idx += 1
    feat[feat_idx] = p - last_p      # price_diff
    feat_idx += 1
    feat[feat_idx] = vol - last_vol  # vol_diff2
    feat_idx += 1

    # 3. 其他特征
    feat[feat_idx] = tp  # time_period
    feat_idx += 1

    # 新增特征示例：
    # feat[feat_idx] = E_row.get("NewFeat", 0.0)  # 新增特征
    # feat_idx += 1

    # 异常值处理
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat

# ===================== IC 评估 =====================
def evaluate_ic(y_true, y_pred) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true/y_pred 长度不匹配")
    if len(y_true) < 2:
        return 0.0
    ic, _ = pearsonr(y_true, y_pred)
    return float(ic) if not np.isnan(ic) else 0.0

# ===================== 批量时间解析 =====================
def parse_time_num_batch(time_nums):
    s = np.char.zfill(time_nums.astype(str), 9)
    h = np.vectorize(lambda x: int(x[:2]))(s)
    m = np.vectorize(lambda x: int(x[2:4]))(s)
    tm = h * 60 + m
    out = np.ones_like(tm, dtype=np.float32)
    out[(tm >= 570) & (tm < 600)] = 0.0
    out[(tm >= 870) & (tm < 900)] = 2.0
    return out