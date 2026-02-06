import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# ===================== 全局常量（唯一入口）=====================
FEATURE_CONFIG = {
    "base": ["vol_ratio1", "price_ratio", "vol_ratio2"],
    "diff": ["vol_diff", "price_diff", "vol_diff2"],
    "other": ["time_period"],
    "alpha": [
        "order_flow_rsi",    # 相对强弱（订单流）
        "return_residual",   # 残差动量
        "A_1min_return",     # 领滞后（A的1分钟收益率）
        "B_1min_return",     # 领滞后（B的1分钟收益率）
        "C_1min_return",     # 领滞后（C的1分钟收益率）
        "D_1min_return"      # 领滞后（D的1分钟收益率）
    ]
}
FEATURE_DIM = sum(len(v) for v in FEATURE_CONFIG.values())

TIME_PERIOD_BINS = [570, 600, 870, 900]
PRICE_CLIP_RANGE = (-0.05, 0.05)
# 新增：收益率极值限制（避免过大/过小值）
RETURN_CLIP_RANGE = (-0.1, 0.1)
# 新增：安全除数（比1e-8更大，避免极小数做分母）
SAFE_DIV = 1e-6

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

# ===================== 新增：数据清洗工具（核心解决inf问题）=====================
def clean_numeric_array(arr: np.ndarray) -> np.ndarray:
    """
    清洗数值数组：替换inf/nan为0，限制极值在合理范围
    :param arr: 输入数组（任意维度）
    :return: 清洗后的float64数组
    """
    arr = arr.astype(np.float64)  # 转64位避免精度溢出
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def calculate_safe_return(current: np.ndarray, last: np.ndarray) -> np.ndarray:
    """
    安全计算收益率：(当前-上一期)/上一期，避免分母为0/inf/nan
    :param current: 当前值数组
    :param last: 上一期值数组
    :return: 清洗后的收益率数组（float64）
    """
    # 转为64位，初始化收益率
    current = clean_numeric_array(current)
    last = clean_numeric_array(last)
    ret = (current - last) / (last + SAFE_DIV)  # 加大安全除数
    # 限制收益率极值+最终清洗
    ret = np.clip(ret, *RETURN_CLIP_RANGE)
    ret = clean_numeric_array(ret)
    return ret

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
        # 加载时直接清洗核心列，避免后续计算出问题
        for col in core:
            df[col] = clean_numeric_array(df[col].values)
        if "time_period" not in df.columns:
            df["time_period"] = 1.0
        else:
            df["time_period"] = clean_numeric_array(df["time_period"].values)
        df = df.dropna(how="all").reset_index(drop=True)
        out[fname.split(".")[0]] = df
    return out

# ===================== 批量特征计算（训练阶段用）=====================
def calculate_batch_features(df_E, df_A, df_B, df_C, df_D) -> tuple[np.ndarray, tuple]:
    """
    训练阶段批量计算特征，返回特征矩阵+回归系数（截距、斜率）
    修复：收益率inf问题、精度溢出、首行数据初始化
    """
    n = len(df_E)
    feat = np.zeros((n, FEATURE_DIM), dtype=np.float64)  # 先64位计算，最后转32位
    
    # 基础数据提取（已在load_day_data中清洗，直接取值）
    p = df_E["LastPrice"].values
    buy = df_E["TradeBuyVolume"].values
    sell = df_E["TradeSellVolume"].values
    tp = df_E["time_period"].values
    vol = buy + sell
    safe = vol + SAFE_DIV  # 替换1e-8为更大的安全除数

    # 映射特征位置（按FEATURE_CONFIG顺序）
    feat_idx = 0
    # 1. 基础特征
    feat[:, feat_idx] = vol / safe  # vol_ratio1
    feat_idx += 1
    feat[:, feat_idx] = (p / (p + SAFE_DIV)) - 1.0  # price_ratio
    feat_idx += 1
    feat[:, feat_idx] = vol / safe  # vol_ratio2
    feat_idx += 1

    # 2. 差分特征（首行保持0，无前置数据）
    feat[1:, feat_idx] = vol[1:] - vol[:-1]  # vol_diff
    feat_idx += 1
    feat[1:, feat_idx] = p[1:] - p[:-1]      # price_diff
    feat_idx += 1
    feat[1:, feat_idx] = vol[1:] - vol[:-1]  # vol_diff2
    feat_idx += 1

    # 3. 其他特征
    feat[:, feat_idx] = tp  # time_period
    feat_idx += 1

    # ------------------- 新增：相对强弱与偏离特征（核心修复）-------------------
    # 1. 相对强弱（订单流：TradeBuyVolume - TradeSellVolume）
    e_order_flow = buy - sell
    sector_order_flow = (df_A["TradeBuyVolume"] - df_A["TradeSellVolume"] + 
                         df_B["TradeBuyVolume"] - df_B["TradeSellVolume"] + 
                         df_C["TradeBuyVolume"] - df_C["TradeSellVolume"] + 
                         df_D["TradeBuyVolume"] - df_D["TradeSellVolume"]) / 4
    sector_order_flow = clean_numeric_array(sector_order_flow.values)
    feat[:, feat_idx] = e_order_flow - sector_order_flow  # order_flow_rsi
    feat_idx += 1

    # 2. 残差动量（E收益率 - 板块收益率的回归残差）【核心修复：安全计算收益率】
    # 安全计算E的1期收益率，首行置0
    e_return = np.zeros(n, dtype=np.float64)
    e_return[1:] = calculate_safe_return(p[1:], p[:-1])
    # 安全计算板块各标的收益率，再取平均
    a_return = np.zeros(n, dtype=np.float64)
    b_return = np.zeros(n, dtype=np.float64)
    c_return = np.zeros(n, dtype=np.float64)
    d_return = np.zeros(n, dtype=np.float64)
    a_p = df_A["LastPrice"].values
    b_p = df_B["LastPrice"].values
    c_p = df_C["LastPrice"].values
    d_p = df_D["LastPrice"].values
    a_return[1:] = calculate_safe_return(a_p[1:], a_p[:-1])
    b_return[1:] = calculate_safe_return(b_p[1:], b_p[:-1])
    c_return[1:] = calculate_safe_return(c_p[1:], c_p[:-1])
    d_return[1:] = calculate_safe_return(d_p[1:], d_p[:-1])
    # 板块平均收益率
    sector_return = (a_return + b_return + c_return + d_return) / 4
    sector_return = clean_numeric_array(sector_return)

    # 回归训练前：最后校验输入数据（无inf/nan）【兜底修复】
    X_reg = sector_return.reshape(-1, 1)
    y_reg = e_return
    if not np.all(np.isfinite(X_reg)) or not np.all(np.isfinite(y_reg)):
        X_reg = clean_numeric_array(X_reg)
        y_reg = clean_numeric_array(y_reg)

    # 训练回归模型（获取截距、斜率）
    reg = LinearRegression().fit(X_reg, y_reg)
    reg_coef = (float(reg.intercept_), float(reg.coef_[0]))
    pred_e = reg.predict(X_reg)
    feat[:, feat_idx] = e_return - pred_e  # return_residual
    feat_idx += 1

    # 3. 领滞后关系（A/B/C/D的过去1分钟收益率）【复用安全计算的收益率】
    feat[:, feat_idx] = a_return  # A_1min_return
    feat_idx += 1
    feat[:, feat_idx] = b_return  # B_1min_return
    feat_idx += 1
    feat[:, feat_idx] = c_return  # C_1min_return
    feat_idx += 1
    feat[:, feat_idx] = d_return  # D_1min_return
    feat_idx += 1

    # 最终清洗+转float32
    feat = clean_numeric_array(feat)
    feat = feat.astype(np.float32)
    return feat, reg_coef

# ===================== 在线特征计算（预测阶段用）=====================
def calculate_online_feature(
    E_row, 
    sector_rows,  # A/B/C/D行数据
    last_vol: float, 
    last_p: float,
    last_sector_p: dict,  # 上一轮板块价格 {A:..., B:..., C:..., D:...}
    reg_coef: tuple  # 回归系数 (intercept, coef)
) -> tuple[np.ndarray, dict]:
    """
    在线预测单条Tick特征计算
    修复：收益率inf、精度溢出问题
    """
    # 基础数据提取（带异常值处理）
    p = E_row["LastPrice"] if not np.isnan(E_row["LastPrice"]) else SAFE_DIV
    buy = E_row["TradeBuyVolume"] if not np.isnan(E_row["TradeBuyVolume"]) else 0
    sell = E_row["TradeSellVolume"] if not np.isnan(E_row["TradeSellVolume"]) else 0
    tp = E_row.get("time_period", 1.0)
    vol = buy + sell
    safe = vol + SAFE_DIV

    # 初始化特征向量（先64位，最后转32）
    feat = np.zeros(FEATURE_DIM, dtype=np.float64)
    feat_idx = 0

    # 1. 基础特征
    feat[feat_idx] = vol / safe  # vol_ratio1
    feat_idx += 1
    feat[feat_idx] = (p / (p + SAFE_DIV)) - 1.0  # price_ratio
    feat_idx += 1
    feat[feat_idx] = vol / safe  # vol_ratio2
    feat_idx += 1

    # 2. 差分特征（首Tick置0）
    feat[feat_idx] = vol - last_vol if last_vol != 0 else 0.0  # vol_diff
    feat_idx += 1
    feat[feat_idx] = p - last_p if last_p != 0 else 0.0        # price_diff
    feat_idx += 1
    feat[feat_idx] = vol - last_vol if last_vol != 0 else 0.0  # vol_diff2
    feat_idx += 1

    # 3. 其他特征
    feat[feat_idx] = tp  # time_period
    feat_idx += 1

    # ------------------- 新增特征（修复版）-------------------
    # 1. 相对强弱（订单流）
    e_order_flow = buy - sell
    sector_order_flow = 0.0
    for s_row in sector_rows:
        s_buy = s_row["TradeBuyVolume"] if not np.isnan(s_row["TradeBuyVolume"]) else 0
        s_sell = s_row["TradeSellVolume"] if not np.isnan(s_row["TradeSellVolume"]) else 0
        sector_order_flow += (s_buy - s_sell)
    sector_order_flow /= 4
    feat[feat_idx] = e_order_flow - sector_order_flow  # order_flow_rsi
    feat_idx += 1

    # 2. 残差动量（用训练好的回归系数+安全收益率）
    # 安全计算E的当前收益率
    e_return = calculate_safe_return(np.array([p]), np.array([last_p]))[0] if last_p != 0 else 0.0
    # 计算板块收益率
    current_sector_p = {}
    sector_return = 0.0
    sector_names = ["A", "B", "C", "D"]
    for idx, s_name in enumerate(sector_names):
        s_row = sector_rows[idx]
        s_p = s_row["LastPrice"] if not np.isnan(s_row["LastPrice"]) else SAFE_DIV
        current_sector_p[s_name] = s_p
        s_last_p = last_sector_p.get(s_name, SAFE_DIV)
        # 安全计算单板块收益率
        s_ret = calculate_safe_return(np.array([s_p]), np.array([s_last_p]))[0]
        sector_return += s_ret
    sector_return /= 4
    # 计算残差
    intercept, coef = reg_coef
    pred_e = intercept + coef * sector_return
    feat[feat_idx] = e_return - pred_e  # return_residual
    feat_idx += 1

    # 3. 领滞后关系（A/B/C/D的1分钟收益率）
    for s_name in sector_names:
        s_last_p = last_sector_p.get(s_name, SAFE_DIV)
        s_p = current_sector_p[s_name]
        s_ret = calculate_safe_return(np.array([s_p]), np.array([s_last_p]))[0]
        feat[feat_idx] = s_ret
        feat_idx += 1

    # 最终清洗+转float32
    feat = clean_numeric_array(feat)
    feat = feat.astype(np.float32)
    return feat, current_sector_p

# ===================== IC 评估 =====================
def evaluate_ic(y_true, y_pred) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true/y_pred 长度不匹配")
    if len(y_true) < 2:
        return 0.0
    # 评估前清洗数据，避免inf/nan影响IC计算
    y_true = clean_numeric_array(y_true)
    y_pred = clean_numeric_array(y_pred)
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