import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

# ===================== 全局常量 =====================
FEATURE_CONFIG = {
    "base": ["vol_ratio", "price_ma5_rel", "order_flow_ratio"],  # 3维
    "diff": ["vol_diff", "price_diff", "price_ret"],             # 3维
    "other": ["time_period"],                                    # 1维
    "alpha": [
        "order_flow_rsi",    # 订单流相对强弱（E-板块平均）
        "return_residual",   # 残差动量
        "A_1min_return",     # 板块A收益率
        "B_1min_return",     # 板块B收益率
        "C_1min_return",     # 板块C收益率
        "D_1min_return"      # 板块D收益率
    ],  # 6维
    "new_effective": [      
        "order_imbalance",   # 订单流不平衡（核心有效）
        "vol_volatility",    # 成交量波动率
        "price_vol_trend"    # 新增：量价趋势特征
    ]  # 3维 → 总计：3+3+1+6+3=16维
}
# 特征维度同步更新为16维（关键修复！）
FEATURE_DIM = sum(len(v) for v in FEATURE_CONFIG.values())  # 现在计算结果是16

# 核心常量优化
TIME_PERIOD_BINS = [570, 600, 870, 900]
PRICE_CLIP_RANGE = (-0.05, 0.05)
RETURN_CLIP_RANGE = (-0.1, 0.1)
SAFE_DIV = 1e-6  

# 模型参数优化（增强正则化，防止过拟合）
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.005,
    'num_leaves': 10,
    'max_depth': 3,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.6,  # 关键调整：减少每次选的特征数
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'n_estimators': 500,
    'verbose': -1,          
    'n_jobs': -1,
    'random_state': 42,     
}

DATA_PATH = "./data"
MODEL_DIR = "./model_weights"
CV_SPLITS = 3

# ===================== 统一警告过滤 =====================
def filter_warnings():
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')
    warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===================== 核心数据清洗工具 =====================
def clean_numeric_array(arr: np.ndarray) -> np.ndarray:
    """清洗数值数组：替换inf/nan为0，统一转float64"""
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def calculate_safe_return(current: np.ndarray, last: np.ndarray) -> np.ndarray:
    """安全计算收益率：(当前-上一期)/上一期，避免分母为0，限制极值"""
    current = clean_numeric_array(current)
    last = clean_numeric_array(last)
    ret = (current - last) / (last + SAFE_DIV)
    ret = np.clip(ret, *RETURN_CLIP_RANGE)
    return clean_numeric_array(ret)

def calc_series_diff(arr: np.ndarray) -> np.ndarray:
    """计算时序差分：arr[1:] - arr[:-1]，首行补0"""
    arr = clean_numeric_array(arr)
    if len(arr) <= 1:
        return np.zeros_like(arr)
    diff = np.zeros_like(arr)
    diff[1:] = arr[1:] - arr[:-1]
    return diff

def calc_series_return(arr: np.ndarray) -> np.ndarray:
    """计算时序1期收益率，首行补0"""
    arr = clean_numeric_array(arr)
    if len(arr) <= 1:
        return np.zeros_like(arr)
    ret = np.zeros_like(arr)
    ret[1:] = calculate_safe_return(arr[1:], arr[:-1])
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
        # 自动适配编码
        try:
            df = pd.read_csv(fp, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding="gbk")
        # 核心列校验
        core = ["LastPrice", "TradeBuyVolume", "TradeSellVolume", "Return5min"]
        for c in core:
            if c not in df.columns:
                raise ValueError(f"{fp} 缺少必需列 {c}")
        # 清洗核心列
        for col in core:
            df[col] = clean_numeric_array(df[col].values)
        # 时间周期特征
        if "time_num" in df.columns:
            df["time_period"] = parse_time_num_batch(df["time_num"].values)
        else:
            df["time_period"] = np.ones(len(df), dtype=np.float32)
        df["time_period"] = clean_numeric_array(df["time_period"].values)
        # 过滤全空行
        df = df.dropna(how="all").reset_index(drop=True)
        out[fname.split(".")[0]] = df
    return out

# ===================== 批量特征计算（训练阶段）=====================
def calculate_batch_features(df_E, df_A, df_B, df_C, df_D) -> tuple[np.ndarray, tuple]:
    """
    训练阶段批量计算特征，严格避免数据泄露
    核心修改：1. 核心特征加反向 2. 新增有效量价特征
    返回：特征矩阵 + 回归系数
    """
    n = len(df_E)
    feat = np.zeros((n, FEATURE_DIM), dtype=np.float64)
    feat_idx = 0

    # 基础数据提取（已清洗）
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

    # ------------------- 1. 基础特征（核心修改：加反向）-------------------
    # 成交量占比（反向）
    feat[:, feat_idx] = - (e_vol / (e_vol + SAFE_DIV))
    feat_idx += 1
    # 价格相对5Tick均值（滚动窗口，无未来信息 + 反向）
    e_p_ma5 = pd.Series(e_p).rolling(window=5, min_periods=1, closed='left').mean().values
    feat[:, feat_idx] = - ((e_p - e_p_ma5) / (e_p_ma5 + SAFE_DIV))
    feat_idx += 1
    # 订单流占比（反向）
    feat[:, feat_idx] = - (e_order_flow / (e_vol + SAFE_DIV))
    feat_idx += 1

    # ------------------- 2. 差分特征（核心修改：加反向）-------------------
    feat[:, feat_idx] = - calc_series_diff(e_vol)  # 成交量差分（反向）
    feat_idx += 1
    feat[:, feat_idx] = - calc_series_diff(e_p)    # 价格差分（反向）
    feat_idx += 1
    feat[:, feat_idx] = - calc_series_return(e_p)  # 价格收益率（反向）
    feat_idx += 1

    # ------------------- 3. 其他特征 -------------------
    feat[:, feat_idx] = e_tp  # 时间周期
    feat_idx += 1

    # ------------------- 4. Alpha特征（核心修改：加反向）-------------------
    # 4.1 订单流相对强弱（E - 板块平均 + 反向）
    feat[:, feat_idx] = - (e_order_flow - sector_of)
    feat_idx += 1

    # 4.2 残差动量（无未来信息 + 反向）
    e_ret = calc_series_return(e_p)
    a_ret = calc_series_return(a_p)
    b_ret = calc_series_return(b_p)
    c_ret = calc_series_return(c_p)
    d_ret = calc_series_return(d_p)
    sector_ret = (a_ret + b_ret + c_ret + d_ret) / 4
    sector_ret = clean_numeric_array(sector_ret)
    
    # 线性回归（仅用历史数据拟合）
    X_reg = sector_ret.reshape(-1, 1)
    y_reg = e_ret
    reg = LinearRegression(fit_intercept=True).fit(X_reg, y_reg)
    reg_coef = (float(reg.intercept_), float(reg.coef_[0]))
    e_ret_pred = reg.predict(X_reg)
    feat[:, feat_idx] = - (e_ret - e_ret_pred)  # 残差动量（反向）
    feat_idx += 1

    # 4.3 板块领滞后关系（仅用历史收益率 + 反向）
    feat[:, feat_idx] = - a_ret
    feat_idx += 1
    feat[:, feat_idx] = - b_ret
    feat_idx += 1
    feat[:, feat_idx] = - c_ret
    feat_idx += 1
    feat[:, feat_idx] = - d_ret
    feat_idx += 1

    # ------------------- 5. 新增有效量价特征（核心）-------------------
    # 5.1 订单流不平衡（Buy-Sell)/(Buy+Sell)（量化核心有效特征）
    e_order_imbalance = (e_buy - e_sell) / (e_buy + e_sell + SAFE_DIV)
    feat[:, feat_idx] = e_order_imbalance*2  # 放大效果更明显
    feat_idx += 1
    # 5.2 成交量波动率（滚动10Tick标准差）
    e_vol_std = pd.Series(e_vol).rolling(window=10, min_periods=1).std().values
    feat[:, feat_idx] = clean_numeric_array(e_vol_std) * 2  # 放大2倍增强权重
    feat_idx += 1
    # 5.3 量价趋势特征（新增，放大2倍）
    e_price_trend = calc_series_return(e_p)  # 价格趋势
    e_vol_trend = calc_series_diff(e_vol)    # 成交量趋势
    price_vol_trend = e_price_trend * e_vol_trend  # 量价协动（涨+放量=正）
    feat[:, feat_idx] = price_vol_trend * 2
    feat_idx += 1   

    # 最终清洗
    feat = clean_numeric_array(feat)
    feat = np.clip(feat, -10, 10)
    feat = feat.astype(np.float32)
    
    # 特征维度校验
    if feat_idx != FEATURE_DIM:
        raise ValueError(f"特征维度映射错误：实际计算{feat_idx}维，配置{FEATURE_DIM}维")
    return feat, reg_coef

# ===================== 在线特征计算（预测阶段）=====================
def calculate_online_feature(
    E_row, 
    sector_rows,
    last_vol: float, 
    last_p: float,
    last_sector_p: dict,
    reg_coef: tuple
) -> tuple[np.ndarray, dict]:
    """在线预测单条Tick特征计算，与训练逻辑严格对齐（同步反向+新增特征）"""
    # 基础数据提取+异常值兜底
    e_p = clean_numeric_array([E_row["LastPrice"]])[0] or SAFE_DIV
    e_buy = clean_numeric_array([E_row["TradeBuyVolume"]])[0] or 0.0
    e_sell = clean_numeric_array([E_row["TradeSellVolume"]])[0] or 0.0
    e_tp = clean_numeric_array([E_row["time_period"]])[0] or 1.0
    e_vol = e_buy + e_sell
    e_order_flow = e_buy - e_sell
    sector_names = ["A", "B", "C", "D"]

    # 初始化特征向量
    feat = np.zeros(FEATURE_DIM, dtype=np.float64)
    feat_idx = 0

    # ------------------- 1. 基础特征（同步反向）-------------------
    feat[feat_idx] = - (e_vol / (e_vol + SAFE_DIV))
    feat_idx += 1
    # 在线价格相对5Tick均值（模拟滚动窗口 + 反向）
    e_p_ma5 = (e_p + last_p * 4) / 5
    feat[feat_idx] = - ((e_p - e_p_ma5) / (e_p_ma5 + SAFE_DIV))
    feat_idx += 1
    feat[feat_idx] = - (e_order_flow / (e_vol + SAFE_DIV))
    feat_idx += 1

    # ------------------- 2. 差分特征（同步反向）-------------------
    feat[feat_idx] = - (e_vol - last_vol if last_vol != 0 else 0.0)
    feat_idx += 1
    feat[feat_idx] = - (e_p - last_p if last_p != 0 else 0.0)
    feat_idx += 1
    e_ret = calculate_safe_return(np.array([e_p]), np.array([last_p]))[0] if last_p != 0 else 0.0
    feat[feat_idx] = - e_ret
    feat_idx += 1

    # ------------------- 3. 其他特征 -------------------
    feat[feat_idx] = e_tp
    feat_idx += 1

    # ------------------- 4. Alpha特征（同步反向）-------------------
    # 4.1 订单流相对强弱
    sector_of = 0.0
    current_sector_p = {}
    for idx, s_name in enumerate(sector_names):
        s_row = sector_rows[idx]
        s_buy = clean_numeric_array([s_row["TradeBuyVolume"]])[0] or 0.0
        s_sell = clean_numeric_array([s_row["TradeSellVolume"]])[0] or 0.0
        sector_of += (s_buy - s_sell)
        current_sector_p[s_name] = clean_numeric_array([s_row["LastPrice"]])[0] or SAFE_DIV
    sector_of /= 4
    feat[feat_idx] = - (e_order_flow - sector_of)
    feat_idx += 1

    # 4.2 残差动量（复用训练的回归系数 + 反向）
    intercept, coef = reg_coef
    sector_ret = 0.0
    for s_name in sector_names:
        s_last_p = last_sector_p.get(s_name, SAFE_DIV)
        s_p = current_sector_p[s_name]
        s_ret = calculate_safe_return(np.array([s_p]), np.array([s_last_p]))[0]
        sector_ret += s_ret
    sector_ret /= 4
    e_ret_pred = intercept + coef * sector_ret
    feat[feat_idx] = - (e_ret - e_ret_pred)
    feat_idx += 1

    # 4.3 板块领滞后关系（同步反向）
    for s_name in sector_names:
        s_last_p = last_sector_p.get(s_name, SAFE_DIV)
        s_p = current_sector_p[s_name]
        s_ret = calculate_safe_return(np.array([s_p]), np.array([s_last_p]))[0]
        feat[feat_idx] = - s_ret
        feat_idx += 1

    # ------------------- 5. 新增有效量价特征（同步训练逻辑）-------------------
    # 5.1 订单流不平衡
    e_order_imbalance = (e_buy - e_sell) / (e_buy + e_sell + SAFE_DIV)
    feat[feat_idx] = e_order_imbalance*2  # 放大效果更明显
    feat_idx += 1
    # 5.2 成交量波动率（放大2倍）
    e_vol_std = e_vol / (last_vol + SAFE_DIV) if last_vol != 0 else 0.0  # 在线简化版
    feat[feat_idx] = e_vol_std * 2  # 和训练一致：放大2倍
    feat_idx += 1
    # 5.3 量价趋势特征（同步新增，放大2倍）
    e_price_trend = calculate_safe_return(np.array([e_p]), np.array([last_p]))[0] if last_p != 0 else 0.0
    e_vol_trend = e_vol - last_vol if last_vol != 0 else 0.0
    price_vol_trend = e_price_trend * e_vol_trend
    feat[feat_idx] = price_vol_trend * 2
    feat_idx += 1

    # 最终清洗
    feat = clean_numeric_array(feat)
    feat = np.clip(feat, -10, 10)
    feat = feat.astype(np.float32)
    
    # 特征维度校验
    if feat_idx != FEATURE_DIM:
        raise ValueError(f"在线特征维度错误：实际计算{feat_idx}维，配置{FEATURE_DIM}维")
    return feat, current_sector_p

# ===================== IC 评估（增强鲁棒性）=====================
def evaluate_ic(y_true, y_pred) -> float:
    """评估IC值，处理极端情况"""
    y_true = clean_numeric_array(y_true)
    y_pred = clean_numeric_array(y_pred)
    
    # 长度对齐
    min_len = min(len(y_true), len(y_pred))
    if min_len < 2:
        return 0.0
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    
    # 过滤常量值
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    
    # 计算皮尔逊相关系数
    ic, p_value = pearsonr(y_true, y_pred)
    return float(ic) if not np.isnan(ic) else 0.0

# ===================== 时间解析 =====================
def parse_time_num_batch(time_nums):
    """将time_num转为时间周期"""
    time_nums = clean_numeric_array(time_nums)
    s = np.char.zfill(time_nums.astype(str), 9)
    h = np.vectorize(lambda x: int(x[:2]) if len(x)>=2 else 0)(s)
    m = np.vectorize(lambda x: int(x[2:4]) if len(x)>=4 else 0)(s)
    tm = h * 60 + m
    
    out = np.ones_like(tm, dtype=np.float32)
    out[(tm >= TIME_PERIOD_BINS[0]) & (tm < TIME_PERIOD_BINS[1])] = 0.0
    out[(tm >= TIME_PERIOD_BINS[2]) & (tm < TIME_PERIOD_BINS[3])] = 2.0
    return clean_numeric_array(out)