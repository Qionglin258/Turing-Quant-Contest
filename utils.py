import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ===================== 基础配置（极简+兼容）=====================
os.environ['LIGHTGBM_VERBOSE'] = '-1'
warnings.filterwarnings('ignore')

# 全局常量（仅保留核心，无冗余）
TICK_PER_5MIN = 6000
SAFE_DIV = 1e-8
DATA_DIR = "./data"
MODEL_DIR = "./model_weights"
OUTPUT_DIR = "./output"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LGB参数（在线预测最优：低学习率+少正则+强泛化）
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.005,  # 降低学习率，提升泛化
    'num_leaves': 31,        # 减少叶子数，避免过拟合
    'max_depth': 8,          # 限制深度，在线更稳
    'n_estimators': 200,     # 增加迭代，充分学习
    'random_state': 42,
    'feature_fraction': 0.8, # 略降特征采样，提升泛化
    'min_data_in_leaf': 20,  # 增加叶子最小数据，避免噪声
    'reg_alpha': 0.1,        # 适度正则，防过拟合
    'reg_lambda': 0.1,
    'force_col_wise': True,
    'verbosity': -1
}

# 核心因子（无冗余）
FEATURE_CONFIG = [
    'price_vol_corr_pos',             
    'lastprice_vol_converge',          
    'vol_volatility',                  
    'return_volatility_pos',           
    'short_vol_ratio',                 
    'daily_rel_turnover',              
    'buy_depth_ratio_enhanced',        
    'e_vs_sector_depth_diff_enhanced'  
]

# ===================== 核心工具函数（在线预测优先）=====================
def clean_numeric_array(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e8, neginf=-1e8)
    arr = np.clip(arr, -1e8, 1e8)
    return arr

def calculate_safe_return(price_series):
    shifted = np.roll(price_series, 1)
    return (price_series - shifted) / (shifted + SAFE_DIV)

# 双目标标签（纯真实涨跌，无适配）
def generate_double_target(labels):
    """
    标签只反映真实涨跌，不做任何适配：
    - dir_target：1=涨，0=跌（纯真实方向）
    - strength_target：涨跌幅度绝对值（纯真实强度）
    """
    dir_target = np.where(labels > 0, 1, 0)
    strength_target = np.abs(labels)
    return dir_target, strength_target

# 在线预测核心合并函数（治本！无动态阈值+无标准化）
def merge_double_predict(dir_pred, strength_pred):
    """
    在线预测最优逻辑：
    1. 方向：固定0.5阈值（外推性最强，不反向）
    2. 强度：保留原始值（无标准化，避免在线漂移）
    3. 无任何适配逻辑，纯真实信号合并
    """
    # 1. 固定0.5方向阈值（在线预测唯一稳的选择）
    dir_pred = np.where(dir_pred >= 0.5, 1, -1)
    # 2. 强度保留原始值（不做任何标准化/缩放）
    # 3. 直接合并（方向×强度，纯真实信号）
    final_pred = dir_pred * strength_pred
    return final_pred

# ===================== 数据加载+因子计算（无反向逻辑）=====================
def get_day_folders(data_dir=DATA_DIR):
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()]
    return sorted(folders, key=int)

def load_day_data(data_dir, day):
    day_path = os.path.join(data_dir, day)
    stocks = ['A', 'B', 'C', 'D', 'E']
    day_data = {}
    for stock in stocks:
        file_path = os.path.join(day_path, f"{stock}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"缺失数据文件：{file_path}")
        df = pd.read_csv(file_path, encoding='utf-8')
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = clean_numeric_array(df[col].values)
        day_data[stock] = df
    return day_data

# 因子计算（纯正向，无×-1反向逻辑）
def calculate_price_vol_corr_pos(price_series, vol_series, window=TICK_PER_5MIN):
    corr_series = np.zeros_like(price_series)
    ret_series = calculate_safe_return(price_series)
    for i in range(window, len(price_series)):
        price_ret = ret_series[i-window:i]
        vol = vol_series[i-window:i]
        if np.var(price_ret) < SAFE_DIV or np.var(vol) < SAFE_DIV:
            corr_series[i] = 0.0
        else:
            corr, _ = pearsonr(price_ret, vol)
            corr_series[i] = corr if not np.isnan(corr) else 0.0
    corr_series[:window] = 0.0
    corr_pos = np.abs(corr_series)  # 纯绝对值，无反向
    return clean_numeric_array(corr_pos)

def calculate_lastprice_vol_converge(price_series, window=TICK_PER_5MIN):
    short_vol = pd.Series(price_series).rolling(window=window, min_periods=1).std().values
    long_vol = pd.Series(price_series).rolling(window=window*4, min_periods=1).std().values
    converge = (short_vol / (long_vol + SAFE_DIV)) - 1.0
    return clean_numeric_array(converge)

def calculate_vol_volatility(vol_series, window=TICK_PER_5MIN):
    vol_std = pd.Series(vol_series).rolling(window=window, min_periods=1).std().values
    vol_std = vol_std * np.sqrt(24*60/5)
    return clean_numeric_array(vol_std)

def calculate_return_volatility_pos(return_series, window=TICK_PER_5MIN):
    vol_series = pd.Series(return_series).rolling(window=window, min_periods=1).std().values
    vol_series = vol_series * np.sqrt(24*60/5)
    vol_pos = np.abs(vol_series)  # 纯绝对值，无反向
    return clean_numeric_array(vol_pos)

def calculate_short_vol_ratio(vol_series):
    short_vol = pd.Series(vol_series).rolling(window=TICK_PER_5MIN, min_periods=1).sum().values
    daily_vol = np.sum(vol_series) + SAFE_DIV
    vol_ratio = short_vol / daily_vol
    vol_ratio = vol_ratio * np.sqrt(24*60/5)
    return clean_numeric_array(vol_ratio)

def calculate_daily_rel_turnover(vol_series, share_cap=1e8):
    daily_total_vol = np.sum(vol_series)
    daily_turnover = daily_total_vol / (share_cap + SAFE_DIV)
    rolling_turnover = pd.Series(vol_series).rolling(window=TICK_PER_5MIN*24, min_periods=1).sum().values
    rolling_turnover = rolling_turnover / (share_cap + SAFE_DIV)
    avg_5day_turnover = pd.Series(rolling_turnover).rolling(window=TICK_PER_5MIN*24*5, min_periods=1).mean().values
    rel_turnover = daily_turnover / (avg_5day_turnover + SAFE_DIV)
    return clean_numeric_array(rel_turnover)

def calculate_buy_depth_ratio_enhanced(e_data):
    buy_depth = (e_data['BidVolume1'].values + e_data['BidVolume2'].values +
                 e_data['BidVolume3'].values + e_data['BidVolume4'].values +
                 e_data['BidVolume5'].values)
    sell_depth = (e_data['AskVolume1'].values + e_data['AskVolume2'].values +
                  e_data['AskVolume3'].values + e_data['AskVolume4'].values +
                  e_data['AskVolume5'].values)
    total_depth = buy_depth + sell_depth + SAFE_DIV
    depth_ratio = buy_depth / total_depth
    smooth_ratio = pd.Series(depth_ratio).rolling(window=TICK_PER_5MIN//5, min_periods=1).mean().values
    return clean_numeric_array(smooth_ratio)

def calculate_e_vs_sector_depth_diff_enhanced(sector_data, e_data):
    sector_depth_ratios = []
    for stock in ['A', 'B', 'C', 'D']:
        df = sector_data[stock]
        buy_depth = (df['BidVolume1'].values + df['BidVolume2'].values +
                     df['BidVolume3'].values + df['BidVolume4'].values +
                     df['BidVolume5'].values)
        sell_depth = (df['AskVolume1'].values + df['AskVolume2'].values +
                      df['AskVolume3'].values + df['AskVolume4'].values +
                      df['AskVolume5'].values)
        total_depth = buy_depth + sell_depth + SAFE_DIV
        sector_depth_ratios.append(buy_depth / total_depth)
    sector_avg_depth = np.mean(np.array(sector_depth_ratios), axis=0)
    e_buy_depth = (e_data['BidVolume1'].values + e_data['BidVolume2'].values +
                   e_data['BidVolume3'].values + e_data['BidVolume4'].values +
                   e_data['BidVolume5'].values)
    e_sell_depth = (e_data['AskVolume1'].values + e_data['AskVolume2'].values +
                    e_data['AskVolume3'].values + e_data['AskVolume4'].values +
                    e_data['AskVolume5'].values)
    e_total_depth = e_buy_depth + e_sell_depth + SAFE_DIV
    e_depth_ratio = e_buy_depth / e_total_depth
    depth_diff = e_depth_ratio - sector_avg_depth
    diff_std = np.std(depth_diff)
    smooth_window = 600 if diff_std > 0.05 else 1200
    smooth_diff = pd.Series(depth_diff).rolling(window=smooth_window, min_periods=1).mean().values
    return clean_numeric_array(smooth_diff)

# 批量特征整合（无冗余）
def calculate_batch_features(day_data):
    e_data = day_data['E']
    e_price = e_data['LastPrice'].values
    e_vol = e_data['TradeBuyVolume'].values + e_data['TradeSellVolume'].values
    e_return = calculate_safe_return(e_price)
    e_return5min = e_data['Return5min'].values if 'Return5min' in e_data.columns else np.zeros_like(e_price)
    sector_data = {s: day_data[s] for s in ['A', 'B', 'C', 'D']}
    
    # 计算纯正向因子
    price_vol_corr_pos = calculate_price_vol_corr_pos(e_price, e_vol)
    lastprice_vol_converge = calculate_lastprice_vol_converge(e_price)
    vol_volatility = calculate_vol_volatility(e_vol)
    return_volatility_pos = calculate_return_volatility_pos(e_return)
    short_vol_ratio = calculate_short_vol_ratio(e_vol)
    daily_rel_turnover = calculate_daily_rel_turnover(e_vol)
    buy_depth_ratio_enhanced = calculate_buy_depth_ratio_enhanced(e_data)
    e_vs_sector_depth_diff_enhanced = calculate_e_vs_sector_depth_diff_enhanced(sector_data, e_data)
    
    # 特征矩阵
    features = np.column_stack([
        price_vol_corr_pos, lastprice_vol_converge, vol_volatility,
        return_volatility_pos, short_vol_ratio, daily_rel_turnover,
        buy_depth_ratio_enhanced, e_vs_sector_depth_diff_enhanced
    ])
    features = clean_numeric_array(features)
    labels = clean_numeric_array(e_return5min)
    
    # 双目标标签
    dir_target, strength_target = generate_double_target(labels)
    
    # 单因子IC（纯真实，无适配）
    single_ic_results = {}
    for i, feat_name in enumerate(FEATURE_CONFIG):
        feat = features[:, i]
        if np.var(feat) < SAFE_DIV or np.var(labels) < SAFE_DIV:
            single_ic_results[feat_name] = 0.0
        else:
            ic, _ = pearsonr(feat, labels)
            single_ic_results[feat_name] = ic if not np.isnan(ic) else 0.0
    
    return features, labels, dir_target, strength_target, single_ic_results

# 评估函数（纯在线视角）
def evaluate_ic(preds, labels):
    preds = clean_numeric_array(preds)
    labels = clean_numeric_array(labels)
    if np.var(preds) < SAFE_DIV or np.var(labels) < SAFE_DIV:
        return 0.0
    ic, _ = pearsonr(preds, labels)
    return ic if not np.isnan(ic) else 0.0