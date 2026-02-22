import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

# ===================== 基础配置 =====================
os.environ['LIGHTGBM_VERBOSE'] = '-1'
warnings.filterwarnings('ignore')

# 全局常量
TICK_PER_5MIN = 6000
SAFE_DIV = 1e-8
DATA_DIR = "./data"
MODEL_DIR = "./model_weights"
OUTPUT_DIR = "./output"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LGB参数（降低复杂度：减少叶子数、增加正则、降低迭代轮数）
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.005,
    'num_leaves': 15,        # 从31→15，大幅降低复杂度
    'max_depth': 5,          # 从8→5，限制树深度
    'n_estimators': 50,      # 从200→50，减少迭代
    'random_state': 42,
    'feature_fraction': 0.7, # 从0.8→0.7，减少每轮使用的特征数
    'min_data_in_leaf': 50,  # 从20→50，增加叶子最小样本数
    'reg_alpha': 0.5,        # 从0.1→0.5，增强L1正则
    'reg_lambda': 0.5,       # 从0.1→0.5，增强L2正则
    'force_col_wise': True,
    'verbosity': -1
}

# 核心因子（删除2个无效因子 + 保留9个有效因子）
FEATURE_CONFIG = [
    # 保留的因子（按加权系数排序）
    'price_vol_corr_pos',              # 加权1.4
    'short_vol_ratio',                 # 加权1.4
    'lastprice_vol_converge',          # 加权1.4+1.1=最终1.5（按你需求叠加）
    'stock_sector_capital_dev',        # 加权1.2 + 1.0=最终1.2
    'return_volatility_pos',           # 加权1.2
    'daily_rel_turnover',              # 加权1.0
    'buy_depth_ratio_enhanced',        # 加权0.8
    'vol_volatility',                  # 加权0.8
    'e_vs_sector_depth_diff_enhanced'  # 加权0.8
]

# 因子加权系数字典（按你的要求配置）
FACTOR_WEIGHTS = {
    'price_vol_corr_pos': 1.4,
    'short_vol_ratio': 1.4,
    'lastprice_vol_converge': 1.5,     # 1.4+1.1叠加后的值
    'stock_sector_capital_dev': 1.2,
    'return_volatility_pos': 1.2,
    'daily_rel_turnover': 1.0,
    'buy_depth_ratio_enhanced': 0.8,
    'vol_volatility': 0.8,
    'e_vs_sector_depth_diff_enhanced': 0.8
}

# ===================== 核心工具函数 =====================
def clean_numeric_array(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e8, neginf=-1e8)
    arr = np.clip(arr, -1e8, 1e8)
    return arr

def calculate_safe_return(price_series):
    shifted = np.roll(price_series, 1)
    return (price_series - shifted) / (shifted + SAFE_DIV)

def generate_double_target(labels):
    dir_target = np.where(labels > 0, 1, 0)
    strength_target = labels  # 保留正负，不再取绝对值（修复IC=0问题）
    return dir_target, strength_target

# 在线预测核心合并函数（软阈值+连续值融合）
def merge_double_predict(dir_pred, strength_pred):
    dir_pred = dir_pred * 2 - 1  # 0-1概率转为-1到1的连续方向
    final_pred = dir_pred * strength_pred
    return final_pred

# ===================== 数据加载函数 =====================
def get_day_folders(data_dir=DATA_DIR):
    """获取data目录下的日期文件夹（数字命名）"""
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()]
    return sorted(folders, key=int)

def load_day_data(data_dir, day):
    """加载单天的所有股票数据（A/B/C/D/E）"""
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

# ===================== 因子计算函数（保留有效因子的计算）=====================
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
            corr_series[i] = -corr if not np.isnan(corr) else 0.0  # 取反让IC转正
    corr_series[:window] = 0.0
    corr_pos = np.abs(corr_series)
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
    vol_pos = np.abs(vol_series)
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

def calculate_stock_sector_capital_dev(e_data, sector_data, window=TICK_PER_5MIN):
    """个股-板块主力资金偏离（保留）"""
    e_buy_depth = (e_data['BidVolume1'].values + e_data['BidVolume2'].values +
                   e_data['BidVolume3'].values + e_data['BidVolume4'].values +
                   e_data['BidVolume5'].values)
    e_sell_depth = (e_data['AskVolume1'].values + e_data['AskVolume2'].values +
                    e_data['AskVolume3'].values + e_data['AskVolume4'].values +
                    e_data['AskVolume5'].values)
    e_price = e_data['LastPrice'].values
    e_capital = (e_buy_depth - e_sell_depth) * e_price
    
    sector_capital_list = []
    for stock in ['A', 'B', 'C', 'D']:
        df = sector_data[stock]
        buy_depth = (df['BidVolume1'].values + df['BidVolume2'].values +
                     df['BidVolume3'].values + df['BidVolume4'].values +
                     df['BidVolume5'].values)
        sell_depth = (df['AskVolume1'].values + df['AskVolume2'].values +
                      df['AskVolume3'].values + df['AskVolume4'].values +
                      df['AskVolume5'].values)
        price = df['LastPrice'].values
        capital = (buy_depth - sell_depth) * price
        sector_capital_list.append(capital)
    
    sector_avg_capital = np.mean(np.array(sector_capital_list), axis=0)
    capital_dev = (e_capital - sector_avg_capital) / (sector_avg_capital + SAFE_DIV)
    smooth_dev = pd.Series(capital_dev).rolling(window=window//10, min_periods=1).mean().values
    return clean_numeric_array(smooth_dev)

# ===================== 批量特征整合（含因子加权）=====================
def calculate_batch_features(day_data):
    e_data = day_data['E']
    e_price = e_data['LastPrice'].values
    e_vol = e_data['TradeBuyVolume'].values + e_data['TradeSellVolume'].values
    e_return = calculate_safe_return(e_price)
    e_return5min = e_data['Return5min'].values if 'Return5min' in e_data.columns else np.zeros_like(e_price)
    sector_data = {s: day_data[s] for s in ['A', 'B', 'C', 'D']}
    
    # 计算所有保留的因子
    price_vol_corr_pos = calculate_price_vol_corr_pos(e_price, e_vol)
    lastprice_vol_converge = calculate_lastprice_vol_converge(e_price)
    vol_volatility = calculate_vol_volatility(e_vol)
    return_volatility_pos = calculate_return_volatility_pos(e_return)
    short_vol_ratio = calculate_short_vol_ratio(e_vol)
    daily_rel_turnover = calculate_daily_rel_turnover(e_vol)
    buy_depth_ratio_enhanced = calculate_buy_depth_ratio_enhanced(e_data)
    e_vs_sector_depth_diff_enhanced = calculate_e_vs_sector_depth_diff_enhanced(sector_data, e_data)
    stock_sector_capital_dev = calculate_stock_sector_capital_dev(e_data, sector_data)
    
    # 整合因子（按FEATURE_CONFIG顺序）
    features_list = [
        price_vol_corr_pos,
        short_vol_ratio,
        lastprice_vol_converge,
        stock_sector_capital_dev,
        return_volatility_pos,
        daily_rel_turnover,
        buy_depth_ratio_enhanced,
        vol_volatility,
        e_vs_sector_depth_diff_enhanced
    ]
    
    # 因子加权（核心：按配置的系数加权）
    weighted_features = []
    for i, feat_name in enumerate(FEATURE_CONFIG):
        weight = FACTOR_WEIGHTS.get(feat_name, 1.0)
        weighted_feat = features_list[i] * weight
        weighted_features.append(weighted_feat)
    
    # 合并为特征矩阵
    features = np.column_stack(weighted_features)
    features = clean_numeric_array(features)
    labels = clean_numeric_array(e_return5min)
    
    # 双目标标签
    dir_target, strength_target = generate_double_target(labels)
    
    # 单因子IC（仅保留有效因子）
    single_ic_results = {}
    for i, feat_name in enumerate(FEATURE_CONFIG):
        feat = features[:, i]
        if np.var(feat) < SAFE_DIV or np.var(labels) < SAFE_DIV:
            single_ic_results[feat_name] = 0.0
        else:
            ic, _ = pearsonr(feat, labels)
            single_ic_results[feat_name] = ic if not np.isnan(ic) else 0.0
    
    return features, labels, dir_target, strength_target, single_ic_results

# ===================== 评估函数 =====================
def evaluate_ic(preds, labels):
    preds = clean_numeric_array(preds)
    labels = clean_numeric_array(labels)
    if np.var(preds) < SAFE_DIV or np.var(labels) < SAFE_DIV:
        return 0.0
    ic, _ = pearsonr(preds, labels)
    return ic if not np.isnan(ic) else 0.0