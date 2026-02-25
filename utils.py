import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
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

# LGB参数（删除categorical_feature，避免重复传参）
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

# 核心因子（新增价格趋势斜率）
FEATURE_CONFIG = [
    # 保留的因子（按加权系数排序）
    'price_vol_corr_pos',
    'short_vol_ratio',
    'lastprice_vol_converge',
    'stock_sector_capital_dev',
    'return_volatility_pos',
    'daily_rel_turnover',
    'buy_depth_ratio_enhanced',
    'vol_volatility',
    'e_vs_sector_depth_diff_enhanced',
    'trade_period',  # 时段类别特征
    'price_trend_slope'  # 新增：价格趋势斜率
]

# 因子加权系数（给价格趋势斜率设合理权重）
FACTOR_WEIGHTS = {
    'price_vol_corr_pos': 1.6,
    'short_vol_ratio': 1.6,
    'lastprice_vol_converge': 1.7,
    'stock_sector_capital_dev': 0.7,
    'return_volatility_pos': 1.0,
    'daily_rel_turnover': 0.9,
    'buy_depth_ratio_enhanced': 0.7,
    'vol_volatility': 0.8,
    'e_vs_sector_depth_diff_enhanced': 0.7,
    'trade_period': 1.0,
    'price_trend_slope': 0.8
}

# ===================== 新增：适配数字格式时间的时段标签生成函数 =====================
def get_trade_period_label(time_int_series):
    """
    专门解析数字格式时间：
    输入示例：93005500 → 9:30:05.500；133104500 → 13:31:04.500
    格式规则：HHMMSSXXX 或 HMMSSXXX（小时1-2位，分钟2位，秒2位，毫秒3位）
    输出时段标签：
    1=9:30-10:00、2=10:00-11:30、3=13:00-13:30、4=13:30-14:00、5=14:00-14:50
    """
    period_label = np.zeros(len(time_int_series), dtype=int)

    for i, t in enumerate(time_int_series):
        try:
            # 转为字符串并处理空值/异常值
            s = str(int(t)) if not pd.isna(t) else ""
            if len(s) < 6:  # 长度不足的异常值标记为0
                period_label[i] = 0
                continue
            
            # 解析规则：最后3位是毫秒，前面是HHMMSS/HMMSS
            ms_part = s[-3:]  # 毫秒（固定3位）
            hhmmss_part = s[:-3]  # 时-分-秒部分
            
            # 从后往前解析：秒(2位) → 分钟(2位) → 剩下的是小时
            sec = hhmmss_part[-2:] if len(hhmmss_part) >=2 else "00"
            minute = hhmmss_part[-4:-2] if len(hhmmss_part) >=4 else "00"
            hour = hhmmss_part[:-4] if len(hhmmss_part) >=4 else hhmmss_part[:-2]
            
            # 转为整数（处理空值/非数字）
            h = int(hour) if hour.isdigit() else 0
            m = int(minute) if minute.isdigit() else 0
            
            # 计算当天总分钟数
            total_min = h * 60 + m
            
            # 划分时段标签
            if 9*60+30 <= total_min < 10*60:          # 9:30-10:00
                period_label[i] = 1
            elif 10*60 <= total_min < 11*60+30:       # 10:00-11:30
                period_label[i] = 2
            elif 13*60 <= total_min < 13*60+30:       # 13:00-13:30
                period_label[i] = 3
            elif 13*60+30 <= total_min < 14*60:       # 13:30-14:00
                period_label[i] = 4
            elif 14*60 <= total_min < 14*60+50:       # 14:00-14:50
                period_label[i] = 5
            else:                                       # 其他时段
                period_label[i] = 0
        except Exception as e:
            period_label[i] = 0  # 异常值标记为0

    return period_label

# ===================== 新增：价格趋势斜率计算函数 =====================
def calculate_price_trend_slope(price_series, window=TICK_PER_5MIN//2):
    """
    计算价格趋势斜率：用线性回归拟合window窗口内的价格序列，返回斜率
    window：默认取5分钟窗口的一半（3000tick），可根据数据密度调整
    """
    slope_series = np.zeros_like(price_series)
    # 初始化线性回归模型
    lr = LinearRegression()
    
    for i in range(window, len(price_series)):
        # 取窗口内的价格数据
        window_prices = price_series[i-window:i].reshape(-1, 1)
        # 生成时间索引（0,1,2,...,window-1）
        x = np.arange(window).reshape(-1, 1)
        
        try:
            # 拟合线性回归
            lr.fit(x, window_prices)
            # 斜率即为趋势（乘以1000放大数值，便于模型学习）
            slope = lr.coef_[0][0] * 1000
            slope_series[i] = slope
        except:
            slope_series[i] = 0.0
    
    # 前window个值设为0
    slope_series[:window] = 0.0
    return clean_numeric_array(slope_series)

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

# ===================== 批量特征整合（含因子加权+时段特征+价格斜率）=====================
def calculate_batch_features(day_data):
    e_data = day_data['E']
    e_price = e_data['LastPrice'].values
    e_vol = e_data['TradeBuyVolume'].values + e_data['TradeSellVolume'].values
    e_return = calculate_safe_return(e_price)
    e_return5min = e_data['Return5min'].values if 'Return5min' in e_data.columns else np.zeros_like(e_price)
    sector_data = {s: day_data[s] for s in ['A', 'B', 'C', 'D']}
    
    # ========== 生成时段特征（适配数字格式Time列） ==========
    if 'Time' not in e_data.columns:
        raise ValueError("数据中缺少Time列！请确保csv包含数字格式的交易时间列（如93005500）")
    trade_period = get_trade_period_label(e_data['Time'].values)
    
    # ========== 计算价格趋势斜率 ==========
    price_trend_slope = calculate_price_trend_slope(e_price)
    
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
        e_vs_sector_depth_diff_enhanced,
        trade_period,
        price_trend_slope  # 新增价格趋势斜率
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
    
    # 单因子IC（价格趋势斜率参与IC计算）
    single_ic_results = {}
    for i, feat_name in enumerate(FEATURE_CONFIG):
        if feat_name == 'trade_period':  # 跳过时段特征的IC计算
            single_ic_results[feat_name] = 0.0
            continue
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