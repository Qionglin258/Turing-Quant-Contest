import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------- 高速数据加载 ----------------------
def get_day_folders(data_path):
    """获取所有日期文件夹（按日期升序）"""
    folders = [
        f for f in os.listdir(data_path) 
        if os.path.isdir(os.path.join(data_path, f)) and f.isdigit()
    ]
    folders.sort(key=int)
    return folders

def load_day_data_fast(data_path, day):
    """修复：加载CSV时保留Time列，不设索引"""
    day_path = os.path.join(data_path, day)
    data = {}
    for stock in ['A', 'B', 'C', 'D', 'E']:
        csv_path = os.path.join(day_path, f"{stock}.csv")
        # 关键修复：index_col=False 确保Time列作为普通列读取，不被当成索引
        df = pd.read_csv(csv_path, encoding='utf-8', index_col=False)
        # 额外兜底：清理列名空格（防止有隐藏空格）
        df.columns = df.columns.str.strip()
        data[stock] = df
    return data

# ---------------------- 向量化特征计算（核心提速） ----------------------

def calculate_features_fast(df_E, df_A, df_B, df_C, df_D):
    """
    最终修复版：适配CSV的Time列（8/9位数字，列名是Time）
    """
    # ===================== 1. 原有核心特征计算（完全保留） =====================
    e_volume = df_E['TradeBuyVolume'] + df_E['TradeSellVolume']
    e_volume_roll = e_volume.rolling(window=5, min_periods=1).mean().fillna(e_volume.mean())
    volume_feature = e_volume / (e_volume_roll + 1e-8)

    e_price_roll = df_E['LastPrice'].rolling(window=5, min_periods=1).mean().fillna(df_E['LastPrice'].mean())
    pe_feature = (df_E['LastPrice'] / e_price_roll) - 1

    e_order_volume = df_E['TradeBuyVolume'] + df_E['TradeSellVolume']
    turnover_feature = (df_E['TradeBuyVolume'] + df_E['TradeSellVolume']) / (e_order_volume + 1e-8)

    volume_diff = volume_feature.diff().fillna(0)
    price_diff = pe_feature.diff().fillna(0)
    turnover_diff = turnover_feature.diff().fillna(0)

    # ===================== 2. 解析Time列（8/9位数字，核心修复） =====================
    def parse_time_num(time_num):
        """解析8/9位数字时间：93000000→093000000，103000000→103000000"""
        # 转整数再转字符串，避免科学计数法
        time_str = str(int(time_num)).zfill(9)
        # 前两位=小时，中间两位=分钟
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        return hour, minute

    # 解析Time列生成小时/分钟
    df_E[['hour', 'minute']] = df_E['Time'].apply(
        lambda x: pd.Series(parse_time_num(x))
    )
    # 计算当天第几分钟（9:30=570，10:00=600）
    df_E['time_min'] = df_E['hour'] * 60 + df_E['minute']

    # 划分日内时段（早盘/午盘/尾盘）
    df_E['time_period'] = pd.cut(
        df_E['time_min'],
        bins=[570, 600, 870, 900],  # 9:30-10:00=早盘，10:00-14:30=午盘，14:30-15:00=尾盘
        labels=[0, 1, 2],
        right=False,
        include_lowest=True
    ).astype(float).fillna(1.0)  # 异常时间默认设为午盘

    # ===================== 3. 合并特征（原有6维+1维时段特征） =====================
    features = np.column_stack([
        volume_feature.values, pe_feature.values, turnover_feature.values,
        volume_diff.values, price_diff.values, turnover_diff.values,
        df_E['time_period'].values
    ]).astype(np.float32)

    # 异常值处理
    features = np.where(np.isnan(features) | np.isinf(features), 0, features)

    return features

# ---------------------- 评估与划分工具 ----------------------
def evaluate_ic(my_preds, ground_truth):
    """向量化计算IC值（信息系数，衡量预测与真实值的相关性）"""
    my_preds = np.array(my_preds).astype(np.float32)
    ground_truth = np.array(ground_truth).astype(np.float32)
    
    # 过滤无效值
    mask = ~(np.isnan(my_preds) | np.isnan(ground_truth) | 
             np.isinf(my_preds) | np.isinf(ground_truth))
    if np.sum(mask) < 2:
        return 0.0
    return np.corrcoef(my_preds[mask], ground_truth[mask])[0, 1]

def split_time_series_data(X, y, day_tick_counts, n_splits=3):
    """
    适配逐行预测的3折滑动窗口验证（修复索引越界问题）
    Args:
        X/y: 特征/标签矩阵
        day_tick_counts: 每个交易日的Tick数（如[27601,27601,27601,27601,27601]）
        n_splits: 折数（固定3）
    Returns:
        3折的(train_idx, val_idx)索引对
    """
    # 计算每个交易日的累计Tick索引
    cum_ticks = [0]
    total = 0
    for cnt in day_tick_counts:
        total += cnt
        cum_ticks.append(total)
    
    # 适配5天数据的3折划分（核心修复）
    splits = []
    # 折1：训练前2天，验证第3天
    splits.append((np.arange(cum_ticks[2]), np.arange(cum_ticks[2], cum_ticks[3])))
    # 折2：训练前3天，验证第4天
    splits.append((np.arange(cum_ticks[3]), np.arange(cum_ticks[3], cum_ticks[4])))
    # 折3：训练前4天，验证第5天
    splits.append((np.arange(cum_ticks[4]), np.arange(cum_ticks[4], cum_ticks[5])))
    
    return splits

# ---------------------- 训练工具 ----------------------
def train_linear_regression_core(X, y, day_tick_counts=None, n_splits=3, model_type="linear"):
    """
    线性回归核心训练逻辑（支持普通线性回归/Ridge）
    Args:
        X/y: 特征/标签矩阵
        day_tick_counts: 每个交易日的Tick数（时序划分用）
        n_splits: 交叉验证折数
        model_type: 模型类型（linear/ridge）
    Returns:
        训练好的模型、scaler、验证指标
    """
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 时序交叉验证划分
    if day_tick_counts is None:
        # 兼容无交易日划分的场景
        tscv = TimeSeriesSplit(n_splits=n_splits)
        tscv_splits = list(tscv.split(X_scaled))
    else:
        tscv_splits = split_time_series_data(X_scaled, y, day_tick_counts, n_splits)
    
    val_ic_list = []
    val_mse_list = []

    # 选择模型
    if model_type == "ridge":
        model_cls = Ridge(alpha=1.0, fit_intercept=True)
    else:
        model_cls = LinearRegression(fit_intercept=True)
    final_model = model_cls.__class__(**model_cls.get_params())

    print(f"\n开始{n_splits}折滑动窗口时序验证...")
    for fold_idx, (train_idx, val_idx) in enumerate(tscv_splits):
        # 拆分训练/验证集
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练模型
        fold_model = model_cls.__class__(**model_cls.get_params())
        fold_model.fit(X_train, y_train)
        
        # 逐Tick预测（模拟线上场景）
        y_val_pred = fold_model.predict(X_val)
        
        # 计算评估指标
        val_ic = evaluate_ic(y_val_pred, y_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_ic_list.append(val_ic)
        val_mse_list.append(val_mse)
        
        print(f"折 {fold_idx+1} - IC: {val_ic:.4f}, MSE: {val_mse:.6f}")

    # 全量训练最终模型
    final_model.fit(X_scaled, y)
    print(f"\n全量训练完成 - 权重: {final_model.coef_}, 截距: {final_model.intercept_}")
    
    # 计算平均验证指标
    avg_ic = np.mean(val_ic_list)
    avg_mse = np.mean(val_mse_list)
    print(f"\n交叉验证结果：")
    print(f"平均IC: {avg_ic:.4f} (±{np.std(val_ic_list):.4f})")
    print(f"平均MSE: {avg_mse:.6f} (±{np.std(val_mse_list):.6f})")

    # 保存模型权重
    os.makedirs("./model_weights", exist_ok=True)
    np.savez("./model_weights/linear_regression_weights.npz", 
             coef=final_model.coef_, intercept=final_model.intercept_)
    joblib.dump(scaler, "./model_weights/scaler.pkl")
    print(f"\n权重已保存至: ./model_weights/linear_regression_weights.npz")

    return final_model, scaler, (avg_ic, avg_mse)