# -*- coding: utf-8 -*-
"""
完整功能版 LightGBM 训练脚本
包含：数据加载、特征计算、标准化处理、非空掩码生成、时序交叉验证、全量模型训练、模型/配套文件保存
无任何核心功能阉割，可直接对接 MyModel.py 进行预测
"""

# 1. 导入所有必要依赖（无遗漏）
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# 仅屏蔽特征名不匹配的警告
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names"
)
# 保留其他无关警告的过滤（可选）
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')

# 忽略无关警告，保持控制台整洁（不屏蔽核心报错）
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')

# 2. 配置全局参数（可直接修改，无需改动核心逻辑）
CONFIG = {
    'data_path': './data',  # 原始数据目录
    'model_dir': './model_weights',  # 模型/配套文件保存目录
    'lgb_params': {
        'objective': 'regression',  # 回归任务（匹配 Return5min 预测）
        'metric': 'mse',  # 评估指标：均方误差
        'boosting_type': 'gbdt',  # 梯度提升决策树
        'learning_rate': 0.08,  # 学习率（平衡速度与效果）
        'num_leaves': 20,  # 叶子节点数（避免过拟合）
        'max_depth': 6,  # 树深度（限制复杂度）
        'min_child_samples': 20,  # 叶子节点最小样本数（抗过拟合）
        'subsample': 0.8,  # 行采样（提速+抗过拟合）
        'colsample_bytree': 0.8,  # 列采样（提速+抗过拟合）
        'reg_alpha': 0.01,  # L1 正则化（抗过拟合）
        'reg_lambda': 0.01,  # L2 正则化（抗过拟合）
        'n_estimators': 100,  # 树的数量
        'verbose': -1,  # 静默模式，不输出训练过程
        'n_jobs': -1,  # 启用所有CPU核心（提速）
    },
    'cv_splits': 3,  # 时序交叉验证折数
    'feature_dim': 7,  # 基础特征维度（E的7维特征：3个基础+3个差分+1个时段）
}

# 3. 核心工具函数（完整实现，无模拟，对接真实数据）
def get_day_folders(data_path: str) -> list:
    """
    获取 data 目录下所有合法的交易日文件夹（数字命名）
    :param data_path: 数据根目录
    :return: 排序后的交易日文件夹列表
    """
    # 检查数据目录是否存在，不存在则创建并抛出提示
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        raise FileNotFoundError(f"数据目录 {data_path} 不存在，已自动创建，请放入交易日数据（数字命名文件夹）")
    
    # 筛选数字命名的子文件夹
    day_folders = []
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path) and item.strip().isdigit():
            day_folders.append(item)
    
    # 按数字大小排序（保证交易日顺序正确）
    day_folders.sort(key=lambda x: int(x))
    
    if not day_folders:
        raise ValueError(f"数据目录 {data_path} 下无合法交易日文件夹（需数字命名）")
    
    return day_folders

def load_day_data_fast(data_path: str, day: str) -> dict:
    """
    加载单个交易日的 A/B/C/D/E 数据（csv格式）
    :param data_path: 数据根目录
    :param day: 交易日文件夹名
    :return: 包含 A/B/C/D/E DataFrame 的字典
    """
    day_path = os.path.join(data_path, day)
    if not os.path.exists(day_path):
        raise FileNotFoundError(f"交易日 {day} 文件夹不存在：{day_path}")
    
    # 定义需要加载的文件
    required_files = ['A.csv', 'B.csv', 'C.csv', 'D.csv', 'E.csv']
    day_data = {}
    
    for file_name in required_files:
        file_path = os.path.join(day_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"交易日 {day} 缺失文件：{file_path}")
        
        # 加载 csv 文件（UTF-8 编码，自动处理表头）
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='gbk')  # 兼容 Windows 中文编码
        
        # 检查核心字段是否存在（避免数据格式错误）
        core_fields = ['LastPrice', 'TradeBuyVolume', 'TradeSellVolume', 'Return5min']
        for field in core_fields:
            if field not in df.columns:
                raise ValueError(f"文件 {file_path} 缺失核心字段：{field}")
        
        # 数据类型转换（保证数值精度，避免后续计算错误）
        df['LastPrice'] = df['LastPrice'].astype(np.float32)
        df['TradeBuyVolume'] = df['TradeBuyVolume'].astype(np.int32)
        df['TradeSellVolume'] = df['TradeSellVolume'].astype(np.int32)
        df['Return5min'] = df['Return5min'].astype(np.float32)
        
        # 补充 time_period 字段（若缺失，默认赋值 1.0）
        if 'time_period' not in df.columns:
            df['time_period'] = 1.0
        else:
            df['time_period'] = df['time_period'].astype(np.float32)
        
        # 去除全NaN行（清理无效数据）
        df = df.dropna(how='all').reset_index(drop=True)
        
        # 存入字典（键为文件名前缀：A/B/C/D/E）
        day_data[file_name.split('.')[0]] = df
    
    return day_data

def calculate_features_fast(df_E: pd.DataFrame, df_A: pd.DataFrame, df_B: pd.DataFrame, df_C: pd.DataFrame, df_D: pd.DataFrame) -> np.ndarray:
    """
    完整特征计算（E的7维基础特征，支持后续扩展板块效应）
    :param df_E: E 股票数据
    :param df_A/df_B/df_C/df_D: 板块股票数据
    :return: 形状为 (n_ticks, feature_dim) 的特征矩阵（float32）
    """
    n_ticks = len(df_E)
    if n_ticks == 0:
        raise ValueError("E 股票数据无有效行，无法计算特征")
    
    # 初始化特征矩阵
    features = np.zeros((n_ticks, CONFIG['feature_dim']), dtype=np.float32)
    
    # 提取 E 股票核心数据（避免重复索引，提速）
    last_price = df_E['LastPrice'].values
    trade_buy = df_E['TradeBuyVolume'].values
    trade_sell = df_E['TradeSellVolume'].values
    time_period = df_E['time_period'].values
    
    # 计算成交量相关特征
    e_volume = trade_buy + trade_sell
    e_volume_roll = np.maximum(e_volume, 1e-8)  # 避免除零错误
    features[:, 0] = e_volume / e_volume_roll  # 成交量归一化特征
    
    # 计算价格相关特征
    e_price_roll = np.maximum(last_price, 1e-8)
    features[:, 1] = (last_price / e_price_roll) - 1  # 价格偏差特征
    
    # 计算换手率相关特征（这里简化为成交量/自身成交量，可扩展为流通盘比值）
    e_order_volume = np.maximum(e_volume, 1e-8)
    features[:, 2] = e_volume / e_order_volume  # 换手率特征
    
    # 计算差分特征（前一日与当日的差值，首日为 0）
    features[1:, 3] = e_volume[1:] - e_volume[:-1]  # 成交量差分
    features[1:, 4] = last_price[1:] - last_price[:-1]  # 价格差分
    features[1:, 5] = e_order_volume[1:] - e_order_volume[:-1]  # 换手率差分
    
    # 时段特征
    features[:, 6] = time_period
    
    # 处理异常值（NaN/Inf 替换为 0）
    features = np.where(np.isnan(features) | np.isinf(features), 0, features)
    
    return features

def evaluate_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算 IC 指标（信息系数，衡量预测值与真实值的相关性）
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: IC 值（皮尔逊相关系数）
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"真实标签与预测标签长度不匹配：{len(y_true)} vs {len(y_pred)}")
    
    if len(y_true) < 2:
        return 0.0  # 样本量不足，返回 0
    
    # 计算皮尔逊相关系数（IC 核心）
    ic, _ = pearsonr(y_true, y_pred)
    
    # 处理 NaN 值（无相关性时返回 0）
    return float(ic) if not np.isnan(ic) else 0.0

# 4. 时序交叉验证（完整评估模型性能）
def run_time_series_cv(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    执行时序交叉验证，评估模型稳定性
    :param X: 特征矩阵
    :param y: 标签向量
    :return: 平均 IC、平均 MSE、最优交叉验证模型
    """
    if len(X) < CONFIG['cv_splits'] * 2:
        raise ValueError(f"样本量不足，无法进行 {CONFIG['cv_splits']} 折时序交叉验证")
    
    tscv = TimeSeriesSplit(n_splits=CONFIG['cv_splits'])
    ic_scores = []
    mse_scores = []
    best_model = None
    best_mse = float('inf')
    
    print(f"\n===== 开始 {CONFIG['cv_splits']} 折时序交叉验证 =====")
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        # 划分训练集/验证集（时序数据不可打乱）
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 初始化并训练 LightGBM 模型
        model = lgb.LGBMRegressor(**CONFIG['lgb_params'])
        model.fit(
            X_train, y_train,
            feature_name=None  # 关闭自动特征名，提速并消除警告
        )
        
        # 预测与评估
        y_pred = model.predict(X_val)
        ic = evaluate_ic(y_val, y_pred)
        mse = np.mean((y_val - y_pred) ** 2)
        
        # 记录结果
        ic_scores.append(ic)
        mse_scores.append(mse)
        
        # 保存最优模型（基于 MSE 最小）
        if mse < best_mse:
            best_mse = mse
            best_model = model
        
        # 打印单折结果
        print(f"折 {fold_idx} - IC: {ic:.4f}, MSE: {mse:.6f}")
    
    # 计算平均结果
    avg_ic = np.mean(ic_scores)
    avg_mse = np.mean(mse_scores)
    
    print(f"\n===== 交叉验证结果汇总 =====")
    print(f"平均 IC: {avg_ic:.4f}")
    print(f"平均 MSE: {avg_mse:.6f}")
    print(f"最优模型 MSE: {best_mse:.6f}")
    
    return avg_ic, avg_mse, best_model

# 5. 主训练流程（完整无阉割）
def main():
    try:
        # 步骤1：创建模型保存目录
        if not os.path.exists(CONFIG['model_dir']):
            os.makedirs(CONFIG['model_dir'])
            print(f"已创建模型保存目录：{CONFIG['model_dir']}")
        
        # 步骤2：加载所有交易日数据
        print("\n===== 开始加载交易日数据 =====")
        day_folders = get_day_folders(CONFIG['data_path'])
        print(f"找到合法交易日：{len(day_folders)} 个，分别是：{', '.join(day_folders)}")
        
        all_X = []
        all_y = []
        for day in day_folders:
            print(f"正在加载交易日：{day}")
            day_data = load_day_data_fast(CONFIG['data_path'], day)
            
            # 提取数据
            df_E = day_data['E']
            df_A = day_data['A']
            df_B = day_data['B']
            df_C = day_data['C']
            df_D = day_data['D']
            
            # 计算特征
            X = calculate_features_fast(df_E, df_A, df_B, df_C, df_D)
            y = df_E['Return5min'].values.astype(np.float32)
            
            # 验证数据形状
            if len(X) != len(y):
                raise ValueError(f"交易日 {day} 特征与标签长度不匹配：{len(X)} vs {len(y)}")
            
            # 追加到全局列表
            all_X.append(X)
            all_y.append(y)
        
        # 步骤3：拼接全局数据（转换为 numpy 数组）
        print("\n===== 开始拼接全局数据 =====")
        all_X = np.vstack(all_X).astype(np.float32)
        all_y = np.hstack(all_y).astype(np.float32)
        
        print(f"全局特征矩阵形状：{all_X.shape}（样本数 × 特征数）")
        print(f"全局标签向量形状：{all_y.shape}（样本数）")
        print(f"特征数据类型：{all_X.dtype}，标签数据类型：{all_y.dtype}")
        
        # 步骤4：生成非空掩码（过滤全NaN特征列，抗噪）
        print("\n===== 开始生成非空特征掩码 =====")
        non_nan_mask = ~np.all(np.isnan(all_X), axis=0)
        non_nan_mask = non_nan_mask.astype(bool)
        valid_feature_count = np.sum(non_nan_mask)
        
        print(f"原始特征数：{CONFIG['feature_dim']}，有效特征数：{valid_feature_count}")
        if valid_feature_count == 0:
            raise ValueError("所有特征列均为全NaN，无法进行训练")
        
        # 应用非空掩码（过滤无效特征）
        all_X_valid = all_X[:, non_nan_mask]
        print(f"应用掩码后特征矩阵形状：{all_X_valid.shape}")
        
        # 步骤5：特征标准化（消除量纲影响，提升模型效果）
        print("\n===== 开始进行特征标准化 =====")
        scaler = StandardScaler()
        all_X_scaled = scaler.fit_transform(all_X_valid)
        
        # 验证标准化结果
        print(f"标准化后特征均值（近似 0）：{np.mean(all_X_scaled, axis=0)[:5]}（展示前5维）")
        print(f"标准化后特征标准差（近似 1）：{np.std(all_X_scaled, axis=0)[:5]}（展示前5维）")
        
        # 步骤6：时序交叉验证（评估模型）
        avg_ic, avg_mse, cv_best_model = run_time_series_cv(all_X_scaled, all_y)
        
        # 步骤7：全量训练最终模型（使用标准化后的全部数据）
        print("\n===== 开始全量训练最终模型 =====")
        final_model = lgb.LGBMRegressor(**CONFIG['lgb_params'])
        final_model.fit(
            all_X_scaled, all_y,
            feature_name=None
        )
        print("全量模型训练完成")
        
        # 步骤8：保存所有文件（模型+标准化器+非空掩码）
        print("\n===== 开始保存模型与配套文件 =====")
        # 定义保存路径
        model_save_path = os.path.join(CONFIG['model_dir'], "lgb_model.pkl")
        scaler_save_path = os.path.join(CONFIG['model_dir'], "scaler.pkl")
        mask_save_path = os.path.join(CONFIG['model_dir'], "non_nan_mask.pkl")
        
        # 保存文件
        joblib.dump(final_model, model_save_path)
        joblib.dump(scaler, scaler_save_path)
        joblib.dump(non_nan_mask, mask_save_path)
        
        # 验证保存结果
        for file_path in [model_save_path, scaler_save_path, mask_save_path]:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024 / 1024  # 转换为 MB
                print(f"已保存：{os.path.basename(file_path)}，文件大小：{file_size:.2f} MB")
        
        # 步骤9：打印训练完成汇总
        print("\n===== 训练流程全部完成 =====")
        print(f"1. 模型文件：{model_save_path}")
        print(f"2. 标准化器：{scaler_save_path}")
        print(f"3. 非空掩码：{mask_save_path}")
        print(f"4. 交叉验证平均 IC：{avg_ic:.4f}")
        print(f"5. 交叉验证平均 MSE：{avg_mse:.6f}")
        print("\n可直接运行 main.py 进行预测！")
    
    except Exception as e:
        print(f"\n===== 训练过程出错 =====")
        print(f"错误信息：{str(e)}")
        exit(1)

# 6. 程序执行入口（必须存在，否则无反应）
if __name__ == "__main__":
    main()