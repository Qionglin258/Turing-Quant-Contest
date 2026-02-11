import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from utils import (
    filter_warnings, load_day_data, calculate_batch_features,
    evaluate_ic, LGB_PARAMS, DATA_PATH, MODEL_DIR, CV_SPLITS
)

def main():
    filter_warnings()
    # 加载交易日列表
    days = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d)) and d.strip().isdigit()]
    days.sort(key=lambda x: int(x))
    if not days:
        raise ValueError(f"{DATA_PATH} 下无有效交易日文件夹")
    print(f"交易日: {days}")

    # 加载所有交易日数据
    all_X = []
    all_y = []
    # 仅定义3维特征名（和utils.py输出的3维完全匹配）
    factor_names = ["vol_ratio", "weighted_sector_ret", "vol_speed"]
    
    # 遍历每个交易日
    for day in days:
        print(f"正在加载并计算 {day} 特征...")
        data = load_day_data(day)
        # 加载3维特征（utils.py已改为输出3维）
        X, _ = calculate_batch_features(data["E"], data["A"], data["B"], data["C"], data["D"])
        y = data["E"]["Return5min"].values
        
        # 输出3维因子的IC（无越界）
        print(f"\n=== 交易日{day} 单因子IC结果 ===")
        for i, name in enumerate(factor_names):
            factor = X[:, i]  # 3维特征，i=0/1/2，完全匹配
            ic = evaluate_ic(y, factor)
            print(f"{name} IC值：{ic:.4f}")
        
        all_X.append(X)
        all_y.append(y)

    # 合并所有数据
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\n总样本数：{len(X)}, 特征维度：{X.shape[1]}（3维正向因子）")

    # 时间序列交叉验证（彻底修复LGB兼容问题）
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    cv_scores = []
    cv_ics = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练LGB模型（移除所有兼容问题参数）
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        # 简化版：移除early_stopping（避免版本兼容），直接训练
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]  # 仅保留eval_set，移除callbacks/verbose
        )
        
        # 预测并评估
        y_pred = model.predict(X_val)
        mse = np.mean((y_val - y_pred) **2)
        ic = evaluate_ic(y_val, y_pred)
        cv_scores.append(mse)
        cv_ics.append(ic)
        print(f"\n折{fold+1} - MSE：{mse:.6f}, IC：{ic:.4f}")

    # 输出平均结果
    print(f"\n=== 交叉验证汇总 ===")
    print(f"平均MSE：{np.mean(cv_scores):.6f}")
    print(f"平均IC：{np.mean(cv_ics):.4f}")

    # 训练最终模型并保存
    final_model = lgb.LGBMRegressor(**LGB_PARAMS)
    final_model.fit(X, y)  # 移除verbose
    model_path = os.path.join(MODEL_DIR, "final_model.txt")
    final_model.booster_.save_model(model_path)
    print(f"\n最终模型已保存至：{model_path}")

if __name__ == "__main__":
    main()