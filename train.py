import numpy as np
import pandas as pd
from utils import (
    DATA_DIR, MODEL_DIR, LGB_PARAMS, FEATURE_CONFIG,
    get_day_folders, load_day_data, calculate_batch_features,
    evaluate_ic, TimeSeriesSplit, lgb, clean_numeric_array
)

def main():
    # 1. 加载数据并计算每日IC
    days = get_day_folders(DATA_DIR)
    all_features = []
    all_labels = []
    all_ic_results = {feat: [] for feat in FEATURE_CONFIG}
    
    for day in days:
        day_data = load_day_data(DATA_DIR, day)
        features, labels, single_ic = calculate_batch_features(day_data)
        
        # 每日IC汇总（极简打印）
        print(f"交易日{day} IC汇总：")
        for feat_name in FEATURE_CONFIG:
            ic_val = single_ic[feat_name]
            all_ic_results[feat_name].append(ic_val)
            print(f"  {feat_name}: {ic_val:.4f}")
        
        all_features.append(features)
        all_labels.append(labels)
        print("-" * 40)
    
    # 2. 合并数据
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    X = clean_numeric_array(X)
    y = clean_numeric_array(y)
    
    # 3. 平均IC打印
    print("\n所有因子平均IC：")
    for feat_name in FEATURE_CONFIG:
        print(f"  {feat_name}: {np.mean(all_ic_results[feat_name]):.4f}")
    print("-" * 40)
    
    # 4. 3折交叉验证（仅打印核心IC/MSE）
    print("\n3折交叉验证结果：")
    tscv = TimeSeriesSplit(n_splits=3)
    cv_mse = []
    cv_ic = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train)
        
        val_pred = model.predict(X_val)
        mse = np.mean((val_pred - y_val)**2)
        ic = evaluate_ic(val_pred, y_val)
        cv_mse.append(mse)
        cv_ic.append(ic)
        print(f"  折{fold} - MSE: {mse:.6f}, IC: {ic:.4f}")
    
    # 交叉验证汇总
    print(f"  平均 - MSE: {np.mean(cv_mse):.6f}, IC: {np.mean(cv_ic):.4f}")
    print("-" * 40)
    
    # 5. 训练并保存最终模型
    final_model = lgb.LGBMRegressor(**LGB_PARAMS)
    final_model.fit(X, y)
    model_path = f"{MODEL_DIR}/online_model.txt"
    final_model.booster_.save_model(model_path)
    
    # 6. 特征重要性（极简打印）
    print("\n特征重要性（降序）：")
    sorted_feats = sorted(zip(FEATURE_CONFIG, final_model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_feats:
        print(f"  {feat}: {imp}")
    print(f"\n模型已保存至：{model_path}")

if __name__ == "__main__":
    main()