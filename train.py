import numpy as np
import pandas as pd
from utils import (
    DATA_DIR, MODEL_DIR, LGB_PARAMS, FEATURE_CONFIG,
    get_day_folders, load_day_data, calculate_batch_features,
    evaluate_ic, TimeSeriesSplit, lgb, clean_numeric_array,
    generate_double_target, merge_double_predict
)

def main():
    # 1. 加载训练数据（1-4天）
    days = get_day_folders(DATA_DIR)
    # 强制只取1-4天训练（在线预测第五天）
    train_days = [d for d in days if int(d) <= 4]
    if len(train_days) < 4:
        raise ValueError("请确保data目录下有1-4天的训练数据！")
    
    all_features = []
    all_labels = []
    all_dir_target = []    
    all_strength_target = []
    all_ic_results = {feat: [] for feat in FEATURE_CONFIG}
    
    print("===== 加载1-4天训练数据 =====")
    for day in train_days:
        day_data = load_day_data(DATA_DIR, day)
        features, labels, dir_target, strength_target, single_ic = calculate_batch_features(day_data)
        
        # 打印每日真实IC（无适配）
        print(f"\n交易日{day} 单因子真实IC：")
        for feat_name in FEATURE_CONFIG:
            ic_val = single_ic[feat_name]
            all_ic_results[feat_name].append(ic_val)
            print(f"  {feat_name}: {ic_val:.4f}")
        
        all_features.append(features)
        all_labels.append(labels)
        all_dir_target.append(dir_target)
        all_strength_target.append(strength_target)
    
    # 2. 合并训练数据（纯真实，无适配）
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    y_dir = np.hstack(all_dir_target)
    y_strength = np.hstack(all_strength_target)
    
    X = clean_numeric_array(X)
    y = clean_numeric_array(y)
    y_dir = clean_numeric_array(y_dir)
    y_strength = clean_numeric_array(y_strength)
    
    # 3. 打印训练集平均IC（纯真实）
    print("\n===== 1-4天训练集平均IC =====")
    for feat_name in FEATURE_CONFIG:
        avg_ic = np.mean(all_ic_results[feat_name])
        print(f"  {feat_name}: {avg_ic:.4f}")
    
    # 4. 时间序列交叉验证（模拟在线泛化）
    print("\n===== 3折时间序列交叉验证（模拟在线） =====")
    tscv = TimeSeriesSplit(n_splits=3)
    cv_mse = []
    cv_ic = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        y_dir_train, y_dir_val = y_dir[train_idx], y_dir[val_idx]
        y_str_train, y_str_val = y_strength[train_idx], y_strength[val_idx]
        
        # 训练方向模型（分类，纯真实涨跌）
        dir_model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            learning_rate=0.005,
            num_leaves=31,
            max_depth=8,
            n_estimators=200,
            random_state=42,
            feature_fraction=0.8,
            min_data_in_leaf=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            force_col_wise=True,
            verbosity=-1
        )
        dir_model.fit(X_train, y_dir_train)
        dir_pred = dir_model.predict_proba(X_val)[:, 1] # 涨的概率
        
        # 训练强度模型（回归，纯真实幅度）
        str_model = lgb.LGBMRegressor(**LGB_PARAMS)
        str_model.fit(X_train, y_str_train)
        str_pred = str_model.predict(X_val)
        
        # 合并预测（在线专用逻辑：固定0.5+无标准化）
        final_pred = merge_double_predict(dir_pred, str_pred)
        
        # 评估（纯在线视角）
        mse = np.mean((final_pred - y_val)**2)
        ic = evaluate_ic(final_pred, y_val)
        cv_mse.append(mse)
        cv_ic.append(ic)
        
        print(f"  折{fold} - MSE: {mse:.6f}, IC: {ic:.4f}")
    
    # 交叉验证汇总
    avg_mse = np.mean(cv_mse)
    avg_ic = np.mean(cv_ic)
    print(f"  平均 - MSE: {avg_mse:.6f}, IC: {avg_ic:.4f}")
    
    # 5. 训练最终在线模型（1-4天全量数据）
    print("\n===== 训练最终在线模型（1-4天全量） =====")
    # 方向模型
    final_dir_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        learning_rate=0.005,
        num_leaves=31,
        max_depth=8,
        n_estimators=200,
        random_state=42,
        feature_fraction=0.8,
        min_data_in_leaf=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        force_col_wise=True,
        verbosity=-1
    )
    final_dir_model.fit(X, y_dir)
    dir_model_path = f"{MODEL_DIR}/online_dir_model.txt"
    final_dir_model.booster_.save_model(dir_model_path)
    
    # 强度模型
    final_str_model = lgb.LGBMRegressor(**LGB_PARAMS)
    final_str_model.fit(X, y_strength)
    str_model_path = f"{MODEL_DIR}/online_strength_model.txt"
    final_str_model.booster_.save_model(str_model_path)
    
    # 6. 特征重要性（在线视角）
    print("\n===== 在线模型特征重要性 =====")
    print("方向模型（降序）：")
    dir_sorted = sorted(zip(FEATURE_CONFIG, final_dir_model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in dir_sorted:
        print(f"  {feat}: {imp}")
    
    print("\n强度模型（降序）：")
    str_sorted = sorted(zip(FEATURE_CONFIG, final_str_model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in str_sorted:
        print(f"  {feat}: {imp}")
    
    # 模型保存路径
    print(f"\n✅ 在线方向模型保存至：{dir_model_path}")
    print(f"✅ 在线强度模型保存至：{str_model_path}")

if __name__ == "__main__":
    main()