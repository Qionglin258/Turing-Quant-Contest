import numpy as np
import pandas as pd
import os
from utils import (
    DATA_DIR, MODEL_DIR, LGB_PARAMS, FEATURE_CONFIG, FACTOR_WEIGHTS,
    get_day_folders, load_day_data, calculate_batch_features,
    evaluate_ic, TimeSeriesSplit, lgb, clean_numeric_array,
    generate_double_target, merge_double_predict
)

def analyze_factor_value(all_ic_results, dir_importance, str_importance):
    """分析因子价值：平均IC+模型权重"""
    factor_report = []
    for feat_name in FEATURE_CONFIG:
        ic_list = all_ic_results[feat_name]
        avg_ic = np.mean(ic_list)
        ic_std = np.std(ic_list)
        ic_abs = np.abs(avg_ic)
        
        dir_imp = dir_importance.get(feat_name, 0)
        str_imp = str_importance.get(feat_name, 0)
        total_imp = dir_imp + str_imp
        
        # 新增：显示因子类型和加权系数
        weight = FACTOR_WEIGHTS.get(feat_name, 1.0)
        feat_type = '类别特征' if feat_name == 'trade_period' else '数值特征'
        
        factor_report.append({
            '因子名称': feat_name,
            '类型': feat_type,
            '加权系数': weight,
            '平均IC': round(avg_ic, 4),
            'IC绝对值': round(ic_abs, 4),
            'IC标准差': round(ic_std, 4),
            '方向模型重要性': dir_imp,
            '强度模型重要性': str_imp,
            '总重要性': total_imp
        })
    
    report_df = pd.DataFrame(factor_report)
    total_all_imp = report_df['总重要性'].sum()
    report_df['模型权重(%)'] = report_df['总重要性'].apply(lambda x: round(x/total_all_imp*100, 2) if total_all_imp>0 else 0)
    report_df = report_df.sort_values(by='IC绝对值', ascending=False)
    return report_df

def main():
    # 1. 加载训练数据
    days = get_day_folders(DATA_DIR)
    train_days = [d for d in days if int(d) < 4]
    if len(train_days) < 1:
        raise ValueError("请确保data目录下有至少1天的训练数据！")
    
    all_features = []
    all_labels = []
    all_dir_target = []    
    all_strength_target = []
    all_ic_results = {feat: [] for feat in FEATURE_CONFIG}
    
    print("===== 加载训练数据（含数字格式时间+时段类别特征） =====")
    for day in train_days:
        day_data = load_day_data(DATA_DIR, day)
        features, labels, dir_target, strength_target, single_ic = calculate_batch_features(day_data)
        
        print(f"\n【交易日{day} 单日IC（参考）】")
        for feat_name in FEATURE_CONFIG:
            ic_val = single_ic[feat_name]
            all_ic_results[feat_name].append(ic_val)
            feat_type = "（类别特征）" if feat_name == 'trade_period' else ""
            print(f"  {feat_name} (权重{FACTOR_WEIGHTS[feat_name]}){feat_type}: {ic_val:.4f}")
        
        all_features.append(features)
        all_labels.append(labels)
        all_dir_target.append(dir_target)
        all_strength_target.append(strength_target)
    
    # 2. 合并训练数据
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    y_dir = np.hstack(all_dir_target)
    y_strength = np.hstack(all_strength_target)
    
    X = clean_numeric_array(X)
    y = clean_numeric_array(y)
    y_dir = clean_numeric_array(y_dir)
    y_strength = clean_numeric_array(y_strength)
    
    # 找到时段特征的索引（用于标记类别特征）
    period_feat_idx = FEATURE_CONFIG.index('trade_period')
    
    # 3. 固定3折时间序列交叉验证（含时段类别特征）
    print("\n===== 3折时间序列交叉验证（含时段类别特征） =====")
    tscv = TimeSeriesSplit(n_splits=3)
    cv_mse = []
    cv_ic = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        y_dir_train, y_dir_val = y_dir[train_idx], y_dir[val_idx]
        y_str_train, y_str_val = y_strength[train_idx], y_strength[val_idx]
        
        # 方向模型（低复杂度+类别特征）
        dir_model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            learning_rate=0.005,
            num_leaves=15,        # 降低复杂度
            max_depth=5,          # 降低复杂度
            n_estimators=50,      # 减少迭代
            random_state=42,
            feature_fraction=0.7, # 减少特征采样
            min_data_in_leaf=50,  # 增加叶子最小样本
            reg_alpha=0.5,        # 增强正则
            reg_lambda=0.5,       # 增强正则
            force_col_wise=True,
            verbosity=-1,
            categorical_feature=[period_feat_idx]  # 标记类别特征的索引
        )
        dir_model.fit(X_train, y_dir_train)
        dir_pred = dir_model.predict_proba(X_val)[:, 1]
        
        # 强度模型（低复杂度+类别特征，只传一次categorical_feature）
        str_model = lgb.LGBMRegressor(
            **LGB_PARAMS,
            categorical_feature=[period_feat_idx]  # 标记类别特征索引
        )
        str_model.fit(X_train, y_str_train)
        str_pred = str_model.predict(X_val)
        
        # 合并预测
        final_pred = merge_double_predict(dir_pred, str_pred)
        
        # 评估
        mse = np.mean((final_pred - y_val)**2)
        ic = evaluate_ic(final_pred, y_val)
        cv_mse.append(mse)
        cv_ic.append(ic)
        
        print(f"  折{fold} - MSE: {mse:.6f}, IC: {ic:.4f}")
    
    avg_mse = np.mean(cv_mse) if cv_mse else 0.0
    avg_ic = np.mean(cv_ic) if cv_ic else 0.0
    print(f"  平均 - MSE: {avg_mse:.6f}, IC: {avg_ic:.4f}")
    
    # 4. 训练最终模型（含类别特征）
    print("\n===== 训练最终在线模型（低复杂度+因子加权+时段类别特征） =====")
    final_dir_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        learning_rate=0.005,
        num_leaves=15,
        max_depth=5,
        n_estimators=50,
        random_state=42,
        feature_fraction=0.7,
        min_data_in_leaf=50,
        reg_alpha=0.5,
        reg_lambda=0.5,
        force_col_wise=True,
        verbosity=-1,
        categorical_feature=[period_feat_idx]  # 标记类别特征
    )
    final_dir_model.fit(X, y_dir)
    
    final_str_model = lgb.LGBMRegressor(
        **LGB_PARAMS,
        categorical_feature=[period_feat_idx]  # 标记类别特征
    )
    final_str_model.fit(X, y_strength)
    
    # 5. 因子价值分析报告
    print("\n" + "="*80)
    print("【核心：因子价值分析报告（含时段类别特征）】")
    print("="*80)
    dir_importance = dict(zip(FEATURE_CONFIG, final_dir_model.feature_importances_))
    str_importance = dict(zip(FEATURE_CONFIG, final_str_model.feature_importances_))
    factor_report = analyze_factor_value(all_ic_results, dir_importance, str_importance)
    print(factor_report.to_string(index=False))
    
    # 6. 关键结论
    print("\n" + "="*80)
    print("【关键结论】")
    print("="*80)
    effective_factors = factor_report[factor_report['IC绝对值']>0.05]['因子名称'].tolist()
    high_weight_factors = factor_report[factor_report['加权系数']>=1.2]['因子名称'].tolist()
    core_factors = factor_report[factor_report['模型权重(%)']>5]['因子名称'].tolist()
    categorical_factors = factor_report[factor_report['类型']=='类别特征']['因子名称'].tolist()
    
    print(f"1. 高有效因子（IC绝对值>0.05）：{effective_factors}")
    print(f"2. 高加权因子（系数≥1.2）：{high_weight_factors}")
    print(f"3. 模型核心因子（权重>5%）：{core_factors}")
    print(f"4. 类别特征：{categorical_factors}")
    print(f"5. 最优因子（有效+高加权+核心）：{list(set(effective_factors) & set(high_weight_factors) & set(core_factors))}")
    
    # 7. 保存模型
    dir_model_path = f"{MODEL_DIR}/online_dir_model.txt"
    str_model_path = f"{MODEL_DIR}/online_strength_model.txt"
    final_dir_model.booster_.save_model(dir_model_path)
    final_str_model.booster_.save_model(str_model_path)
    
    print(f"\n✅ 模型保存完成：")
    print(f"   - 方向模型：{dir_model_path}")
    print(f"   - 强度模型：{str_model_path}")

if __name__ == "__main__":
    main()