import os
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from utils import (
    filter_warnings,
    get_day_folders,
    load_day_data,
    calculate_batch_features,
    evaluate_ic,
    LGB_PARAMS,
    DATA_PATH,
    MODEL_DIR,
    CV_SPLITS,
)

filter_warnings()

def run_time_series_cv(X, y):
    """优化时间序列CV：增加early stopping，无IC反转，自然转正"""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    ic_list = []
    mse_list = []
    best = None
    best_ic = -np.inf  # 最大化IC（自然转正）
    
    for tr, va in tscv.split(X):
        Xt, Xv = X[tr], X[va]
        yt, yv = y[tr], y[va]
        
        # 严格清洗：只保留非NaN的样本
        valid_tr = ~(np.isnan(Xt).any(axis=1) | np.isnan(yt))
        valid_va = ~(np.isnan(Xv).any(axis=1) | np.isnan(yv))
        Xt, yt = Xt[valid_tr], yt[valid_tr]
        Xv, yv = Xv[valid_va], yv[valid_va]
        
        # 训练时增加early stopping（兼容所有LightGBM版本）
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(
            Xt, yt,
            eval_set=[(Xv, yv)],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )
        pred = m.predict(Xv)
        
        # 核心：无IC反转，靠特征反向自然转正
        ic = evaluate_ic(yv, pred)
        ic_list.append(ic)
        mse_list.append(np.mean((yv - pred) ** 2))
        
        if ic > best_ic:  # 以IC为核心指标选最优模型
            best_ic = ic
            best = m
    return np.mean(ic_list), np.mean(mse_list), best

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    days = [d for d in get_day_folders(DATA_PATH) if d != '5']  # 排除测试集
    print("交易日:", days)

    X_list = []
    y_list = []
    reg_coef = None
    
    for d in days:
        print(f"正在加载并计算 {d} 特征...")
        data = load_day_data(d)
        
        # 核心：标签滞后（用t时刻特征预测t+1时刻收益，避免数据泄露）
        X, rc = calculate_batch_features(data["E"], data["A"], data["B"], data["C"], data["D"])
        y = data["E"]["Return5min"].values.astype(np.float32)
        
        # 标签滞后1期（t特征→t+1收益），最后一行补NaN
        y_shifted = np.roll(y, -1)
        y_shifted[-1] = np.nan
        
        if reg_coef is None:
            reg_coef = rc
        
        X_list.append(X)
        y_list.append(y_shifted)

    # 合并数据并过滤NaN样本（核心：移除标签NaN行）
    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.hstack(y_list).astype(np.float32)
    
    # 过滤NaN样本（标签/特征有NaN的全部移除）
    valid_mask = ~(np.isnan(y_all) | np.isnan(X_all).any(axis=1))
    X_all = X_all[valid_mask]
    y_all = y_all[valid_mask]
    
    print(f"过滤后特征矩阵形状: X={X_all.shape}, y={y_all.shape}")
    if len(X_all) == 0:
        raise ValueError("过滤后无有效数据，请检查数据质量")

    # 过滤全NaN列（保留有效特征）
    mask = ~np.all(np.isnan(X_all), axis=0)
    X_valid = X_all[:, mask]
    
    # 标准化（仅用训练数据拟合，避免泄露）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # 时间序列交叉验证（无IC反转）
    avg_ic, avg_mse, best_model = run_time_series_cv(X_scaled, y_all)
    print(f"交叉验证结果 - 平均IC={avg_ic:.4f} | 平均MSE={avg_mse:.6f}")

    # 保存模型文件（全量）
    joblib.dump(best_model, os.path.join(MODEL_DIR, "lgb_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(mask, os.path.join(MODEL_DIR, "non_nan_mask.pkl"))
    joblib.dump(reg_coef, os.path.join(MODEL_DIR, "reg_coef.pkl"))
    print("模型文件保存完成：model_weights/ 下的4个pkl文件")

if __name__ == "__main__":
    main()