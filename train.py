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
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    ic_list = []
    mse_list = []
    best = None
    best_mse = np.inf
    for tr, va in tscv.split(X):
        Xt, Xv = X[tr], X[va]
        yt, yv = y[tr], y[va]
        # 交叉验证前再清洗一次，兜底
        Xt = np.nan_to_num(Xt, 0)
        Xv = np.nan_to_num(Xv, 0)
        yt = np.nan_to_num(yt, 0)
        yv = np.nan_to_num(yv, 0)
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(Xt, yt, feature_name=None)
        pred = m.predict(Xv)
        ic = evaluate_ic(yv, pred)
        mse = np.mean((yv - pred) ** 2)
        ic_list.append(ic)
        mse_list.append(mse)
        if mse < best_mse:
            best_mse = mse
            best = m
    return np.mean(ic_list), np.mean(mse_list), best

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    ###days = get_day_folders(DATA_PATH)
    days = [d for d in get_day_folders(DATA_PATH) if d != '5'] # 排除交易日 '5'，后面可以删掉这行
    print("交易日:", days)

    X_list = []
    y_list = []
    reg_coef = None  # 存储回归系数
    for d in days:
        print(f"正在加载并计算 {d} 特征...")  # 新增日志，看进度
        data = load_day_data(d)
        # 计算特征+回归系数
        X, rc = calculate_batch_features(data["E"], data["A"], data["B"], data["C"], data["D"])
        if reg_coef is None:
            reg_coef = rc  # 用第一个交易日的回归系数（也可训练全局系数）
        X_list.append(X)
        y_list.append(data["E"]["Return5min"].values.astype(np.float32))

    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.hstack(y_list).astype(np.float32)
    print(f"特征矩阵形状: X={X_all.shape}, y={y_all.shape}")  # 新增日志，验证数据

    # 过滤全NaN列
    mask = ~np.all(np.isnan(X_all), axis=0)
    X_valid = X_all[:, mask]
    # 标准化前清洗
    X_valid = np.nan_to_num(X_valid, 0)
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # 时间序列交叉验证
    avg_ic, avg_mse, _ = run_time_series_cv(X_scaled, y_all)
    print(f"交叉验证结果 - 平均IC={avg_ic:.4f} | 平均MSE={avg_mse:.6f}")

    # 训练最终模型
    final = lgb.LGBMRegressor(**LGB_PARAMS)
    final.fit(X_scaled, y_all, feature_name=None)

    # 保存模型/标准化器/掩码/回归系数
    joblib.dump(final, os.path.join(MODEL_DIR, "lgb_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(mask, os.path.join(MODEL_DIR, "non_nan_mask.pkl"))
    joblib.dump(reg_coef, os.path.join(MODEL_DIR, "reg_coef.pkl"))
    print("模型文件保存完成：model_weights/ 下的4个pkl文件")

if __name__ == "__main__":
    main()