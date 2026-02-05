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
    calculate_batch_features,  # 替换原有函数
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
    days = get_day_folders(DATA_PATH)
    print("交易日:", days)

    X_list = []
    y_list = []
    for d in days:
        print("加载", d)
        data = load_day_data(d)
        # 替换为新的批量特征计算函数
        X = calculate_batch_features(data["E"], data["A"], data["B"], data["C"], data["D"])
        y = data["E"]["Return5min"].values.astype(np.float32)
        X_list.append(X)
        y_list.append(y)

    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.hstack(y_list).astype(np.float32)
    print("X shape:", X_all.shape)

    mask = ~np.all(np.isnan(X_all), axis=0)
    X_valid = X_all[:, mask]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    avg_ic, avg_mse, _ = run_time_series_cv(X_scaled, y_all)
    print(f"CV IC={avg_ic:.4f} MSE={avg_mse:.6f}")

    final = lgb.LGBMRegressor(**LGB_PARAMS)
    final.fit(X_scaled, y_all, feature_name=None)

    joblib.dump(final, os.path.join(MODEL_DIR, "lgb_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(mask, os.path.join(MODEL_DIR, "non_nan_mask.pkl"))
    print("保存完成：模型/标准化器/掩码")

if __name__ == "__main__":
    main()