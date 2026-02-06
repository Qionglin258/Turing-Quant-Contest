import os
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr
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
    clean_numeric_array
)
filter_warnings()
# 自定义IC损失函数（保留）
def ic_loss(y_pred, dataset):
    y_true = dataset.get_label()
    y_true = clean_numeric_array(y_true)
    y_pred = clean_numeric_array(y_pred)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return "ic_loss", 0.0, False
    ic = pearsonr(y_true, y_pred)[0]
    ic = 0.0 if np.isnan(ic) else ic
    return "ic_loss", ic, False  
def run_time_series_cv(X, y):
    """保留你原来的时间序列CV逻辑"""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    ic_list = []
    best = None
    best_ic = -np.inf
    for tr, va in tscv.split(X):
        Xt, Xv = X[tr], X[va]
        yt, yv = y[tr], y[va]
        # 数据清洗（保留你原来的）
        valid_tr = ~(np.isnan(Xt).any(axis=1) | np.isnan(yt))
        valid_va = ~(np.isnan(Xv).any(axis=1) | np.isnan(yv))
        Xt, yt = Xt[valid_tr], yt[valid_tr]
        Xv, yv = Xv[valid_va], yv[valid_va]
        # 【保留】注释的特征反向不动，因为已经在utils中统一取反
        #Xt = -Xt
        #Xv = -Xv
        # 训练（保留你原来的）
        dtrain = lgb.Dataset(Xt, label=yt)
        dvalid = lgb.Dataset(Xv, label=yv)
        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=200,
            valid_sets=[dvalid],
            feval=ic_loss,  # 只优化IC
            callbacks=[
                lgb.early_stopping(10, verbose=False),  # 早停更严格
                lgb.log_evaluation(0)
            ]
        )
        pred = model.predict(Xv)
        ic = evaluate_ic(yv, pred)
        ic_list.append(ic)
        if ic > best_ic:
            best_ic = ic
            best = model
    avg_ic = np.mean(ic_list) if ic_list else -1.0
    return avg_ic, best
def main():
    """完全保留你原来的训练流程，只修复标签移位"""
    days = [d for d in get_day_folders(DATA_PATH) if d != '5']  # 排除测试集（保留）
    print("交易日:", days)
    X_list = []
    y_list = []
    reg_coef = None
    for d in days:
        print(f"正在加载并计算 {d} 特征...")
        data = load_day_data(d)
        X, rc = calculate_batch_features(data["E"], data["A"], data["B"], data["C"], data["D"])
        y = data["E"]["Return5min"].values.astype(np.float32)
        # 【修改】核心修复：标签移位后，**删除最后一行**（无未来收益，避免nan参与训练）
        y_shifted =np.roll(y, -1)  
        y_shifted = y_shifted[:-1]  # 新增：删除最后一行nan
        X = X[:-1]  # 同步删除特征最后一行，保证特征和标签行数严格一致
        if reg_coef is None:
            reg_coef = rc
        X_list.append(X)
        y_list.append(y_shifted)
    # 数据合并+过滤（保留你原来的）
    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.hstack(y_list).astype(np.float32)
    valid_mask = ~(np.isnan(y_all) | np.isnan(X_all).any(axis=1))
    X_all = X_all[valid_mask]
    y_all = y_all[valid_mask]
    print(f"过滤后特征矩阵形状: X={X_all.shape}, y={y_all.shape}")
    if len(X_all) == 0:
        raise ValueError("过滤后无有效数据")
    # 特征过滤+标准化（保留你原来的）
    mask = ~np.all(np.isnan(X_all), axis=0)
    X_valid = X_all[:, mask]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    # 训练（保留）
    avg_ic, best_model = run_time_series_cv(X_scaled, y_all)
    print(f"交叉验证结果 - 平均IC={avg_ic:.4f}")
    # 保存模型（保留你原来的4个文件）
    joblib.dump(best_model, os.path.join(MODEL_DIR, "lgb_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(mask, os.path.join(MODEL_DIR, "non_nan_mask.pkl"))
    joblib.dump(reg_coef, os.path.join(MODEL_DIR, "reg_coef.pkl"))
    print("模型文件保存完成：model_weights/ 下的4个pkl文件")
if __name__ == "__main__":
    main()