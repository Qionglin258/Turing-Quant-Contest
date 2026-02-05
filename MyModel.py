import numpy as np
import joblib
from utils import (
    filter_warnings, 
    FEATURE_DIM, 
    PRICE_CLIP_RANGE,
    calculate_online_feature  # 导入在线特征函数
)

filter_warnings()

class MyModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.non_nan_mask = None
        self.price_mean = 1e-8
        # 新增：缓存上一轮的特征值，用于计算差分
        self.last_vol = 0.0
        self.last_p = 0.0
        self.load_weights()
        print("模型加载完成")

    def load_weights(self):
        try:
            self.model = joblib.load("./model_weights/lgb_model.pkl")
            self.scaler = joblib.load("./model_weights/scaler.pkl")
            self.non_nan_mask = joblib.load("./model_weights/non_nan_mask.pkl")
        except Exception as e:
            raise Exception(f"加载失败: {e}")

    def reset(self):
        # 重置差分缓存
        self.last_vol = 0.0
        self.last_p = 0.0

    def online_predict(self, E_row, sector_row_datas):
        # 调用utils的在线特征计算函数（解耦特征逻辑）
        feat = calculate_online_feature(E_row, self.last_vol, self.last_p)

        # 更新缓存（用于下一轮差分计算）
        p = E_row["LastPrice"] if not np.isnan(E_row["LastPrice"]) else self.price_mean
        buy = E_row["TradeBuyVolume"] if not np.isnan(E_row["TradeBuyVolume"]) else 0
        sell = E_row["TradeSellVolume"] if not np.isnan(E_row["TradeSellVolume"]) else 0
        self.last_vol = buy + sell
        self.last_p = p

        # 特征过滤、标准化、预测
        feat_valid = feat[self.non_nan_mask].reshape(1, -1)
        feat_scaled = self.scaler.transform(feat_valid)
        pred = self.model.predict(feat_scaled)[0]
        return float(np.clip(pred, *PRICE_CLIP_RANGE))