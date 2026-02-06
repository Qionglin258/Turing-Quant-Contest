import numpy as np
import joblib
from utils import (
    filter_warnings, 
    FEATURE_DIM, 
    PRICE_CLIP_RANGE,
    calculate_online_feature
)

filter_warnings()

class MyModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.non_nan_mask = None
        self.reg_coef = None  # 回归系数（截距、斜率）
        # 缓存字段扩展
        self.last_vol = 0.0
        self.last_p = 0.0
        self.last_sector_p = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}  # 板块价格缓存
        self.load_weights()
        print("模型加载完成")

    def load_weights(self):
        """加载模型权重+标准化器+掩码+回归系数"""
        try:
            self.model = joblib.load("./model_weights/lgb_model.pkl")
            self.scaler = joblib.load("./model_weights/scaler.pkl")
            self.non_nan_mask = joblib.load("./model_weights/non_nan_mask.pkl")
            self.reg_coef = joblib.load("./model_weights/reg_coef.pkl")
        except Exception as e:
            raise Exception(f"加载失败: {e}")

    def reset(self):
        """重置缓存（每日首Tick调用）"""
        self.last_vol = 0.0
        self.last_p = 0.0
        self.last_sector_p = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

    def online_predict(self, E_row, sector_row_datas):
        """
        在线预测
        :param E_row: 当前Tick的E行数据（pd.Series）
        :param sector_row_datas: [A_row, B_row, C_row, D_row]
        :return: 预测值
        """
        # 计算在线特征
        feat, current_sector_p = calculate_online_feature(
            E_row=E_row,
            sector_rows=sector_row_datas,
            last_vol=self.last_vol,
            last_p=self.last_p,
            last_sector_p=self.last_sector_p,
            reg_coef=self.reg_coef
        )

        # 更新缓存（用于下一轮差分计算）
        p = E_row["LastPrice"] if not np.isnan(E_row["LastPrice"]) else 1e-8
        buy = E_row["TradeBuyVolume"] if not np.isnan(E_row["TradeBuyVolume"]) else 0
        sell = E_row["TradeSellVolume"] if not np.isnan(E_row["TradeSellVolume"]) else 0
        self.last_vol = buy + sell
        self.last_p = p
        self.last_sector_p = current_sector_p

        # 特征过滤、标准化、预测
        feat_valid = feat[self.non_nan_mask].reshape(1, -1)
        feat_scaled = self.scaler.transform(feat_valid)
        pred = self.model.predict(feat_scaled)[0]
        return float(np.clip(pred, *PRICE_CLIP_RANGE))