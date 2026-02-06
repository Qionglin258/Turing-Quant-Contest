import numpy as np
import joblib
import lightgbm as lgb
from utils import (
    filter_warnings,
    calculate_online_feature,
    PRICE_CLIP_RANGE,
    FEATURE_DIM
)
filter_warnings()
class MyModel:
    # 新增model_dir参数，设置默认值，兼容你的调用（保留）
    def __init__(self, model_dir="./model_weights"):
        """初始化模型（支持model_dir参数）"""
        self.model_dir = model_dir  # 新增：保存模型目录（保留）
        self.model = None
        self.scaler = None
        self.non_nan_mask = None
        self.reg_coef = None
        self.last_vol = 0.0
        self.last_p = 0.0
        self.last_sector_p = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        self.load_weights()  # 加载时用model_dir（保留）
        print("模型加载完成")
    def load_weights(self):
        """加载模型（适配model_dir参数，保留）"""
        try:
            # 所有路径都用self.model_dir（保留）
            self.model = joblib.load(f"{self.model_dir}/lgb_model.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
            self.non_nan_mask = joblib.load(f"{self.model_dir}/non_nan_mask.pkl")
            self.reg_coef = joblib.load(f"{self.model_dir}/reg_coef.pkl")
        except Exception as e:
            raise Exception(f"加载失败: {e}")
    def reset(self):
        self.last_vol = 0.0
        self.last_p = 0.0
        self.last_sector_p = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    def online_predict(self, E_row, sector_row_datas):
        feat, current_sector_p = calculate_online_feature(
            E_row=E_row,
            sector_rows=sector_row_datas,
            last_vol=self.last_vol,
            last_p=self.last_p,
            last_sector_p=self.last_sector_p,
            reg_coef=self.reg_coef
        )
        # 【修改】删除你这里的单独取反（已在utils的特征计算中统一取反，避免二次取反导致方向又错！）
        # 原代码：feat = -feat  →  删除，因为utils中已经对非板块因子取反，这里再取反会抵消
        p = E_row["LastPrice"] if not np.isnan(E_row["LastPrice"]) else 1e-8
        buy = E_row["TradeBuyVolume"] if not np.isnan(E_row["TradeBuyVolume"]) else 0
        sell = E_row["TradeSellVolume"] if not np.isnan(E_row["TradeSellVolume"]) else 0
        self.last_vol = buy + sell
        self.last_p = p
        self.last_sector_p = current_sector_p
        feat_valid = feat[self.non_nan_mask].reshape(1, -1)
        feat_scaled = self.scaler.transform(feat_valid)
        pred = self.model.predict(feat_scaled)[0]
        # 【保留】你的clip逻辑，一字未改
        return float(np.clip(pred, *PRICE_CLIP_RANGE))
    def save_data(self):
        pass