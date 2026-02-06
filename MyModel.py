# 保留你原有所有导入/类结构，只改核心方法
import os
import numpy as np
import joblib
import lightgbm as lgb
from utils import calculate_online_feature  # 保留你原有导入

class MyModel:
    def __init__(self, model_dir="./model_weights"):
        # 1. 加载模型组件（保留你原有路径/变量名）
        self.model = joblib.load(os.path.join(model_dir, "lgb_model.pkl"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.non_nan_mask = joblib.load(os.path.join(model_dir, "non_nan_mask.pkl"))
        self.reg_coef = joblib.load(os.path.join(model_dir, "reg_coef.pkl"))
        
        # 2. 初始化缓存（保留你原有变量名）
        self.last_vol = 0.0
        self.last_p = 0.0
        self.last_sector_p = {"A":0.0, "B":0.0, "C":0.0, "D":0.0}

    def reset(self):
        # 保留你原有重置逻辑，只补全初始值
        self.last_vol = 0.0
        self.last_p = 0.0
        self.last_sector_p = {"A":0.0, "B":0.0, "C":0.0, "D":0.0}

    def predict(self, e_row, sector_rows):
        # 3. 计算在线特征（完全保留你原有调用方式）
        feat, current_sector_p = calculate_online_feature(
            E_row=e_row,
            sector_rows=sector_rows,
            last_vol=self.last_vol,
            last_p=self.last_p,
            last_sector_p=self.last_sector_p,
            reg_coef=self.reg_coef
        )
        
        # 4. 特征预处理（核心修正：和训练100%对齐）
        # 原问题：可能没过滤NaN列/标准化，或维度错误
        feat = np.array(feat)  # 确保是numpy数组（避免list）
        feat_filtered = feat[self.non_nan_mask].reshape(1, -1)  # 过滤全NaN列
        feat_scaled = self.scaler.transform(feat_filtered)      # 标准化
        
        # 5. 模型预测（核心修正：删除任何IC反转操作）
        # 原问题：如果加了pred=-pred，这里必须删掉！
        pred = self.model.predict(feat_scaled)[0]
        
        # 6. 更新缓存（保留你原有逻辑，只保证无未来信息）
        self.last_vol = e_row["TradeBuyVolume"] + e_row["TradeSellVolume"]
        self.last_p = e_row["LastPrice"]
        self.last_sector_p = current_sector_p
        
        return pred