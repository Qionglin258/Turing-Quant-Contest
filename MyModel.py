import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb  # 新增导入
import warnings
# 精准屏蔽特征名不匹配的警告
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names"
)

class MyModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.non_nan_mask = None
        self.history_features = []
        self.window_size = 5
        self.price_mean = 0.0
        self.load_weights()
        print("权重和标准化器加载成功")

    def load_weights(self):
        try:
            # 核心修改：加载 LightGBM 模型（替换原有线性回归权重）
            self.model = joblib.load("./model_weights/lgb_model.pkl")
            self.scaler = joblib.load("./model_weights/scaler.pkl")
            self.non_nan_mask = joblib.load("./model_weights/non_nan_mask.pkl")
            self.price_mean = self.scaler.mean_[1] if len(self.scaler.mean_)>=2 else 1e-8
        except FileNotFoundError as e:
            raise Exception(f"权重文件缺失：{e}")
        except Exception as e:
            raise Exception(f"权重加载失败：{e}")

    def reset(self):
        self.history_features = []
        print("时序状态已重置")

    def online_predict(self, E_row, sector_dfs):
        # 1. 数据类型转换 + 预处理（逻辑完全不变，来自你现有代码）
        if not isinstance(E_row, pd.Series):
            E_row = pd.Series(E_row)
        
        last_price = E_row['LastPrice']
        if np.isnan(last_price) or last_price == 0:
            last_price = self.price_mean
        
        trade_buy = E_row['TradeBuyVolume'] if not np.isnan(E_row['TradeBuyVolume']) else 0
        trade_sell = E_row['TradeSellVolume'] if not np.isnan(E_row['TradeSellVolume']) else 0
        e_volume = trade_buy + trade_sell
        
        # 2. 特征计算（逻辑完全不变，保留 7 维/31 维特征）
        e_volume_roll = e_volume if e_volume != 0 else 1e-8
        volume_feature = e_volume / (e_volume_roll + 1e-8)

        e_price_roll = last_price
        pe_feature = (last_price / (e_price_roll + 1e-8)) - 1

        e_order_volume = e_volume if e_volume != 0 else 1e-8
        turnover_feature = e_volume / (e_order_volume + 1e-8)

        volume_diff = 0.0
        price_diff = 0.0
        turnover_diff = 0.0
        
        # 时段特征（逻辑完全不变，来自你现有代码）
        time_period = E_row['time_period']

        # 3. 合并特征（逻辑完全不变）
        features = np.array([
            volume_feature, pe_feature, turnover_feature,
            volume_diff, price_diff, turnover_diff,
            time_period
        ], dtype=np.float32)

        # 4. 掩码补零 + 标准化（逻辑完全不变）
        features[~self.non_nan_mask] = 0
        features = np.where(np.isnan(features) | np.isinf(features), 0, features)
        features_scaled = self.scaler.transform(features.reshape(1, -1))[0]

        # 5. 核心修改：LightGBM 预测（替换原有线性回归的 dot 计算）
        pred = self.model.predict(features_scaled.reshape(1, -1))[0]

        # 6. 时序窗口管理 + 结果裁剪（逻辑完全不变）
        self.history_features.append(features)
        if len(self.history_features) > self.window_size:
            self.history_features.pop(0)
        
        return float(np.clip(pred, -0.05, 0.05))