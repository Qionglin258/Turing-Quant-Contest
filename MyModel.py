import os
import numpy as np
import lightgbm as lgb
from utils import (
    clean_numeric_array, calculate_safe_return,
    DYNAMIC_WEIGHT_WINDOW, PRICE_VOL_SPEED_WINDOW,
    FEATURE_DIM, SAFE_DIV, calculate_dynamic_weight
)

class MyModel:
    def __init__(self, model_dir: str = "./model_weights"):
        """初始化在线预测模型"""
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "final_model.txt")
        self.model = self._load_model()
        
        # 核心状态缓存（键名已修复为大写）
        self.tick_count = 0
        self.sector_weight = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        self.history_data = {
            "e_return": [], "A_ret": [], "B_ret": [], "C_ret": [], "D_ret": [],
            "e_p": [], "e_vol": []
        }
        self.last_vol = 0.0
        self.last_p = 0.0
        self.last_sector_p = {"A": SAFE_DIV, "B": SAFE_DIV, "C": SAFE_DIV, "D": SAFE_DIV}

    def _load_model(self) -> lgb.Booster:
        """加载LightGBM 4.6.0模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")
        try:
            model = lgb.Booster(model_file=self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败（LightGBM 4.6.0）：{str(e)}")

    def reset(self):
        """考题要求的每日初始化接口"""
        self.tick_count = 0
        self.sector_weight = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        self.history_data = {k: [] for k in self.history_data.keys()}
        self.last_vol = 0.0
        self.last_p = 0.0
        self.last_sector_p = {"A": SAFE_DIV, "B": SAFE_DIV, "C": SAFE_DIV, "D": SAFE_DIV}

    def online_predict(self, E_row: dict, sector_rows: list) -> float:
        """考题要求的核心在线预测接口"""
        self.tick_count += 1
        # 1. 提取并清洗E股数据
        e_p = clean_numeric_array([E_row.get("LastPrice", 0.0)])[0] or SAFE_DIV
        e_buy = clean_numeric_array([E_row.get("TradeBuyVolume", 0.0)])[0] or 0.0
        e_sell = clean_numeric_array([E_row.get("TradeSellVolume", 0.0)])[0] or 0.0
        e_vol = e_buy + e_sell
        e_return = clean_numeric_array([E_row.get("Return5min", 0.0)])[0]

        # 2. 计算板块收益率（A/B/C/D）
        sector_names = ["A", "B", "C", "D"]
        current_sector_p = {}
        sector_rets = {}
        for idx, s_name in enumerate(sector_names):
            s_row = sector_rows[idx] if idx < len(sector_rows) else {}
            s_p = clean_numeric_array([s_row.get("LastPrice", 0.0)])[0] or SAFE_DIV
            current_sector_p[s_name] = s_p
            # 计算板块收益率
            last_s_p = self.last_sector_p.get(s_name, SAFE_DIV)
            s_ret = calculate_safe_return(np.array([s_p]), np.array([last_s_p]))[0]
            sector_rets[s_name] = s_ret

        # 3. 更新历史数据（键名匹配：A_ret/B_ret等）
        self.history_data["e_return"].append(e_return)
        self.history_data["e_p"].append(e_p)
        self.history_data["e_vol"].append(e_vol)
        for s_name in sector_names:
            self.history_data[f"{s_name}_ret"].append(sector_rets[s_name])

        # 4. 窗口截断（避免内存溢出）
        self.history_data["e_return"] = self.history_data["e_return"][-DYNAMIC_WEIGHT_WINDOW:]
        self.history_data["e_p"] = self.history_data["e_p"][-PRICE_VOL_SPEED_WINDOW:]
        self.history_data["e_vol"] = self.history_data["e_vol"][-PRICE_VOL_SPEED_WINDOW:]
        for s_name in sector_names:
            self.history_data[f"{s_name}_ret"] = self.history_data[f"{s_name}_ret"][-DYNAMIC_WEIGHT_WINDOW:]

        # 5. 动态更新板块权重
        if (self.tick_count % DYNAMIC_WEIGHT_WINDOW == 0) and (len(self.history_data["e_return"]) >= DYNAMIC_WEIGHT_WINDOW):
            e_return_arr = np.array(self.history_data["e_return"])
            a_ret_arr = np.array(self.history_data["A_ret"])
            b_ret_arr = np.array(self.history_data["B_ret"])
            c_ret_arr = np.array(self.history_data["C_ret"])
            d_ret_arr = np.array(self.history_data["D_ret"])
            w_a, w_b, w_c, w_d = calculate_dynamic_weight(e_return_arr, a_ret_arr, b_ret_arr, c_ret_arr, d_ret_arr)
            self.sector_weight = {
                "A": np.mean(w_a) if len(w_a) > 0 else 0.25,
                "B": np.mean(w_b) if len(w_b) > 0 else 0.25,
                "C": np.mean(w_c) if len(w_c) > 0 else 0.25,
                "D": np.mean(w_d) if len(w_d) > 0 else 0.25
            }

        # 6. 计算3维特征
        feat = np.zeros(FEATURE_DIM, dtype=np.float64)
        # 因子1：成交量比率
        feat[0] = (e_vol / (e_vol + SAFE_DIV)) * 30
        # 因子2：动态权重板块收益率
        weighted_sector_ret = (
            self.sector_weight["A"] * sector_rets["A"] +
            self.sector_weight["B"] * sector_rets["B"] +
            self.sector_weight["C"] * sector_rets["C"] +
            self.sector_weight["D"] * sector_rets["D"]
        )
        feat[1] = -weighted_sector_ret * 15
        # 因子3：成交量涨速
        if len(self.history_data["e_vol"]) >= 2:
            x = np.arange(len(self.history_data["e_vol"])).reshape(-1, 1)
            y = np.array(self.history_data["e_vol"])
            if np.var(y) > 1e-8:
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression().fit(x, y)
                feat[2] = lr.coef_[0] * 150

        # 7. 特征清洗
        feat = clean_numeric_array(feat)
        feat = np.clip(feat, -10, 10).astype(np.float32)

        # 8. 更新缓存
        self.last_vol = e_vol
        self.last_p = e_p
        self.last_sector_p = current_sector_p

        # 9. 模型预测
        pred = self.model.predict(feat.reshape(1, -1))[0]
        return float(pred)