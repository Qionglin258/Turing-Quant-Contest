import numpy as np
import lightgbm as lgb
import os
from utils import (
    FEATURE_CONFIG, MODEL_DIR, SAFE_DIV, TICK_PER_5MIN, ONLINE_SMOOTH_WINDOW,
    clean_numeric_array, calculate_price_vol_corr_pos, calculate_lastprice_vol_converge,
    calculate_vol_volatility, calculate_return_volatility_pos, calculate_short_vol_ratio,
    calculate_daily_rel_turnover, calculate_buy_depth_ratio_enhanced,
    calculate_e_vs_sector_depth_diff_enhanced, calculate_safe_return
)

class MyModel:
    def __init__(self):
        # 修复：校验模型文件是否存在
        self.model_path = f"{MODEL_DIR}/online_model.txt"
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件缺失：{self.model_path}，请先运行train.py训练模型")
        self.model = lgb.Booster(model_file=self.model_path)
        
        # 修复：调整历史数据最大长度（匹配因子窗口）
        self.max_history_len = TICK_PER_5MIN * 8  # 40分钟，覆盖所有因子计算窗口
        self.history = {
            'E_price': [], 'E_vol': [], 'E_return': [],
            'E_bid1': [], 'E_bid2': [], 'E_bid3': [], 'E_bid4': [], 'E_bid5': [],
            'E_ask1': [], 'E_ask2': [], 'E_ask3': [], 'E_ask4': [], 'E_ask5': [],
            'sector_bid_ask': {}
        }
        # 初始化板块股票历史
        for stock in ['A', 'B', 'C', 'D']:
            self.history['sector_bid_ask'][stock] = {
                'bid1': [], 'bid2': [], 'bid3': [], 'bid4': [], 'bid5': [],
                'ask1': [], 'ask2': [], 'ask3': [], 'ask4': [], 'ask5': []
            }

    def reset(self):
        """每个交易日初始化模型状态，清空历史数据"""
        self.history = {
            'E_price': [], 'E_vol': [], 'E_return': [],
            'E_bid1': [], 'E_bid2': [], 'E_bid3': [], 'E_bid4': [], 'E_bid5': [],
            'E_ask1': [], 'E_ask2': [], 'E_ask3': [], 'E_ask4': [], 'E_ask5': [],
            'sector_bid_ask': {stock: {'bid1': [], 'bid2': [], 'bid3': [], 'bid4': [], 'bid5': [],
                                       'ask1': [], 'ask2': [], 'ask3': [], 'ask4': [], 'ask5': []} 
                               for stock in ['A', 'B', 'C', 'D']}
        }

    def _update_history(self, E_row, sector_rows):
        """修复：先截断再追加，避免历史数据超限"""
        # 先截断历史数据（控制内存）
        for key in ['E_price', 'E_vol', 'E_return', 'E_bid1', 'E_bid2', 'E_bid3', 'E_bid4', 'E_bid5',
                    'E_ask1', 'E_ask2', 'E_ask3', 'E_ask4', 'E_ask5']:
            if len(self.history[key]) >= self.max_history_len:
                self.history[key] = self.history[key][-self.max_history_len + 1:]  # 留1个位置给新数据
        
        for stock in ['A', 'B', 'C', 'D']:
            for bid_ask_key in ['bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'ask1', 'ask2', 'ask3', 'ask4', 'ask5']:
                lst = self.history['sector_bid_ask'][stock][bid_ask_key]
                if len(lst) >= self.max_history_len:
                    self.history['sector_bid_ask'][stock][bid_ask_key] = lst[-self.max_history_len + 1:]
        
        # 再追加新数据
        self.history['E_price'].append(E_row['LastPrice'])
        self.history['E_vol'].append(E_row['TradeBuyVolume'] + E_row['TradeSellVolume'])
        self.history['E_bid1'].append(E_row['BidVolume1'])
        self.history['E_bid2'].append(E_row['BidVolume2'])
        self.history['E_bid3'].append(E_row['BidVolume3'])
        self.history['E_bid4'].append(E_row['BidVolume4'])
        self.history['E_bid5'].append(E_row['BidVolume5'])
        self.history['E_ask1'].append(E_row['AskVolume1'])
        self.history['E_ask2'].append(E_row['AskVolume2'])
        self.history['E_ask3'].append(E_row['AskVolume3'])
        self.history['E_ask4'].append(E_row['AskVolume4'])
        self.history['E_ask5'].append(E_row['AskVolume5'])
        
        # 计算收益率（修复：用utils中的安全收益率函数）
        if len(self.history['E_price']) >= 2:
            ret = calculate_safe_return(np.array(self.history['E_price']))[-1]
        else:
            ret = 0.0
        self.history['E_return'].append(ret)
        
        # 更新板块股票历史
        for idx, stock in enumerate(['A', 'B', 'C', 'D']):
            row = sector_rows[idx]
            self.history['sector_bid_ask'][stock]['bid1'].append(row['BidVolume1'])
            self.history['sector_bid_ask'][stock]['bid2'].append(row['BidVolume2'])
            self.history['sector_bid_ask'][stock]['bid3'].append(row['BidVolume3'])
            self.history['sector_bid_ask'][stock]['bid4'].append(row['BidVolume4'])
            self.history['sector_bid_ask'][stock]['bid5'].append(row['BidVolume5'])
            self.history['sector_bid_ask'][stock]['ask1'].append(row['AskVolume1'])
            self.history['sector_bid_ask'][stock]['ask2'].append(row['AskVolume2'])
            self.history['sector_bid_ask'][stock]['ask3'].append(row['AskVolume3'])
            self.history['sector_bid_ask'][stock]['ask4'].append(row['AskVolume4'])
            self.history['sector_bid_ask'][stock]['ask5'].append(row['AskVolume5'])

    def online_predict(self, E_row, sector_rows):
        """
        在线预测函数：输入当前Tick数据，返回E股票未来5分钟收益率预测值
        :param E_row: dict，E股票当前Tick数据
        :param sector_rows: list[dict]，A/B/C/D股票当前Tick数据（按顺序）
        :return: float，预测值
        """
        # 修复：轻量异常值处理（仅替换nan/inf，不截断合法值）
        def safe_clean(v):
            if isinstance(v, (int, float)):
                return 0.0 if np.isnan(v) or np.isinf(v) else v
            return v
        
        E_row = {k: safe_clean(v) for k, v in E_row.items()}
        sector_rows = [{k: safe_clean(v) for k, v in row.items()} for row in sector_rows]
        
        # 维护历史数据
        self._update_history(E_row, sector_rows)
        
        # 构建因子计算数组（修复：空数据处理）
        E_price = np.array(self.history['E_price']) if self.history['E_price'] else np.array([0.0])
        E_vol = np.array(self.history['E_vol']) if self.history['E_vol'] else np.array([0.0])
        E_return = np.array(self.history['E_return']) if self.history['E_return'] else np.array([0.0])
        
        e_depth_data = {
            'BidVolume1': np.array(self.history['E_bid1']),
            'BidVolume2': np.array(self.history['E_bid2']),
            'BidVolume3': np.array(self.history['E_bid3']),
            'BidVolume4': np.array(self.history['E_bid4']),
            'BidVolume5': np.array(self.history['E_bid5']),
            'AskVolume1': np.array(self.history['E_ask1']),
            'AskVolume2': np.array(self.history['E_ask2']),
            'AskVolume3': np.array(self.history['E_ask3']),
            'AskVolume4': np.array(self.history['E_ask4']),
            'AskVolume5': np.array(self.history['E_ask5'])
        }
        
        sector_depth_data = {}
        for idx, stock in enumerate(['A', 'B', 'C', 'D']):
            sector_depth_data[stock] = {
                'BidVolume1': np.array(self.history['sector_bid_ask'][stock]['bid1']),
                'BidVolume2': np.array(self.history['sector_bid_ask'][stock]['bid2']),
                'BidVolume3': np.array(self.history['sector_bid_ask'][stock]['bid3']),
                'BidVolume4': np.array(self.history['sector_bid_ask'][stock]['bid4']),
                'BidVolume5': np.array(self.history['sector_bid_ask'][stock]['bid5']),
                'AskVolume1': np.array(self.history['sector_bid_ask'][stock]['ask1']),
                'AskVolume2': np.array(self.history['sector_bid_ask'][stock]['ask2']),
                'AskVolume3': np.array(self.history['sector_bid_ask'][stock]['ask3']),
                'AskVolume4': np.array(self.history['sector_bid_ask'][stock]['ask4']),
                'AskVolume5': np.array(self.history['sector_bid_ask'][stock]['ask5'])
            }
        
        # 修复：降低因子计算的最小历史长度要求，避免前N个Tick全为0
        min_window = TICK_PER_5MIN // 10  # 从6000降到600，逐步积累历史
        try:
            price_vol_corr_pos = calculate_price_vol_corr_pos(E_price, E_vol)[-1] if len(E_price) >= min_window else 0.0
            lastprice_vol_converge = calculate_lastprice_vol_converge(E_price)[-1] if len(E_price) >= min_window else 0.0
            vol_volatility = calculate_vol_volatility(E_vol)[-1] if len(E_vol) >= min_window else 0.0
            return_volatility_pos = calculate_return_volatility_pos(E_return)[-1] if len(E_return) >= min_window else 0.0
            short_vol_ratio = calculate_short_vol_ratio(E_vol)[-1] if len(E_vol) >= min_window else 0.0
            daily_rel_turnover = calculate_daily_rel_turnover(E_vol)[-1] if len(E_vol) >= min_window*2 else 0.0
            buy_depth_ratio_enhanced = calculate_buy_depth_ratio_enhanced(e_depth_data)[-1] if len(e_depth_data['BidVolume1']) >= min_window//2 else 0.0
            e_vs_sector_depth_diff_enhanced = calculate_e_vs_sector_depth_diff_enhanced(sector_depth_data, e_depth_data)[-1] if len(e_depth_data['BidVolume1']) >= ONLINE_SMOOTH_WINDOW//10 else 0.0
        except Exception as e:
            print(f"因子计算异常：{str(e)}")
            price_vol_corr_pos = lastprice_vol_converge = vol_volatility = return_volatility_pos = 0.0
            short_vol_ratio = daily_rel_turnover = buy_depth_ratio_enhanced = e_vs_sector_depth_diff_enhanced = 0.0
        
        # 特征拼接与清洗（修复：维度校验）
        features = np.array([
            price_vol_corr_pos, lastprice_vol_converge, vol_volatility,
            return_volatility_pos, short_vol_ratio, daily_rel_turnover,
            buy_depth_ratio_enhanced, e_vs_sector_depth_diff_enhanced
        ]).reshape(1, -1)
        features = clean_numeric_array(features)
        
        # 修复：预测值范围限制（避免极端值）
        pred = self.model.predict(features)[0]
        pred = np.clip(pred, -0.1, 0.1)  # 限制收益率在±10%以内
        return float(pred)