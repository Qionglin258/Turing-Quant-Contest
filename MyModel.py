import numpy as np
import pandas as pd
import lightgbm as lgb
from utils import (
    MODEL_DIR, SAFE_DIV, TICK_PER_5MIN, FACTOR_WEIGHTS,
    clean_numeric_array, merge_double_predict
)

class MyModel:
    def __init__(self):
        # 初始化双模型（方向+强度）
        self.dir_model = None
        self.str_model = None
        self.load_double_models()
        
        # 核心缓存（数组版，支持9个因子）
        self.max_cache = 100000
        self.e_price = np.zeros(self.max_cache, dtype=np.float32)
        self.e_vol = np.zeros(self.max_cache, dtype=np.float32)
        self.e_return = np.zeros(self.max_cache, dtype=np.float32)
        self.cache_idx = 0
        
        # ========== 增量因子缓存（9个因子）==========
        self.features = np.zeros((self.max_cache, 9), dtype=np.float32)
        
        # 增量计算需要的滚动状态（仅保留9个有效因子）
        self.rolling = {
            # 9个有效因子的滚动状态
            'pv_corr': {'sum_xy':0.0, 'sum_x2':0.0, 'sum_y2':0.0, 'sum_x':0.0, 'sum_y':0.0, 'n':0},
            'short_vol_ratio': {'ma_short':0.0, 'ma_long':0.0, 'alpha_short':2/(20+1), 'alpha_long':2/(100+1)},
            'price_conv': {'short_std':0.0, 'long_std':0.0, 'alpha_short':2/(TICK_PER_5MIN+1), 'alpha_long':2/(TICK_PER_5MIN*3+1)},
            'capital_dev': {'e_capital':0.0, 'sector_avg_capital':0.0, 'alpha':2/(TICK_PER_5MIN//10+1)},
            'ret_vol': {'std':0.0, 'alpha':2/(TICK_PER_5MIN+1)},
            'turnover': {'ma':0.0, 'alpha':2/(TICK_PER_5MIN+1)},
            'buy_depth': {'last':0.0},
            'vol_vol': {'std':0.0, 'alpha':2/(TICK_PER_5MIN+1)},
            'sector_diff': {'last':0.0}
        }

    def load_double_models(self):
        """加载双目标模型（方向+强度）"""
        dir_model_path = f"{MODEL_DIR}/online_dir_model.txt"
        str_model_path = f"{MODEL_DIR}/online_strength_model.txt"
        try:
            self.dir_model = lgb.Booster(model_file=dir_model_path)
            self.str_model = lgb.Booster(model_file=str_model_path)
        except Exception as e:
            raise RuntimeError(f"双模型加载失败：{str(e)}\n请先运行train.py训练双模型！")

    def reset(self):
        """每日重置：仅重置索引和滚动状态"""
        self.cache_idx = 0
        # 重置9个因子的滚动状态
        for k in self.rolling.keys():
            if k == 'pv_corr':
                self.rolling[k] = {'sum_xy':0.0, 'sum_x2':0.0, 'sum_y2':0.0, 'sum_x':0.0, 'sum_y':0.0, 'n':0}
            elif k in ['short_vol_ratio', 'price_conv', 'ret_vol', 'turnover', 'vol_vol']:
                for sk in self.rolling[k]:
                    if 'alpha' not in sk:
                        self.rolling[k][sk] = 0.0
            elif k == 'capital_dev':
                self.rolling[k] = {'e_capital':0.0, 'sector_avg_capital':0.0, 'alpha':2/(TICK_PER_5MIN//10+1)}
            else:  # buy_depth, sector_diff
                self.rolling[k]['last'] = 0.0

    def _update_rolling_features(self):
        """核心：增量更新9个因子（O(1)复杂度）+ 因子加权"""
        i = self.cache_idx - 1
        if i < 1:  # 至少2个Tick才计算
            for f in range(9):
                self.features[i, f] = 0.0
            return
        
        # ========== 1. 9个因子增量计算（按FEATURE_CONFIG顺序） ==========
        # 1.1 price_vol_corr_pos（量价相关性，取反+绝对值）
        pv = self.rolling['pv_corr']
        ret = self.e_return[i]
        vol = self.e_vol[i]
        window = TICK_PER_5MIN
        if i >= window:
            old_ret = self.e_return[i - window]
            old_vol = self.e_vol[i - window]
            pv['sum_xy'] -= old_ret * old_vol
            pv['sum_x2'] -= old_ret **2
            pv['sum_y2'] -= old_vol** 2
            pv['sum_x'] -= old_ret
            pv['sum_y'] -= old_vol
            pv['n'] -= 1
        pv['sum_xy'] += ret * vol
        pv['sum_x2'] += ret **2
        pv['sum_y2'] += vol** 2
        pv['sum_x'] += ret
        pv['sum_y'] += vol
        pv['n'] += 1
        if pv['n'] >= 2:
            cov = (pv['sum_xy'] - pv['sum_x']*pv['sum_y']/pv['n']) / pv['n']
            std_x = np.sqrt((pv['sum_x2'] - pv['sum_x']**2/pv['n']) / pv['n'])
            std_y = np.sqrt((pv['sum_y2'] - pv['sum_y']**2/pv['n']) / pv['n'])
            corr = cov / (std_x * std_y + SAFE_DIV)
            corr = -corr  # 取反让IC转正
            feat_val = np.abs(corr)
        else:
            feat_val = 0.0
        # 因子加权（核心）
        self.features[i, 0] = feat_val * FACTOR_WEIGHTS['price_vol_corr_pos']

        # 1.2 short_vol_ratio（量比）
        vr = self.rolling['short_vol_ratio']
        vr['ma_short'] = vr['alpha_short'] * self.e_vol[i] + (1 - vr['alpha_short']) * vr['ma_short']
        vr['ma_long'] = vr['alpha_long'] * self.e_vol[i] + (1 - vr['alpha_long']) * vr['ma_long']
        vol_ratio = vr['ma_short'] / (vr['ma_long'] + SAFE_DIV)
        feat_val = vol_ratio * np.sqrt(24*60/5)
        self.features[i, 1] = feat_val * FACTOR_WEIGHTS['short_vol_ratio']

        # 1.3 lastprice_vol_converge（价格收敛率）
        pc = self.rolling['price_conv']
        price_ret = self.e_return[i]
        pc['short_std'] = pc['alpha_short'] * (price_ret**2) + (1 - pc['alpha_short']) * pc['short_std']
        pc['long_std'] = pc['alpha_long'] * pc['short_std'] + (1 - pc['alpha_long']) * pc['long_std']
        feat_val = (pc['short_std'] / (pc['long_std'] + SAFE_DIV)) - 1.0
        self.features[i, 2] = feat_val * FACTOR_WEIGHTS['lastprice_vol_converge']

        # 1.4 stock_sector_capital_dev（个股-板块资金偏离）
        cd = self.rolling['capital_dev']
        capital_dev = (cd['e_capital'] - cd['sector_avg_capital']) / (cd['sector_avg_capital'] + SAFE_DIV)
        feat_val = cd['alpha'] * capital_dev + (1 - cd['alpha']) * self.features[i-1, 3]  # 平滑
        self.features[i, 3] = feat_val * FACTOR_WEIGHTS['stock_sector_capital_dev']

        # 1.5 return_volatility_pos（收益率波动率）
        rv = self.rolling['ret_vol']
        rv['std'] = rv['alpha'] * (self.e_return[i]**2) + (1 - rv['alpha']) * rv['std']
        feat_val = np.sqrt(rv['std']) * np.sqrt(24*60/5)
        self.features[i, 4] = feat_val * FACTOR_WEIGHTS['return_volatility_pos']

        # 1.6 daily_rel_turnover（相对换手率）
        to = self.rolling['turnover']
        to['ma'] = to['alpha'] * self.e_vol[i] + (1 - to['alpha']) * to['ma']
        daily_total_vol = np.sum(self.e_vol[:i+1]) + SAFE_DIV
        share_cap = 1e8
        daily_turnover = daily_total_vol / share_cap
        rolling_turnover = to['ma'] / share_cap
        avg_5day_turnover = rolling_turnover
        feat_val = daily_turnover / (avg_5day_turnover + SAFE_DIV)
        self.features[i, 5] = feat_val * FACTOR_WEIGHTS['daily_rel_turnover']

        # 1.7 buy_depth_ratio_enhanced（买盘深度）
        feat_val = self.rolling['buy_depth']['last']
        self.features[i, 6] = feat_val * FACTOR_WEIGHTS['buy_depth_ratio_enhanced']

        # 1.8 vol_volatility（成交量波动率）
        vv = self.rolling['vol_vol']
        vol_ret = (self.e_vol[i] - self.e_vol[i-1]) / (self.e_vol[i-1] + SAFE_DIV)
        vv['std'] = vv['alpha'] * (vol_ret**2) + (1 - vv['alpha']) * vv['std']
        feat_val = np.sqrt(vv['std']) * np.sqrt(24*60/5)
        self.features[i, 7] = feat_val * FACTOR_WEIGHTS['vol_volatility']

        # 1.9 e_vs_sector_depth_diff_enhanced（板块深度差异）
        feat_val = self.rolling['sector_diff']['last']
        self.features[i, 8] = feat_val * FACTOR_WEIGHTS['e_vs_sector_depth_diff_enhanced']

    def _cache_tick(self, E_row, sector_rows):
        """缓存当前Tick数据（仅保留9个因子所需字段）"""
        if self.cache_idx >= self.max_cache:
            return
        
        # 缓存核心数据
        self.e_price[self.cache_idx] = E_row.get('LastPrice', 0.0)
        self.e_vol[self.cache_idx] = E_row.get('TradeBuyVolume', 0.0) + E_row.get('TradeSellVolume', 0.0)
        
        # 计算收益率
        if self.cache_idx > 0:
            self.e_return[self.cache_idx] = (self.e_price[self.cache_idx] - self.e_price[self.cache_idx-1]) / (self.e_price[self.cache_idx-1] + SAFE_DIV)
        else:
            self.e_return[self.cache_idx] = 0.0
        
        # ========== 缓存9个因子所需字段 ==========
        # 买盘深度（buy_depth_ratio_enhanced）
        buy_depth = (
            E_row.get('BidVolume1', 0.0) + E_row.get('BidVolume2', 0.0) +
            E_row.get('BidVolume3', 0.0) + E_row.get('BidVolume4', 0.0) +
            E_row.get('BidVolume5', 0.0)
        )
        sell_depth = (
            E_row.get('AskVolume1', 0.0) + E_row.get('AskVolume2', 0.0) +
            E_row.get('AskVolume3', 0.0) + E_row.get('AskVolume4', 0.0) +
            E_row.get('AskVolume5', 0.0)
        )
        total_depth = buy_depth + sell_depth + SAFE_DIV
        depth_ratio = buy_depth / total_depth
        alpha = 2/(TICK_PER_5MIN//5 + 1)
        self.rolling['buy_depth']['last'] = alpha * depth_ratio + (1 - alpha) * self.rolling['buy_depth']['last']
        
        # 板块深度差异（e_vs_sector_depth_diff_enhanced）
        sector_depth_ratios = []
        for s_row in sector_rows:
            s_buy_depth = (
                s_row.get('BidVolume1', 0.0) + s_row.get('BidVolume2', 0.0) +
                s_row.get('BidVolume3', 0.0) + s_row.get('BidVolume4', 0.0) +
                s_row.get('BidVolume5', 0.0)
            )
            s_sell_depth = (
                s_row.get('AskVolume1', 0.0) + s_row.get('AskVolume2', 0.0) +
                s_row.get('AskVolume3', 0.0) + s_row.get('AskVolume4', 0.0) +
                s_row.get('AskVolume5', 0.0)
            )
            s_total_depth = s_buy_depth + s_sell_depth + SAFE_DIV
            sector_depth_ratios.append(s_buy_depth / s_total_depth)
        sector_avg_depth = np.mean(sector_depth_ratios)
        depth_diff = depth_ratio - sector_avg_depth
        diff_std = np.std([self.rolling['sector_diff']['last'], depth_diff])
        smooth_window = 600 if diff_std > 0.05 else 1200
        alpha = 2/(smooth_window + 1)
        self.rolling['sector_diff']['last'] = alpha * depth_diff + (1 - alpha) * self.rolling['sector_diff']['last']
        
        # 个股-板块资金偏离（stock_sector_capital_dev）
        cd = self.rolling['capital_dev']
        e_buy_depth = buy_depth
        e_sell_depth = sell_depth
        cd['e_capital'] = (e_buy_depth - e_sell_depth) * self.e_price[self.cache_idx]
        # 计算板块平均主力资金
        sector_capital_list = []
        for s_row in sector_rows:
            s_buy_depth = (
                s_row.get('BidVolume1', 0.0) + s_row.get('BidVolume2', 0.0) +
                s_row.get('BidVolume3', 0.0) + s_row.get('BidVolume4', 0.0) +
                s_row.get('BidVolume5', 0.0)
            )
            s_sell_depth = (
                s_row.get('AskVolume1', 0.0) + s_row.get('AskVolume2', 0.0) +
                s_row.get('AskVolume3', 0.0) + s_row.get('AskVolume4', 0.0) +
                s_row.get('AskVolume5', 0.0)
            )
            s_price = s_row.get('LastPrice', 0.0)
            s_capital = (s_buy_depth - s_sell_depth) * s_price
            sector_capital_list.append(s_capital)
        cd['sector_avg_capital'] = np.mean(sector_capital_list)

        # 索引自增 + 增量更新因子
        self.cache_idx += 1
        self._update_rolling_features()

    def online_predict(self, E_row, sector_rows):
        """在线预测（9因子+加权+软阈值合并）"""
        if not E_row or not sector_rows or len(sector_rows) != 4:
            return 0.0
        
        # 缓存并增量更新因子
        self._cache_tick(E_row, sector_rows)
        i = self.cache_idx - 1
        
        # 获取当前Tick的9个加权特征（适配FEATURE_CONFIG顺序）
        current_feat = self.features[i:i+1, :]
        current_feat = clean_numeric_array(current_feat)
        
        # 双模型预测（低复杂度模型）
        try:
            # 方向模型：预测涨的概率（0-1）
            dir_pred = self.dir_model.predict(current_feat)[0]
            # 强度模型：预测涨跌幅度（保留正负，不再取绝对值）
            str_pred = self.str_model.predict(current_feat)[0]
            # 软阈值合并预测（0-1→-1到1连续值）
            final_pred = merge_double_predict(np.array([dir_pred]), np.array([str_pred]))[0]
            # 限制预测范围
            final_pred = np.clip(final_pred, -0.1, 0.1)
            return float(final_pred)
        except Exception as e:
            print(f"预测异常（Tick {i}）：{e}")
            return 0.0