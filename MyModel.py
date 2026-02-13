import numpy as np
import pandas as pd
import lightgbm as lgb
from utils import (
    MODEL_DIR, SAFE_DIV, TICK_PER_5MIN, ONLINE_SMOOTH_WINDOW,
    clean_numeric_array
)

class MyModel:
    def __init__(self):
        # 初始化模型
        self.model = None
        self.load_model()
        
        # 核心缓存（数组版，避免列表扩容）
        self.max_cache = 100000
        self.e_price = np.zeros(self.max_cache, dtype=np.float32)
        self.e_vol = np.zeros(self.max_cache, dtype=np.float32)
        self.e_return = np.zeros(self.max_cache, dtype=np.float32)
        self.cache_idx = 0
        
        # ========== 增量因子缓存（核心！）==========
        self.features = np.zeros((self.max_cache, 8), dtype=np.float32)
        
        # 增量计算需要的滚动状态
        self.rolling = {
            # 量价相关性
            'pv_corr': {'sum_xy':0.0, 'sum_x2':0.0, 'sum_y2':0.0, 'sum_x':0.0, 'sum_y':0.0, 'n':0},
            # 价格收敛率
            'price_conv': {'short_std':0.0, 'long_std':0.0, 'alpha_short':2/(TICK_PER_5MIN+1), 'alpha_long':2/(TICK_PER_5MIN*3+1)},
            # 成交量波动率
            'vol_vol': {'std':0.0, 'alpha':2/(TICK_PER_5MIN+1)},
            # 收益率波动率
            'ret_vol': {'std':0.0, 'alpha':2/(TICK_PER_5MIN+1)},
            # 量比
            'vol_ratio': {'ma_short':0.0, 'ma_long':0.0, 'alpha_short':2/(20+1), 'alpha_long':2/(100+1)},
            # 换手率
            'turnover': {'ma':0.0, 'alpha':2/(TICK_PER_5MIN+1)},
            # 买盘深度
            'buy_depth': {'last':0.0},
            # 板块差异
            'sector_diff': {'last':0.0}
        }

    def load_model(self):
        """加载模型（原版逻辑）"""
        model_path = f"{MODEL_DIR}/online_model.txt"
        try:
            self.model = lgb.Booster(model_file=model_path)
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{str(e)}")

    def reset(self):
        """每日重置：仅重置索引和滚动状态"""
        self.cache_idx = 0
        # 重置滚动状态
        for k in self.rolling:
            if k == 'pv_corr':
                self.rolling[k] = {'sum_xy':0.0, 'sum_x2':0.0, 'sum_y2':0.0, 'sum_x':0.0, 'sum_y':0.0, 'n':0}
            elif k in ['price_conv', 'vol_vol', 'ret_vol', 'vol_ratio', 'turnover']:
                for sk in self.rolling[k]:
                    if 'alpha' not in sk:
                        self.rolling[k][sk] = 0.0
            else:
                self.rolling[k]['last'] = 0.0

    def _update_rolling_features(self):
        """核心：增量更新所有因子（O(1)复杂度）"""
        i = self.cache_idx - 1
        if i < 1:  # 至少2个Tick才计算
            for f in range(8):
                self.features[i, f] = 0.0
            return
        
        # ========== 1. 量价相关性（price_vol_corr_pos） ==========
        pv = self.rolling['pv_corr']
        ret = self.e_return[i]
        vol = self.e_vol[i]
        
        # 窗口内增量更新sum
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
        
        # 计算相关系数
        if pv['n'] >= 2:
            cov = (pv['sum_xy'] - pv['sum_x']*pv['sum_y']/pv['n']) / pv['n']
            std_x = np.sqrt((pv['sum_x2'] - pv['sum_x']**2/pv['n']) / pv['n'])
            std_y = np.sqrt((pv['sum_y2'] - pv['sum_y']**2/pv['n']) / pv['n'])
            corr = cov / (std_x * std_y + SAFE_DIV)
            self.features[i, 0] = -np.abs(corr)  # 正向化
        else:
            self.features[i, 0] = 0.0

        # ========== 2. 价格收敛率（lastprice_vol_converge） ==========
        pc = self.rolling['price_conv']
        price_ret = self.e_return[i]
        # EWMA近似滚动标准差
        pc['short_std'] = pc['alpha_short'] * (price_ret**2) + (1 - pc['alpha_short']) * pc['short_std']
        pc['long_std'] = pc['alpha_long'] * pc['short_std'] + (1 - pc['alpha_long']) * pc['long_std']
        self.features[i, 1] = (pc['short_std'] / (pc['long_std'] + SAFE_DIV)) - 1.0

        # ========== 3. 成交量波动率（vol_volatility） ==========
        vv = self.rolling['vol_vol']
        vol_ret = (self.e_vol[i] - self.e_vol[i-1]) / (self.e_vol[i-1] + SAFE_DIV)
        vv['std'] = vv['alpha'] * (vol_ret**2) + (1 - vv['alpha']) * vv['std']
        self.features[i, 2] = np.sqrt(vv['std'])

        # ========== 4. 收益率波动率（return_volatility_pos） ==========
        rv = self.rolling['ret_vol']
        rv['std'] = rv['alpha'] * (self.e_return[i]**2) + (1 - rv['alpha']) * rv['std']
        self.features[i, 3] = np.sqrt(rv['std'])

        # ========== 5. 量比（short_vol_ratio） ==========
        vr = self.rolling['vol_ratio']
        vr['ma_short'] = vr['alpha_short'] * self.e_vol[i] + (1 - vr['alpha_short']) * vr['ma_short']
        vr['ma_long'] = vr['alpha_long'] * self.e_vol[i] + (1 - vr['alpha_long']) * vr['ma_long']
        self.features[i, 4] = vr['ma_short'] / (vr['ma_long'] + SAFE_DIV)

        # ========== 6. 换手率（daily_rel_turnover） ==========
        to = self.rolling['turnover']
        to['ma'] = to['alpha'] * self.e_vol[i] + (1 - to['alpha']) * to['ma']
        self.features[i, 5] = self.e_vol[i] / (to['ma'] + SAFE_DIV)

        # ========== 7. 买盘深度（buy_depth_ratio_enhanced） ==========
        # 从E_row提取（示例，按你原版逻辑适配）
        self.features[i, 6] = self.rolling['buy_depth']['last']

        # ========== 8. 板块差异（e_vs_sector_depth_diff_enhanced） ==========
        self.features[i, 7] = self.rolling['sector_diff']['last']

    def _cache_tick(self, E_row, sector_rows):
        """缓存当前Tick数据（数组版）"""
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
        
        # 更新买盘深度/板块差异（从传入的行提取）
        self.rolling['buy_depth']['last'] = E_row.get('BuyDepthRatio', 0.0) if 'BuyDepthRatio' in E_row else 0.0
        sector_depth = np.mean([s.get('BuyDepthRatio', 0.0) for s in sector_rows])
        self.rolling['sector_diff']['last'] = self.rolling['buy_depth']['last'] - sector_depth
        
        # 索引自增 + 增量更新因子
        self.cache_idx += 1
        self._update_rolling_features()

    def online_predict(self, E_row, sector_rows):
        """在线预测（增量版）"""
        if not E_row or not sector_rows or len(sector_rows) != 4:
            return 0.0
        
        # 缓存并增量更新因子
        self._cache_tick(E_row, sector_rows)
        i = self.cache_idx - 1
        
        # 获取当前Tick的特征
        current_feat = self.features[i:i+1, :]
        current_feat = clean_numeric_array(current_feat)
        
        # 模型预测
        try:
            pred = self.model.predict(current_feat)[0]
            pred = np.clip(pred, -0.1, 0.1)
            return float(pred)
        except Exception:
            return 0.0