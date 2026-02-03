import os
import time
import numpy as np
import pandas as pd
from MyModel import MyModel
from utils import get_day_folders, load_day_data_fast
import warnings
# 精准屏蔽特征名不匹配的警告
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names"
)

def run_test_fast():
    start_total = time.time()
    model = MyModel()
    data_path = "./data"
    days = get_day_folders(data_path)
    print(f"待预测交易日：{days}")

    # 提前定义时间解析函数（只定义一次，避免重复创建）
    def parse_time_num_batch(time_nums):
        """批量解析时间数字，返回时段特征数组（一次性处理整列）"""
        time_strs = np.char.zfill(time_nums.astype(str), 9)  # 批量补零
        hours = np.array([int(s[:2]) for s in time_strs])
        minutes = np.array([int(s[2:4]) for s in time_strs])
        time_mins = hours * 60 + minutes
        
        # 批量划分时段（用numpy向量化，比逐行if快10倍）
        time_periods = np.ones_like(time_mins, dtype=np.float32)  # 默认午盘
        time_periods[(time_mins >= 570) & (time_mins < 600)] = 0.0  # 早盘
        time_periods[(time_mins >= 870) & (time_mins < 900)] = 0.0  # 尾盘
        return time_periods

    for day in days:
        start_day = time.time()
        model.reset()
        day_data = load_day_data_fast(data_path, day)
        df_E = day_data['E']
        sector_dfs = [day_data['A'], day_data['B'], day_data['C'], day_data['D']]
        
        # ===================== 核心提速1：批量预处理所有数据 =====================
        # 1. 提取关键列并转numpy（避免循环内取值）
        df_E_np = df_E[['Time', 'TradeBuyVolume', 'TradeSellVolume', 'LastPrice']].values
        sector_np = [df[['TradeBuyVolume', 'TradeSellVolume', 'LastPrice']].values for df in sector_dfs]
        
        # 2. 批量解析所有Time列，生成时段特征（一次性处理，不用逐Tick解析）
        time_nums = df_E_np[:, 0].astype(int)  # 所有Time数字
        time_periods = parse_time_num_batch(time_nums)  # 批量生成时段特征
        
        # 3. 预提取E列的关键值（避免循环内重复取值）
        e_buy = df_E_np[:, 1]
        e_sell = df_E_np[:, 2]
        e_price = df_E_np[:, 3]
        
        # 4. 兜底填充异常值（批量处理）
        e_buy = np.where(np.isnan(e_buy), 0, e_buy)
        e_sell = np.where(np.isnan(e_sell), 0, e_sell)
        e_price = np.where((np.isnan(e_price) | (e_price == 0)), model.price_mean, e_price)
        e_volume = e_buy + e_sell
        
        preds = []
        # ===================== 核心提速2：简化循环内逻辑 =====================
        for idx in range(len(df_E)):
            # 直接从numpy数组取预处理后的值，不用逐行构造Series
            E_row = {
                'Time': time_nums[idx],
                'TradeBuyVolume': e_buy[idx],
                'TradeSellVolume': e_sell[idx],
                'LastPrice': e_price[idx],
                'time_period': time_periods[idx]  # 直接用批量生成的时段特征
            }
            
            # 简化sector_single构造（减少DataFrame创建）
            sector_single = []
            for i in range(4):
                sector_row = {
                    'TradeBuyVolume': sector_np[i][idx, 0] if not np.isnan(sector_np[i][idx, 0]) else 0,
                    'TradeSellVolume': sector_np[i][idx, 1] if not np.isnan(sector_np[i][idx, 1]) else 0,
                    'LastPrice': sector_np[i][idx, 2] if not np.isnan(sector_np[i][idx, 2]) else model.price_mean
                }
                sector_single.append(pd.DataFrame([sector_row]))
            
            # 调用预测（此时E_row已带预计算的时段特征，MyModel里可简化）
            pred = model.online_predict(E_row, sector_single)
            preds.append(pred)
        
        # 保存结果
        output_dir = f"./output/{day}"
        os.makedirs(output_dir, exist_ok=True)
        out_df = pd.DataFrame({
            'TickIndex': df_E.index,
            'PredictReturn5min': preds
        })
        out_df.to_csv(os.path.join(output_dir, "E.csv"), index=False, encoding='utf-8')
        
        day_cost = time.time() - start_day
        print(f"✅ 交易日 {day} 完成，耗时 {day_cost:.2f} 秒，预测{len(preds)}个Tick")

    total_cost = time.time() - start_total
    print(f"\n🎉 全部完成！总耗时 {total_cost:.2f} 秒")

if __name__ == "__main__":
    run_test_fast()