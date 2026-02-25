import os
import time
import gc
import numpy as np
import pandas as pd
from utils import DATA_DIR, OUTPUT_DIR, get_day_folders, clean_numeric_array
from MyModel import MyModel

def align_tick_data(day_data):
    """极致优化：返回数组而非字典，减少转换开销（新增Time列处理）"""
    start = time.time()
    # 处理E数据
    e_df = day_data['E'].sort_values('Time').reset_index(drop=True)
    e_cols = e_df.columns.tolist()
    e_vals = e_df.values.astype(np.float32)  # 单精度浮点，减少内存
    
    # 处理板块数据（提前构建索引+数组）
    sector_data = {}
    for stock in ['A', 'B', 'C', 'D']:
        df = day_data[stock].copy()
        df = df.sort_values('Time').reset_index(drop=True)
        # 提前构建Time索引数组
        time_index = df['Time'].values
        # 转换为数组（单精度）
        s_cols = df.columns.tolist()
        s_vals = df.values.astype(np.float32)
        sector_data[stock] = {
            'cols': s_cols,
            'vals': s_vals,
            'time_index': time_index,
            'time_to_idx': dict(zip(time_index, range(len(time_index))))  # 预构建Time→索引映射
        }
    print(f"✅ 对齐数据耗时：{time.time()-start:.2f}秒")
    return e_cols, e_vals, sector_data

def main():
    # 启用Pandas加速选项（核心优化）
    pd.set_option('compute.use_bottleneck', True)
    pd.set_option('compute.use_numexpr', True)
    gc.disable()  # 禁用GC，减少停顿
    
    # 初始化模型
    model = MyModel()
    
    # 获取所有交易日文件夹
    days = get_day_folders(DATA_DIR)
    
    # 可修改为需要预测的交易日（如["5"]或days）
    predict_days = ["4", "5"]
    if not predict_days:
        predict_days = [d for d in days if int(d) > 4]  # 预测5天及以后
    
    for day in predict_days:
        print(f"\n===== 开始处理交易日 {day}（10因子+时段类别特征+新加权系数） =====")
        start_day = time.time()
        
        # 1. 加载数据阶段
        start_load = time.time()
        day_path = os.path.join(DATA_DIR, day)
        day_data = {}
        valid_stocks = ['A', 'B', 'C', 'D', 'E']
        missing = False
        
        for stock in valid_stocks:
            file_path = os.path.join(day_path, f"{stock}.csv")
            if not os.path.exists(file_path):
                missing = True
                break
            
            # 极速CSV读取（低内存+向量化清洗）
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            df = df.sort_values('Time').reset_index(drop=True)
            
            # 批量清洗数值列（向量化，替代逐列循环）
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].apply(lambda x: clean_numeric_array(x.values))
            day_data[stock] = df
        
        if missing:
            print(f"❌ 交易日{day}缺失数据，跳过")
            continue
        print(f"✅ 加载+清洗数据耗时：{time.time()-start_load:.2f}秒")
        
        # 2. 模型重置+数据对齐
        model.reset()
        e_cols, e_vals, sector_data = align_tick_data(day_data)
        if len(e_vals) == 0:
            print(f"❌ 交易日{day}无E股数据，跳过")
            continue
        total_ticks = len(e_vals)
        print(f"✅ 待处理Tick总数：{total_ticks}")
        
        # 3. 逐Tick预测（极致优化：数组遍历+预构建映射）
        start_predict = time.time()
        ticktimes = np.zeros(total_ticks, dtype=np.int64)
        my_preds = np.zeros(total_ticks, dtype=np.float32)
        
        # 获取列索引（新增Time列必选）
        try:
            time_col_idx = e_cols.index('Time')
        except ValueError:
            print(f"❌ 交易日{day}的E股数据缺少Time列！")
            continue
        
        # 预创建字典（复用，减少创建开销）
        E_row = {col: 0.0 for col in e_cols}
        empty_sector_rows = {}
        for stock in ['A', 'B', 'C', 'D']:
            empty_sector_rows[stock] = {col: 0.0 for col in sector_data[stock]['cols']}
        
        # 逐Tick快速处理
        for idx in range(total_ticks):
            e_vals_row = e_vals[idx]
            tick_time = e_vals_row[time_col_idx]
            ticktimes[idx] = tick_time
            
            # 复用E_row字典
            for col_idx, col in enumerate(e_cols):
                E_row[col] = e_vals_row[col_idx]
            
            # 快速匹配板块数据
            sector_rows = []
            for stock in ['A', 'B', 'C', 'D']:
                s_data = sector_data[stock]
                s_row = empty_sector_rows[stock]
                if tick_time in s_data['time_to_idx']:
                    s_idx = s_data['time_to_idx'][tick_time]
                    s_vals_row = s_data['vals'][s_idx]
                    for col_idx, col in enumerate(s_data['cols']):
                        s_row[col] = s_vals_row[col_idx]
                else:
                    s_row['Time'] = tick_time
                sector_rows.append(s_row)
            
            # 增量预测（核心提速）
            pred = model.online_predict(E_row, sector_rows)
            my_preds[idx] = pred
            
            # 进度打印（每1000个Tick）
            if idx % 1000 == 0 and idx > 0:
                elapsed = time.time() - start_predict
                speed = idx / elapsed
                print(f"🔹 已处理{idx}/{total_ticks} Tick | 耗时{elapsed:.2f}秒 | 速度{speed:.2f} Tick/秒")
        
        # 4. 收尾阶段
        print(f"✅ 逐Tick预测总耗时：{time.time()-start_predict:.2f}秒")
        
        # 保存结果（仅保留Time和Predict列）
        start_save = time.time()
        output_day_dir = os.path.join(OUTPUT_DIR, day)
        os.makedirs(output_day_dir, exist_ok=True)
        output_csv_path = os.path.join(output_day_dir, "E.csv")
        
        out_frame = pd.DataFrame({
            'Time': ticktimes,
            'Predict': my_preds
        })
        out_frame.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"✅ 保存结果耗时：{time.time()-start_save:.2f}秒")
        print(f"\n===== 交易日{day}总耗时：{time.time()-start_day:.2f}秒 =====")
        
        gc.enable()  # 恢复GC

if __name__ == "__main__":
    # 关闭无关警告+启用加速
    pd.options.mode.chained_assignment = None
    np.seterr(divide='ignore', invalid='ignore')
    pd.set_option('mode.copy_on_write', True)  # 减少内存拷贝
    main()