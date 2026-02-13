import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from utils import DATA_DIR, OUTPUT_DIR, get_day_folders, clean_numeric_array
from MyModel import MyModel

def align_tick_data(day_data):
    """修复：按Time排序并填充，保障Tick时序性"""
    # 以E股票的Time为基准（核心标的），其他股票按Time匹配
    e_df = day_data['E'].sort_values('Time').reset_index(drop=True)
    aligned_list = []
    
    for idx, e_row in e_df.iterrows():
        time = e_row['Time']
        sector_rows = []
        # 按A/B/C/D顺序匹配同时间的板块数据
        for stock in ['A', 'B', 'C', 'D']:
            stock_df = day_data[stock]
            match_row = stock_df[stock_df['Time'] == time]
            if not match_row.empty:
                sector_rows.append(match_row.iloc[0].to_dict())
            else:
                # 修复：用前一个Tick数据填充，避免数据断裂
                prev_row = stock_df[stock_df['Time'] < time].tail(1)
                sector_rows.append(prev_row.iloc[0].to_dict() if not prev_row.empty else None)
        
        # 仅保留所有板块数据都存在的Tick
        if all(s is not None for s in sector_rows):
            aligned_list.append({
                'Time': time,
                'E': e_row.to_dict(),
                'sector': sector_rows
            })
    
    # 转换为按Time排序的字典（保障时序）
    aligned_data = {item['Time']: item for item in aligned_list}
    return aligned_data

def main():
    # 初始化模型
    model = MyModel()
    
    # 获取所有交易日文件夹
    days = get_day_folders(DATA_DIR)
    
    for day in ["5"]:  # 记得改回来
        print(f"\n处理交易日：{day}")
        # 加载当日数据
        day_path = os.path.join(DATA_DIR, day)
        day_data = {}
        valid_stocks = ['A', 'B', 'C', 'D', 'E']
        missing = False
        for stock in valid_stocks:
            file_path = os.path.join(day_path, f"{stock}.csv")
            if not os.path.exists(file_path):
                print(f"警告：缺失{day}/{stock}.csv，跳过该交易日")
                missing = True
                break
            df = pd.read_csv(file_path, encoding='utf-8')
            # 修复：先去重Time（避免重复Tick），再清洗数值
            df = df.drop_duplicates(subset=['Time']).sort_values('Time').reset_index(drop=True)
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = clean_numeric_array(df[col].values)
            day_data[stock] = df
        
        if missing:
            continue
        
        # 初始化模型状态（每日清空历史）
        model.reset()
        
        # 对齐Tick数据（保障时序）
        aligned_data = align_tick_data(day_data)
        if not aligned_data:
            print(f"警告：{day}无有效对齐数据，跳过")
            continue
        
        # 逐Tick预测（严格按Time排序）
        times = []
        preds = []
        labels = []
        sorted_times = sorted(aligned_data.keys())
        
        for time in sorted_times:
            tick_data = aligned_data[time]
            E_row = tick_data['E']
            sector_rows = tick_data['sector']
            
            # 预测并收集结果（捕获预测异常）
            try:
                pred = model.online_predict(E_row, sector_rows)
            except Exception as e:
                print(f"警告：Time={time}预测失败，原因：{str(e)}")
                pred = 0.0
            
            times.append(time)
            preds.append(pred)
            # 收集真实标签（修复：空值处理）
            label = E_row.get('Return5min', 0.0)
            label = 0.0 if np.isnan(label) else label
            labels.append(label)
        
        # 计算当日IC（修复：方差校验+空值处理）
        preds_arr = clean_numeric_array(np.array(preds))
        labels_arr = clean_numeric_array(np.array(labels))
        ic = 0.0
        if np.var(preds_arr) > 1e-8 and np.var(labels_arr) > 1e-8:
            ic, _ = pearsonr(preds_arr, labels_arr)
            ic = 0.0 if np.isnan(ic) else ic
        
        print(f"交易日{day} - IC值: {ic:.4f}")
        
        # 保存预测结果（修复：追加真实标签）
        output_df = pd.DataFrame({
            'Time': times,
            'E_Return5min_Pred': preds,
            'E_Return5min_True': labels
        })
        output_path = os.path.join(OUTPUT_DIR, f"{day}_pred.csv")
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"交易日{day}预测结果已保存至：{output_path}")
    
    print("\n所有交易日处理完成！")

if __name__ == "__main__":
    # 优化：仅关闭必要警告，保留关键错误提示
    pd.options.mode.chained_assignment = None
    np.seterr(divide='ignore', invalid='ignore')
    main()