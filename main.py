# 保留你原有所有导入/逻辑，只改「删除IC反转」
import os
import pandas as pd
from utils import get_day_folders, load_day_data
from MyModel import MyModel

# 保留你原有路径配置
DATA_PATH = "./data"
OUTPUT_PATH = "./output"
MODEL_DIR = "./model_weights"

def main():
    # 1. 初始化模型（保留）
    model = MyModel(model_dir=MODEL_DIR)
    
    # 2. 获取交易日列表（保留）
    days = get_day_folders(DATA_PATH)
    
    # 3. 遍历交易日（保留）
    for day in days:
        print(f"处理交易日 {day}...")
        # 加载数据（保留）
        day_data = load_day_data(day, DATA_PATH)
        df_E = day_data["E"]
        sector_dfs = [day_data["A"], day_data["B"], day_data["C"], day_data["D"]]
        
        # 重置模型缓存（保留）
        model.reset()
        
        # 逐Tick预测（保留循环结构，只删反转）
        preds = []
        for idx in range(len(df_E)):
            e_row = df_E.iloc[idx]
            sector_rows = [df.iloc[idx] for df in sector_dfs]
            
            # 调用模型预测（核心：无任何反转）
            pred = model.predict(e_row, sector_rows)
            preds.append(pred)
        
        # 保存结果（保留）
        df_E["pred"] = preds
        output_day_dir = os.path.join(OUTPUT_PATH, day)
        os.makedirs(output_day_dir, exist_ok=True)
        df_E[["pred"]].to_csv(os.path.join(output_day_dir, "E.csv"), index=False)
        print(f"交易日 {day} 完成")

if __name__ == "__main__":
    main()