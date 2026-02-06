import os
import pandas as pd
import numpy as np
from utils import (
    DATA_PATH,  # 替换原来的DATA_DIR
    MODEL_DIR,  # 路径统一
    get_day_folders,
    load_day_data,
    evaluate_ic,
    PRICE_CLIP_RANGE  # 补充必要导入
)
from MyModel import MyModel

# 统一输出目录（和utils.py路径风格一致）
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """主流程：逐天加载数据→在线预测→保存结果→计算IC"""
    # 初始化模型
    model = MyModel(model_dir=MODEL_DIR)
    # 获取交易日列表
    ###day_folders = get_day_folders(data_path=DATA_PATH)
    day_folders=["5"]  # 仅测试集交易日5
    all_ic = []
    
    for day in day_folders:
        print(f"\n========== 处理交易日：{day} ==========")
        # 每个交易日重置模型状态
        model.reset()
        # 加载当日数据
        try:
            data = load_day_data(day=day, data_path=DATA_PATH)
        except Exception as e:
            print(f"❌ 加载{day}数据失败：{e}")
            continue
        
        # 提取E股票和板块股票数据
        e_df = data["E"].copy()
        sector_stocks = ["A", "B", "C", "D"]
        n_ticks = len(e_df)
        predictions = []
        
        # 逐Tick在线预测
        for idx in range(n_ticks):
            # 提取E股票当前Tick数据
            e_row = e_df.iloc[idx].to_dict()
            # 提取板块股票当前Tick数据
            sector_rows = []
            for stock in sector_stocks:
                sector_rows.append(data[stock].iloc[idx].to_dict())
            
            # 调用模型预测
            try:
                pred = model.online_predict(E_row=e_row, sector_row_datas=sector_rows)
                predictions.append(pred)
            except Exception as e:
                print(f"⚠️  Tick{idx}预测失败：{e}")
                predictions.append(0.0)
        
        # 保存预测结果
        output_day_dir = os.path.join(OUTPUT_DIR, day)
        os.makedirs(output_day_dir, exist_ok=True)
        output_path = os.path.join(output_day_dir, "E.csv")
        
        # 构造结果文件（和你的check.py列名匹配）
        result_df = pd.DataFrame({
            "Time": e_df.get("time_num", range(n_ticks)),  # 兼容time_num列
            "Return5min": e_df["Return5min"].values,       # 真实值
            "pred": predictions                             # 预测值（和check.py的PRED_COL一致）
        })
        result_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ 预测结果已保存：{output_path}")
        
        # 计算当日IC
        if "Return5min" in e_df.columns and len(predictions) == len(e_df):
            y_true = e_df["Return5min"].values
            y_pred = np.array(predictions)
            ic = evaluate_ic(y_true=y_true, y_pred=y_pred)
            all_ic.append(ic)
            print(f"📊 交易日{day} IC值：{ic:.4f}")
        else:
            print(f"⚠️  {day}无有效IC数据")
    
    # 输出整体结果
    print("\n========== 所有交易日结果汇总 ==========")
    if all_ic:
        avg_ic = np.mean(all_ic)
        print(f"📈 平均IC值：{avg_ic:.4f}")
        print(f"📊 IC列表：{[round(x,4) for x in all_ic]}")
    else:
        print("❌ 无有效IC数据")

if __name__ == "__main__":
    main()