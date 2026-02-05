import numpy as np
from utils import filter_warnings, get_day_folders, load_day_data
from MyModel import MyModel
import pandas as pd
import os
import traceback

# 过滤警告（保持和其他模块一致）
filter_warnings()

def run_test():
    # ========== 初始化模型 ==========
    try:
        model = MyModel()
    except Exception as e:
        print(f"模型初始化失败: {e}")
        traceback.print_exc()
        return

    # ========== 获取交易日列表 ==========
    try:
        days = get_day_folders("./data")
        print(f"待处理交易日: {days}")
    except Exception as e:
        print(f"获取交易日文件夹失败: {e}")
        traceback.print_exc()
        return

    # ========== 逐交易日在线预测 ==========
    output_root = "./output"
    os.makedirs(output_root, exist_ok=True)  # 确保输出根目录存在

    for day in days:
        try:
            # 每个交易日开始前重置模型缓存（关键修复）
            model.reset()
            
            # 加载当日数据（修复参数顺序）
            day_data = load_day_data(day, "./data")
            n_ticks = len(day_data['E'])

            if n_ticks == 0:
                print(f"警告: 交易日 {day} 无数据，跳过")
                continue

            # 提取时间戳（简化操作，避免冗余转置）
            ticktimes = day_data['E'].iloc[:, 0].values  # 假设第一列是时间
            my_preds = np.zeros(n_ticks, dtype=np.float32)

            # 逐Tick预测
            for tick_index in range(n_ticks):
                # 获取当前Tick的E数据和行业数据（A/B/C/D）
                E_row_data = day_data['E'].iloc[tick_index]
                sector_row_datas = [
                    day_data['A'].iloc[tick_index],
                    day_data['B'].iloc[tick_index],
                    day_data['C'].iloc[tick_index],
                    day_data['D'].iloc[tick_index]
                ]

                # 在线预测
                my_preds[tick_index] = model.online_predict(E_row_data, sector_row_datas)

            # ========== 保存预测结果 ==========
            day_output_dir = os.path.join(output_root, day)
            os.makedirs(day_output_dir, exist_ok=True)  # 简化目录创建逻辑

            # 构造输出DataFrame（保证数据类型正确）
            out_frame = pd.DataFrame({
                'Time': ticktimes,
                'Predict': my_preds
            })
            # 保存CSV（指定编码避免乱码）
            out_path = os.path.join(day_output_dir, "E.csv")
            out_frame.to_csv(out_path, index=False, encoding="utf-8")
            
            print(f"成功处理交易日 {day}，结果保存至: {out_path}")

        except Exception as e:
            print(f"处理交易日 {day} 失败: {e}")
            traceback.print_exc()
            continue

    print("所有交易日处理完成！")

if __name__ == '__main__':
    # 主程序入口
    run_test()