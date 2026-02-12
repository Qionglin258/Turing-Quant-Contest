import os
import pandas as pd
from MyModel import MyModel

def main():
    """
    修复pandas频率兼容问题：50L → 50ms
    保留所有你的逻辑，仅修改时间生成的频率写法
    """
    # 1. 初始化参数
    model_dir = "./model_weights"
    data_root = "./data"
    output_root = "./output"

    # 2. 初始化模型（每日reset）
    model = MyModel(model_dir=model_dir)

    # 3. 遍历交易日（仅处理数字为5的文件夹，保留你的修改）
    days = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.strip().isdigit() and int(d) == 5]###5是预测集
    for day in days:
        model.reset()  # 每日初始化
        day_data_path = os.path.join(data_root, day)
        day_output_path = os.path.join(output_root, day)
        os.makedirs(day_output_path, exist_ok=True)
        
        if not os.path.exists(day_data_path):
            print(f"跳过不存在的交易日：{day}")
            continue
        
        # 4. 加载数据（适配无time列）
        e_file = os.path.join(day_data_path, "E.csv")
        a_file = os.path.join(day_data_path, "A.csv")
        b_file = os.path.join(day_data_path, "B.csv")
        c_file = os.path.join(day_data_path, "C.csv")
        d_file = os.path.join(day_data_path, "D.csv")
        
        try:
            E_df = pd.read_csv(e_file, encoding="utf-8")
            A_df = pd.read_csv(a_file, encoding="utf-8")
            B_df = pd.read_csv(b_file, encoding="utf-8")
            C_df = pd.read_csv(c_file, encoding="utf-8")
            D_df = pd.read_csv(d_file, encoding="utf-8")
            
            # 校验行数一致
            rows = len(E_df)
            if len(A_df) != rows or len(B_df) != rows or len(C_df) != rows or len(D_df) != rows:
                raise ValueError(f"交易日{day}数据行数不一致：E({rows})/A({len(A_df)})/B({len(B_df)})/C({len(C_df)})/D({len(D_df)})")
        
        except Exception as e:
            print(f"加载交易日{day}数据失败：{str(e)}")
            continue

        # 5. 逐Tick在线预测
        predictions = []
        for idx in range(rows):
            E_row = E_df.iloc[idx].to_dict()
            sector_rows = [A_df.iloc[idx].to_dict(), B_df.iloc[idx].to_dict(), C_df.iloc[idx].to_dict(), D_df.iloc[idx].to_dict()]
            pred = model.online_predict(E_row, sector_rows)
            predictions.append(pred)

        # 6. 生成time列（修复频率写法：50ms替代50L，兼容所有pandas版本）
        if "time" in E_df.columns:
            time_col = E_df["time"].astype(str).values
        else:
            # 核心修复：freq="50ms"（毫秒的标准写法）
            time_series = pd.date_range(start="09:30:00", periods=rows, freq="50ms")  
            # 格式化为8-9位纯数字（小时不补零）
            time_col = [
                f"{t.hour}{t.minute:02d}{t.second:02d}{t.microsecond//10000:02d}"
                for t in time_series
            ]
        
        # 7. 保存输出文件
        output_df = pd.DataFrame({
            "time": time_col,
            "prediction": predictions
        })
        output_e_file = os.path.join(day_output_path, "E.csv")
        output_df.to_csv(output_e_file, index=False, encoding="utf-8")
        print(f"交易日{day}预测完成，输出文件：{output_e_file}")
        print(f"示例time值：{time_col[:3]}（前3个Tick）")

if __name__ == "__main__":
    main()