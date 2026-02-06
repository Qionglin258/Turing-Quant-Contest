import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_ic_mse_r2(
    true_csv_path,    # 真实值CSV文件路径
    pred_csv_path,    # 预测值CSV文件路径
    true_col: str,    # 真实收益率列名（比如"Return5min"）
    pred_col: str,    # 预测收益率列名（比如"pred_Return5min"）
    drop_na: bool = True  # 是否删除空值，默认开启
):
    """
    计算真实值与预测值的IC、MSE、R²
    :param true_csv_path: 真实值CSV路径，如"./data/6_E.csv"
    :param pred_csv_path: 预测值CSV路径，如"./output/6_E_pred.csv"
    :param true_col: 真实收益率列名，需和CSV中一致
    :param pred_col: 预测收益率列名，需和CSV中一致
    :param drop_na: 是否删除空值行，默认True
    :return: 字典形式的指标结果
    """
    # 1. 读取CSV文件
    try:
        df_true = pd.read_csv(true_csv_path)
        df_pred = pd.read_csv(pred_csv_path)
        print("✅ 成功读取两个CSV文件")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"❌ 文件不存在：{e}")
    except Exception as e:
        raise Exception(f"❌ 读取CSV失败：{e}")

    # 2. 校验列名是否存在
    if true_col not in df_true.columns:
        raise ValueError(f"❌ 真实值CSV中无列名：{true_col}，请检查列名是否正确")
    if pred_col not in df_pred.columns:
        raise ValueError(f"❌ 预测值CSV中无列名：{pred_col}，请检查列名是否正确")
    print("✅ 列名校验通过")

    # 3. 提取收益率列并对齐长度（按行取交集，确保一一对应）
    true_vals = df_true[true_col].values
    pred_vals = df_pred[pred_col].values
    if len(true_vals) != len(pred_vals):
        print(f"⚠️  警告：真实值({len(true_vals)})和预测值({len(pred_vals)})行数不一致，将按短的对齐")
        min_len = min(len(true_vals), len(pred_vals))
        true_vals = true_vals[:min_len]
        pred_vals = pred_vals[:min_len]

    # 4. 处理空值
    df_temp = pd.DataFrame({"true": true_vals, "pred": pred_vals})
    if drop_na:
        df_temp = df_temp.dropna()
        if len(df_temp) == 0:
            raise ValueError("❌ 处理空值后无有效数据，请检查CSV中是否有非空的收益率值")
        print(f"✅ 空值处理完成，剩余有效数据行数：{len(df_temp)}")
    true_vals = df_temp["true"].values
    pred_vals = df_temp["pred"].values

    # 5. 校验是否为常量列（常量列无计算意义）
    if len(set(true_vals)) == 1:
        raise ValueError(f"❌ 真实值列{true_col}为常量，无法计算指标")
    if len(set(pred_vals)) == 1:
        raise ValueError(f"❌ 预测值列{pred_col}为常量，无法计算指标")

    # 6. 计算三个核心指标
    # IC值：用皮尔逊相关系数（股票预测中IC的标准计算方式）
    ic, _ = pearsonr(true_vals, pred_vals)
    # MSE均方误差
    mse = mean_squared_error(true_vals, pred_vals)
    # R²决定系数
    r2 = r2_score(true_vals, pred_vals)

    # 7. 整理结果
    result = {
        "IC值(信息系数)": round(ic, 4),
        "MSE(均方误差)": round(mse, 6),
        "R²(决定系数)": round(r2, 4),
        "有效数据行数": len(df_temp)
    }
    return result

# 主函数：运行脚本时调用
if __name__ == "__main__":
    # ===================== 这里是需要你手动修改的参数 =====================
    TRUE_CSV = "./data/5/E.csv"    # 你的真实值CSV文件路径
    PRED_CSV = "./output/5/E.csv"    # 你的预测值CSV文件路径
    TRUE_COL = "Return5min"         # 真实收益率列名（改成你CSV中的列名，比如"利润率"）
    PRED_COL = "Predict"    # 预测收益率列名（改成你CSV中的列名）
    # ======================================================================

    # 计算指标并打印结果
    try:
        metrics = calculate_ic_mse_r2(TRUE_CSV, PRED_CSV, TRUE_COL, PRED_COL)
        print("\n" + "="*50)
        print("📊 收益率指标计算结果：")
        for k, v in metrics.items():
            print(f"✅ {k}：{v}")
        print("="*50)
    except Exception as e:
        print(f"\n❌ 计算失败：{e}")