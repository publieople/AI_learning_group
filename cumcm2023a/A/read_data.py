import pandas as pd
import numpy as np
import math
import os
from datetime import datetime

# 读取附件数据
def read_attachment_data():
    """读取附件中的定日镜位置数据
    
    Returns:
        DataFrame: 包含定日镜位置的数据框
    """
    file_path = '../附件.xlsx'
    try:
        df = pd.read_excel(file_path)
        print(f"成功读取附件数据，共{len(df)}行")
        print("数据列名:", df.columns.tolist())
        print("数据前5行:")
        print(df.head())
        return df
    except Exception as e:
        print(f"读取附件数据失败: {e}")
        return None

# 主函数
if __name__ == "__main__":
    # 读取附件数据
    mirrors_data = read_attachment_data()
    
    # 如果成功读取数据，打印一些基本统计信息
    if mirrors_data is not None:
        print("\n基本统计信息:")
        print(f"定日镜总数: {len(mirrors_data)}")
        
        # 检查是否有缺失值
        missing_values = mirrors_data.isnull().sum()
        print("\n缺失值统计:")
        print(missing_values)
        
        # 不假设列名，而是直接打印所有列的统计信息
        print("\n各列统计信息:")
        print(mirrors_data.describe())