#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor
from data_cleaning import DataCleaner

def main():
    """
    主程序：运行数据预处理和清洗流程
    """
    print("===== 马拉松经济高质量发展数据处理 =====")

    # 第一步：数据预处理
    # 检查是否已经存在预处理数据
    if os.path.exists("processed_data") and len(os.listdir("processed_data")) > 0:
        print("\n预处理数据已存在，是否重新进行预处理？(y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            run_preprocessing()
        else:
            print("跳过预处理步骤")
    else:
        run_preprocessing()

    # 第二步：数据清洗
    # 检查是否已经存在清洗数据
    if os.path.exists("processed_data/cleaned") and len(os.listdir("processed_data/cleaned")) > 0:
        print("\n清洗后的数据已存在，是否重新进行数据清洗？(y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            run_data_cleaning()
        else:
            print("跳过数据清洗步骤")
    else:
        run_data_cleaning()

    # 第三步：测试数据质量
    print("\n是否进行数据质量测试？(y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        test_data_quality()

    print("\n===== 数据处理完成 =====")
    print("预处理数据保存在: processed_data/")
    print("清洗后的数据保存在: processed_data/cleaned/")

def run_preprocessing():
    """运行数据预处理"""
    print("\n开始数据预处理...")

    # 创建数据预处理器
    preprocessor = DataPreprocessor(base_path="附件")

    # 运行预处理
    processed_data = preprocessor.run_all_preprocessing()

    print("数据预处理完成")

    # 打印预处理后的数据统计信息
    print("\n预处理数据统计:")
    for key, data in processed_data.items():
        if key != 'cleaned_data' and data is not None:
            if isinstance(data, pd.DataFrame):
                print(f"{key}: {data.shape[0]} 行, {data.shape[1]} 列")
            elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], np.ndarray):
                print(f"{key}: 数组形状 {data[0].shape}")
            else:
                print(f"{key}: 类型 {type(data)}")

def run_data_cleaning():
    """运行数据清洗"""
    print("\n开始数据清洗...")

    # 创建数据清洗器
    cleaner = DataCleaner(data_dir="processed_data")

    # 运行数据清洗
    cleaned_data = cleaner.clean_all_data()

    print("数据清洗完成")

    # 打印清洗后的数据统计信息
    print("\n清洗后数据统计:")
    for key, data in cleaned_data.items():
        if data is not None and isinstance(data, pd.DataFrame):
            print(f"{key}: {data.shape[0]} 行, {data.shape[1]} 列")

def test_data_quality():
    """测试数据质量"""
    print("\n开始数据质量测试...")

    # 导入测试模块
    from test_data_cleaning import test_data_cleaning

    # 运行测试
    test_data_cleaning()

    print("数据质量测试完成")

if __name__ == "__main__":
    main()