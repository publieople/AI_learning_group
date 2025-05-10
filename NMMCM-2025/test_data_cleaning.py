#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning import DataCleaner

def test_data_cleaning():
    """测试数据清洗效果"""
    # 创建数据清洗器
    cleaner = DataCleaner()

    # 测试气象数据清洗
    test_meteorological_cleaning(cleaner)

    # 测试轨道交通数据清洗
    test_subway_traffic_cleaning(cleaner)

    # 测试马拉松赛历数据清洗
    test_marathon_history_cleaning(cleaner)

def test_meteorological_cleaning(cleaner):
    """测试气象数据清洗效果"""
    print("\n==== 测试气象数据清洗 ====")

    # 加载原始数据
    raw_data = cleaner.load_data("附件1_meteorological_data.csv")
    if raw_data is None:
        print("无法加载原始气象数据")
        return

    # 清洗数据
    cleaned_data = cleaner.clean_meteorological_data()
    if cleaned_data is None:
        print("数据清洗失败")
        return

    # 对比缺失值
    raw_missing = raw_data.isnull().sum().sum()
    cleaned_missing = cleaned_data.isnull().sum().sum()
    print(f"原始数据缺失值: {raw_missing}")
    print(f"清洗后数据缺失值: {cleaned_missing}")
    print(f"减少缺失值: {raw_missing - cleaned_missing}")

    # 对比异常值
    for col in ['temperature', 'wind_speed', 'precipitation']:
        if col in raw_data.columns and col in cleaned_data.columns:
            # 原始数据的异常值
            q1_raw = raw_data[col].quantile(0.25)
            q3_raw = raw_data[col].quantile(0.75)
            iqr_raw = q3_raw - q1_raw
            lower_bound_raw = q1_raw - 1.5 * iqr_raw
            upper_bound_raw = q3_raw + 1.5 * iqr_raw
            outliers_raw = raw_data[(raw_data[col] < lower_bound_raw) | (raw_data[col] > upper_bound_raw)].shape[0]

            # 清洗后数据的异常值
            q1_clean = cleaned_data[col].quantile(0.25)
            q3_clean = cleaned_data[col].quantile(0.75)
            iqr_clean = q3_clean - q1_clean
            lower_bound_clean = q1_clean - 1.5 * iqr_clean
            upper_bound_clean = q3_clean + 1.5 * iqr_clean
            outliers_clean = cleaned_data[(cleaned_data[col] < lower_bound_clean) | (cleaned_data[col] > upper_bound_clean)].shape[0]

            print(f"{col} 原始异常值: {outliers_raw}")
            print(f"{col} 清洗后异常值: {outliers_clean}")
            print(f"{col} 减少异常值: {outliers_raw - outliers_clean}")

    # 绘制清洗前后对比图
    try:
        plt.figure(figsize=(15, 10))

        # 温度对比
        plt.subplot(2, 2, 1)
        sns.histplot(raw_data['temperature'].dropna(), color='blue', alpha=0.5, label='原始数据')
        sns.histplot(cleaned_data['temperature'].dropna(), color='red', alpha=0.5, label='清洗后数据')
        plt.title('温度分布对比')
        plt.legend()

        # 风速对比
        plt.subplot(2, 2, 2)
        sns.histplot(raw_data['wind_speed'].dropna(), color='blue', alpha=0.5, label='原始数据')
        sns.histplot(cleaned_data['wind_speed'].dropna(), color='red', alpha=0.5, label='清洗后数据')
        plt.title('风速分布对比')
        plt.legend()

        # 降水量对比
        plt.subplot(2, 2, 3)
        sns.histplot(raw_data['precipitation'].dropna(), color='blue', alpha=0.5, label='原始数据')
        sns.histplot(cleaned_data['precipitation'].dropna(), color='red', alpha=0.5, label='清洗后数据')
        plt.title('降水量分布对比')
        plt.legend()

        # 气压对比
        plt.subplot(2, 2, 4)
        sns.histplot(raw_data['pressure'].dropna(), color='blue', alpha=0.5, label='原始数据')
        sns.histplot(cleaned_data['pressure'].dropna(), color='red', alpha=0.5, label='清洗后数据')
        plt.title('气压分布对比')
        plt.legend()

        plt.tight_layout()
        plt.savefig('气象数据清洗效果对比.png')
        plt.close()
        print("已保存气象数据清洗效果对比图")
    except Exception as e:
        print(f"绘图时出错: {e}")

def test_subway_traffic_cleaning(cleaner):
    """测试轨道交通数据清洗效果"""
    print("\n==== 测试轨道交通数据清洗 ====")

    # 加载原始数据
    raw_data = cleaner.load_data("附件2_subway_traffic.csv")
    if raw_data is None:
        print("无法加载原始轨道交通数据")
        return

    # 清洗数据
    cleaned_data = cleaner.clean_subway_traffic_data()
    if cleaned_data is None:
        print("数据清洗失败")
        return

    # 对比缺失值
    raw_missing = raw_data.isnull().sum().sum()
    cleaned_missing = cleaned_data.isnull().sum().sum()
    print(f"原始数据缺失值: {raw_missing}")
    print(f"清洗后数据缺失值: {cleaned_missing}")
    print(f"减少缺失值: {raw_missing - cleaned_missing}")

    # 对比异常值
    for col in ['客运量（万人次）', '运营里程（公里）', '客运强度（万人次每公里日）']:
        if col in raw_data.columns and col in cleaned_data.columns:
            # 原始数据的异常值
            q1_raw = raw_data[col].quantile(0.25)
            q3_raw = raw_data[col].quantile(0.75)
            iqr_raw = q3_raw - q1_raw
            lower_bound_raw = q1_raw - 1.5 * iqr_raw
            upper_bound_raw = q3_raw + 1.5 * iqr_raw
            outliers_raw = raw_data[(raw_data[col] < lower_bound_raw) | (raw_data[col] > upper_bound_raw)].shape[0]

            # 清洗后数据的异常值
            q1_clean = cleaned_data[col].quantile(0.25)
            q3_clean = cleaned_data[col].quantile(0.75)
            iqr_clean = q3_clean - q1_clean
            lower_bound_clean = q1_clean - 1.5 * iqr_clean
            upper_bound_clean = q3_clean + 1.5 * iqr_clean
            outliers_clean = cleaned_data[(cleaned_data[col] < lower_bound_clean) | (cleaned_data[col] > upper_bound_clean)].shape[0]

            print(f"{col} 原始异常值: {outliers_raw}")
            print(f"{col} 清洗后异常值: {outliers_clean}")
            print(f"{col} 减少异常值: {outliers_raw - outliers_clean}")

    # 绘制清洗前后对比图
    try:
        plt.figure(figsize=(15, 10))

        # 客运量对比
        plt.subplot(2, 2, 1)
        sns.boxplot(y=raw_data['客运量（万人次）'].dropna(), color='blue', label='原始数据')
        plt.title('原始客运量箱线图')

        plt.subplot(2, 2, 2)
        sns.boxplot(y=cleaned_data['客运量（万人次）'].dropna(), color='red', label='清洗后数据')
        plt.title('清洗后客运量箱线图')

        # 运营里程对比
        plt.subplot(2, 2, 3)
        sns.boxplot(y=raw_data['运营里程（公里）'].dropna(), color='blue', label='原始数据')
        plt.title('原始运营里程箱线图')

        plt.subplot(2, 2, 4)
        sns.boxplot(y=cleaned_data['运营里程（公里）'].dropna(), color='red', label='清洗后数据')
        plt.title('清洗后运营里程箱线图')

        plt.tight_layout()
        plt.savefig('轨道交通数据清洗效果对比.png')
        plt.close()
        print("已保存轨道交通数据清洗效果对比图")
    except Exception as e:
        print(f"绘图时出错: {e}")

def test_marathon_history_cleaning(cleaner):
    """测试马拉松赛历数据清洗效果"""
    print("\n==== 测试马拉松赛历数据清洗 ====")

    # 加载原始数据
    raw_data = cleaner.load_data("附件12_marathon_history.csv")
    if raw_data is None:
        print("无法加载原始马拉松赛历数据")
        return

    # 清洗数据
    cleaned_data = cleaner.clean_marathon_history_data()
    if cleaned_data is None:
        print("数据清洗失败")
        return

    # 对比缺失值
    raw_missing = raw_data.isnull().sum().sum()
    cleaned_missing = cleaned_data.isnull().sum().sum()
    print(f"原始数据缺失值: {raw_missing}")
    print(f"清洗后数据缺失值: {cleaned_missing}")
    print(f"减少缺失值: {raw_missing - cleaned_missing}")

    # 对比异常值
    for col in ['报名人数', '完赛人数', '报名费']:
        if col in raw_data.columns and col in cleaned_data.columns:
            # 原始数据的异常值
            q1_raw = raw_data[col].quantile(0.25)
            q3_raw = raw_data[col].quantile(0.75)
            iqr_raw = q3_raw - q1_raw
            lower_bound_raw = q1_raw - 1.5 * iqr_raw
            upper_bound_raw = q3_raw + 1.5 * iqr_raw
            outliers_raw = raw_data[(raw_data[col] < lower_bound_raw) | (raw_data[col] > upper_bound_raw)].shape[0]

            # 清洗后数据的异常值
            q1_clean = cleaned_data[col].quantile(0.25)
            q3_clean = cleaned_data[col].quantile(0.75)
            iqr_clean = q3_clean - q1_clean
            lower_bound_clean = q1_clean - 1.5 * iqr_clean
            upper_bound_clean = q3_clean + 1.5 * iqr_clean
            outliers_clean = cleaned_data[(cleaned_data[col] < lower_bound_clean) | (cleaned_data[col] > upper_bound_clean)].shape[0]

            print(f"{col} 原始异常值: {outliers_raw}")
            print(f"{col} 清洗后异常值: {outliers_clean}")
            print(f"{col} 减少异常值: {outliers_raw - outliers_clean}")

    # 绘制清洗前后对比图
    try:
        plt.figure(figsize=(15, 10))

        # 报名人数对比
        plt.subplot(2, 2, 1)
        sns.histplot(raw_data['报名人数'].dropna(), color='blue', alpha=0.5, label='原始数据')
        sns.histplot(cleaned_data['报名人数'].dropna(), color='red', alpha=0.5, label='清洗后数据')
        plt.title('报名人数分布对比')
        plt.legend()

        # 完赛人数对比
        plt.subplot(2, 2, 2)
        sns.histplot(raw_data['完赛人数'].dropna(), color='blue', alpha=0.5, label='原始数据')
        sns.histplot(cleaned_data['完赛人数'].dropna(), color='red', alpha=0.5, label='清洗后数据')
        plt.title('完赛人数分布对比')
        plt.legend()

        # 完赛率对比
        if '完赛率' in raw_data.columns and '完赛率' in cleaned_data.columns:
            plt.subplot(2, 2, 3)
            sns.histplot(raw_data['完赛率'].dropna(), color='blue', alpha=0.5, label='原始数据')
            sns.histplot(cleaned_data['完赛率'].dropna(), color='red', alpha=0.5, label='清洗后数据')
            plt.title('完赛率分布对比')
            plt.legend()

        # 报名费对比
        plt.subplot(2, 2, 4)
        sns.histplot(raw_data['报名费'].dropna(), color='blue', alpha=0.5, label='原始数据')
        sns.histplot(cleaned_data['报名费'].dropna(), color='red', alpha=0.5, label='清洗后数据')
        plt.title('报名费分布对比')
        plt.legend()

        plt.tight_layout()
        plt.savefig('马拉松赛历数据清洗效果对比.png')
        plt.close()
        print("已保存马拉松赛历数据清洗效果对比图")
    except Exception as e:
        print(f"绘图时出错: {e}")

if __name__ == "__main__":
    test_data_cleaning()