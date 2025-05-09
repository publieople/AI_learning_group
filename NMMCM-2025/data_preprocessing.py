import pandas as pd
import numpy as np
import os
import glob
import re
import datetime

# 附件1：气象数据处理
def read_weather_data(file_path):
    """读取单个气象数据文件

    Args:
        file_path: 气象数据文件路径

    Returns:
        DataFrame: 包含气象数据的DataFrame
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析数据
    data = []
    for line in lines:
        if line.strip() and not line.startswith('#'):
            data.append(line.strip().split())

    # 确保有数据可以处理
    if len(data) < 2:
        print(f"警告: 文件 {file_path} 中没有足够的数据行")
        return pd.DataFrame()

    # 转换为DataFrame
    try:
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return pd.DataFrame()

def process_weather_data(base_path):
    """处理附件1的气象数据

    Args:
        base_path: 附件1的根目录路径

    Returns:
        DataFrame: 合并后的气象数据
    """
    print("开始处理气象数据...")

    # 检查路径是否存在
    if not os.path.exists(base_path):
        print(f"错误: 路径 {base_path} 不存在")
        return pd.DataFrame()

    # 获取所有数据文件
    all_files = glob.glob(os.path.join(base_path, '*', '*'))
    if not all_files:
        print(f"警告: 在 {base_path} 下未找到任何文件")
        return pd.DataFrame()

    print(f"找到 {len(all_files)} 个气象数据文件")

    # 处理每个文件
    all_data = []
    for file in all_files:
        print(f"处理文件: {file}")
        df = read_weather_data(file)

        if df.empty:
            continue

        # 提取年份和站点信息
        year_match = re.search(r'\\(\d{4})\\', file)
        year = year_match.group(1) if year_match else 'unknown'

        station_match = re.search(r'\\([^\\]+)$', file)
        station = os.path.basename(file) if not station_match else station_match.group(1)

        # 添加年份和站点列
        df['年份'] = year
        df['站点'] = station

        # 提取关键气象要素
        # 根据数据格式调整列名
        weather_columns = {
            '气温': ['TEM', 'TEMP', 'T', '温度', 'Temperature'],
            '气压': ['PRS', 'PRES', 'P', '气压', 'Pressure'],
            '露点': ['DPT', 'DEW', 'D', '露点', 'Dew Point'],
            '风向': ['WIN_D', 'WD', '风向', 'Wind Direction'],
            '风速': ['WIN_S', 'WS', '风速', 'Wind Speed'],
            '云量': ['CLD', 'CLOUD', 'C', '云量', 'Cloud Cover'],
            '降水量': ['PRE', 'RAIN', 'R', '降水', 'Precipitation']
        }

        # 标准化列名
        for standard_name, possible_names in weather_columns.items():
            for col in df.columns:
                if any(possible_name in col for possible_name in possible_names):
                    df.rename(columns={col: standard_name}, inplace=True)
                    break

        all_data.append(df)

    # 合并所有数据
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # 保存处理后的数据
        output_path = os.path.join(os.path.dirname(base_path), 'processed_weather_data.csv')
        combined_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"气象数据处理完成，已保存至: {output_path}")

        return combined_df
    else:
        print("警告: 没有有效的气象数据可以处理")
        return pd.DataFrame()

# 附件2：轨道交通客运量数据处理
def process_transit_data(file_path):
    """处理附件2的轨道交通客运量数据

    Args:
        file_path: 轨道交通数据文件路径

    Returns:
        DataFrame: 处理后的轨道交通数据
    """
    print("开始处理轨道交通客运量数据...")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return pd.DataFrame()

    try:
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            print(f"错误: 不支持的文件格式 {file_path}")
            return pd.DataFrame()

        # 数据清洗和转换
        # 1. 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')

        # 2. 标准化列名
        df.columns = [col.strip() for col in df.columns]

        # 3. 转换数据类型
        for col in df.columns:
            if '日期' in col or '时间' in col:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
            elif df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass

        # 保存处理后的数据
        output_path = os.path.join(os.path.dirname(file_path), 'processed_transit_data.csv')
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"轨道交通数据处理完成，已保存至: {output_path}")

        return df
    except Exception as e:
        print(f"处理轨道交通数据时出错: {str(e)}")
        return pd.DataFrame()

# 附件3：人口普查数据处理
def process_census_data(file_path):
    """处理附件3的人口普查数据

    Args:
        file_path: 人口普查数据文件路径

    Returns:
        DataFrame: 处理后的人口普查数据
    """
    print("开始处理人口普查数据...")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return pd.DataFrame()

    try:
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            print(f"错误: 不支持的文件格式 {file_path}")
            return pd.DataFrame()

        # 数据清洗和转换
        # 1. 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')

        # 2. 标准化列名
        df.columns = [col.strip() for col in df.columns]

        # 3. 计算人口密度（如果有相关数据）
        if '人口' in df.columns and '面积' in df.columns:
            df['人口密度'] = df['人口'] / df['面积']

        # 保存处理后的数据
        output_path = os.path.join(os.path.dirname(file_path), 'processed_census_data.csv')
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"人口普查数据处理完成，已保存至: {output_path}")

        return df
    except Exception as e:
        print(f"处理人口普查数据时出错: {str(e)}")
        return pd.DataFrame()

# 附件12：历史报名人数数据处理
def process_registration_data(file_path):
    """处理附件12的历史报名人数数据

    Args:
        file_path: 历史报名人数数据文件路径

    Returns:
        DataFrame: 处理后的历史报名人数数据
    """
    print("开始处理历史报名人数数据...")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return pd.DataFrame()

    try:
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            print(f"错误: 不支持的文件格式 {file_path}")
            return pd.DataFrame()

        # 数据清洗和转换
        # 1. 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')

        # 2. 标准化列名
        df.columns = [col.strip() for col in df.columns]

        # 3. 计算增长率
        if '年份' in df.columns and '报名人数' in df.columns:
            df = df.sort_values('年份')
            df['报名人数增长率'] = df['报名人数'].pct_change() * 100
            df['报名人数增长率'] = df['报名人数增长率'].fillna(0)

        # 保存处理后的数据
        output_path = os.path.join(os.path.dirname(file_path), 'processed_registration_data.csv')
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"历史报名人数数据处理完成，已保存至: {output_path}")

        return df
    except Exception as e:
        print(f"处理历史报名人数数据时出错: {str(e)}")
        return pd.DataFrame()

if __name__ == '__main__':
    # 设置数据路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    attachment_dir = os.path.join(base_dir, '附件')

    # 检查附件目录是否存在
    if not os.path.exists(attachment_dir):
        print(f"警告: 附件目录 {attachment_dir} 不存在，请确保数据文件已放置在正确位置")
        attachment_dir = os.path.join(os.path.dirname(base_dir), '附件')  # 尝试上一级目录
        if not os.path.exists(attachment_dir):
            print(f"错误: 附件目录 {attachment_dir} 也不存在，请手动设置正确的数据路径")
            exit(1)

    # 处理附件1：气象数据
    weather_data_path = os.path.join(attachment_dir, '附件1')
    weather_data = process_weather_data(weather_data_path)

    # 处理附件2：轨道交通客运量数据
    transit_data_path = os.path.join(attachment_dir, '附件2.csv')  # 假设为CSV格式
    transit_data = process_transit_data(transit_data_path)

    # 处理附件3：人口普查数据
    census_data_path = os.path.join(attachment_dir, '附件3.csv')  # 假设为CSV格式
    census_data = process_census_data(census_data_path)

    # 处理附件12：历史报名人数数据
    registration_data_path = os.path.join(attachment_dir, '附件12.csv')  # 假设为CSV格式
    registration_data = process_registration_data(registration_data_path)

    print('数据预处理全部完成。')

    # 数据整合分析（可选）
    print('\n开始数据整合分析...')

    # 1. 检查各数据集是否成功处理
    datasets = {
        '气象数据': weather_data,
        '轨道交通数据': transit_data,
        '人口普查数据': census_data,
        '历史报名人数数据': registration_data
    }

    for name, data in datasets.items():
        if not data.empty:
            print(f"{name}处理成功，共 {len(data)} 行数据")
            print(f"数据列: {', '.join(data.columns)}")
        else:
            print(f"{name}处理失败或数据为空")

    print('\n数据预处理与分析完成。')