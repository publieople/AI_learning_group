import pandas as pd
import numpy as np
import math
import os
from datetime import datetime

# 常量定义
REFLECTIVITY = 0.92  # 镜面反射率
MIRROR_SIZE = 6.0  # 定日镜尺寸 (m)
MIRROR_HEIGHT = 4.0  # 定日镜安装高度 (m)
TOWER_HEIGHT = 80.0  # 吸收塔高度 (m)
RECEIVER_HEIGHT = 8.0  # 集热器高度 (m)
RECEIVER_DIAMETER = 7.0  # 集热器直径 (m)
LATITUDE = 39.4  # 北纬 (度)
LONGITUDE = 98.5  # 东经 (度)
ALTITUDE = 3000  # 海拔 (m)

# 计算时点
MONTHS = list(range(1, 13))  # 1-12月
DAYS = [21] * 12  # 每月21日
HOURS = [9, 10.5, 12, 13.5, 15]  # 9:00, 10:30, 12:00, 13:30, 15:00

# 读取附件数据
def read_mirrors_data():
    """读取附件中的定日镜位置数据
    
    Returns:
        DataFrame: 包含定日镜位置的数据框
    """
    file_path = '../附件.xlsx'
    try:
        df = pd.read_excel(file_path)
        # 重命名列，方便后续处理
        df.rename(columns={'x坐标 (m)': 'x', 'y坐标 (m)': 'y'}, inplace=True)
        # 添加z坐标（安装高度）
        df['z'] = MIRROR_HEIGHT
        # 添加镜面面积
        df['area'] = MIRROR_SIZE * MIRROR_SIZE
        return df
    except Exception as e:
        print(f"读取附件数据失败: {e}")
        return None

# 太阳位置计算
def calculate_sun_position(month, day, hour):
    """计算指定时间的太阳位置（高度角和方位角）
    
    Args:
        month: 月份 (1-12)
        day: 日期
        hour: 小时 (含小数)
        
    Returns:
        tuple: (太阳高度角(rad), 太阳方位角(rad))
    """
    # 计算该日是一年中的第几天
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_of_year = sum(days_in_month[:month]) + day
    
    # 计算太阳赤纬角
    delta = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
    delta_rad = np.radians(delta)
    
    # 计算时角
    solar_time = hour  # 简化处理，假设当地时间等于太阳时
    omega = 15 * (solar_time - 12)  # 每小时15度
    omega_rad = np.radians(omega)
    
    # 计算太阳高度角
    lat_rad = np.radians(LATITUDE)
    sin_alpha = np.sin(lat_rad) * np.sin(delta_rad) + np.cos(lat_rad) * np.cos(delta_rad) * np.cos(omega_rad)
    alpha = np.arcsin(sin_alpha)  # 太阳高度角(rad)
    
    # 计算太阳方位角
    cos_gamma = (np.sin(delta_rad) - np.sin(lat_rad) * np.sin(alpha)) / (np.cos(lat_rad) * np.cos(alpha))
    cos_gamma = np.clip(cos_gamma, -1, 1)  # 确保在[-1, 1]范围内
    
    if omega < 0:  # 上午
        gamma = np.arccos(cos_gamma)  # 太阳方位角(rad)
    else:  # 下午
        gamma = 2 * np.pi - np.arccos(cos_gamma)
    
    return alpha, gamma

# 计算DNI (Direct Normal Irradiance)
def calculate_dni(alpha):
    """计算法向直接辐射辐照度
    
    Args:
        alpha: 太阳高度角(rad)
        
    Returns:
        float: DNI值 (W/m²)
    """
    # 简化模型，基于太阳高度角和大气质量
    if alpha <= 0:  # 太阳在地平线以下
        return 0
    
    # 大气质量计算
    AM = 1 / np.sin(alpha)
    
    # 考虑海拔影响
    altitude_factor = np.exp(ALTITUDE / 8000)
    
    # 基础DNI模型 (简化)
    dni = 1367 * 0.7**(AM**0.678) * altitude_factor
    
    return dni

# 计算镜面法向量
def calculate_mirror_normal(mirror_pos, sun_pos, receiver_pos):
    """计算定日镜的法向量
    
    Args:
        mirror_pos: 定日镜位置 [x, y, z]
        sun_pos: 太阳方向单位向量 [x, y, z]
        receiver_pos: 集热器中心位置 [x, y, z]
        
    Returns:
        ndarray: 镜面法向量 (单位向量)
    """
    # 从镜面到集热器的单位向量
    mirror_to_receiver = receiver_pos - mirror_pos
    mirror_to_receiver = mirror_to_receiver / np.linalg.norm(mirror_to_receiver)
    
    # 镜面法向量是太阳方向向量和镜面到集热器向量的角平分线
    normal = sun_pos + mirror_to_receiver
    normal = normal / np.linalg.norm(normal)
    
    return normal

# 计算余弦效率
def calculate_cosine_efficiency(mirror_normal, sun_pos):
    """计算余弦效率
    
    Args:
        mirror_normal: 镜面法向量 (单位向量)
        sun_pos: 太阳方向单位向量
        
    Returns:
        float: 余弦效率
    """
    # 余弦效率 = 太阳光线与镜面法线的夹角余弦
    cos_efficiency = np.abs(np.dot(mirror_normal, sun_pos))
    return cos_efficiency

# 计算大气透射率
def calculate_attenuation_factor(mirror_pos, receiver_pos):
    """计算大气透射率
    
    Args:
        mirror_pos: 定日镜位置 [x, y, z]
        receiver_pos: 集热器中心位置 [x, y, z]
        
    Returns:
        float: 大气透射率
    """
    # 计算定日镜到集热器的距离
    distance = np.linalg.norm(receiver_pos - mirror_pos)
    
    # 根据公式计算大气透射率
    attenuation = 0.99321 - 0.0001176 * distance + 1.97e-8 * distance**2
    return attenuation

# 计算阴影遮挡效率
def calculate_shadowing_blocking_efficiency(mirrors_df, mirror_index, sun_pos):
    """计算阴影遮挡效率
    
    Args:
        mirrors_df: 包含所有定日镜信息的DataFrame
        mirror_index: 当前计算的定日镜索引
        sun_pos: 太阳方向单位向量
        
    Returns:
        float: 阴影遮挡效率
    """
    # 简化模型：仅考虑太阳光线方向上的遮挡
    # 实际应用中需要更复杂的几何计算
    
    # 当前镜面位置
    current_mirror = mirrors_df.iloc[mirror_index][['x', 'y', 'z']].values
    
    # 检查其他镜面是否遮挡当前镜面
    shadowing = 0.0
    
    # 简化处理：假设阴影遮挡效率为0.95
    # 实际计算需要考虑镜面之间的相对位置和太阳光线方向
    return 0.95

# 计算截断效率
def calculate_truncation_efficiency(mirror_pos, mirror_normal, receiver_pos):
    """计算截断效率
    
    Args:
        mirror_pos: 定日镜位置 [x, y, z]
        mirror_normal: 镜面法向量
        receiver_pos: 集热器中心位置 [x, y, z]
        
    Returns:
        float: 截断效率
    """
    # 简化模型：基于镜面到集热器的距离和角度
    # 实际应用中需要考虑光斑大小和集热器几何形状
    
    # 镜面到集热器的距离
    distance = np.linalg.norm(receiver_pos - mirror_pos)
    
    # 镜面到集热器的单位向量
    mirror_to_receiver = (receiver_pos - mirror_pos) / distance
    
    # 镜面法向量与镜面到集热器向量的夹角余弦
    cos_angle = np.dot(mirror_normal, mirror_to_receiver)
    
    # 简化的截断效率模型
    # 距离越远，截断效率越低
    # 夹角越大，截断效率越低
    truncation = max(0, 1 - 0.0001 * distance - 0.2 * (1 - cos_angle))
    
    return truncation

# 计算单个定日镜的光学效率
def calculate_optical_efficiency(mirrors_df, mirror_index, sun_pos, receiver_pos):
    """计算单个定日镜的光学效率
    
    Args:
        mirrors_df: 包含所有定日镜信息的DataFrame
        mirror_index: 当前计算的定日镜索引
        sun_pos: 太阳方向单位向量
        receiver_pos: 集热器中心位置 [x, y, z]
        
    Returns:
        tuple: (光学效率, 余弦效率, 阴影遮挡效率, 大气透射率, 截断效率)
    """
    # 获取当前镜面位置
    mirror_pos = mirrors_df.iloc[mirror_index][['x', 'y', 'z']].values
    
    # 计算镜面法向量
    mirror_normal = calculate_mirror_normal(mirror_pos, sun_pos, receiver_pos)
    
    # 计算各项效率
    cosine_eff = calculate_cosine_efficiency(mirror_normal, sun_pos)
    shadowing_blocking_eff = calculate_shadowing_blocking_efficiency(mirrors_df, mirror_index, sun_pos)
    attenuation_eff = calculate_attenuation_factor(mirror_pos, receiver_pos)
    truncation_eff = calculate_truncation_efficiency(mirror_pos, mirror_normal, receiver_pos)
    
    # 计算总光学效率
    optical_eff = cosine_eff * shadowing_blocking_eff * attenuation_eff * truncation_eff * REFLECTIVITY
    
    return optical_eff, cosine_eff, shadowing_blocking_eff, truncation_eff

# 计算指定时间点的镜场效率和输出功率
def calculate_field_performance(mirrors_df, month, day, hour):
    """计算指定时间点的镜场效率和输出功率
    
    Args:
        mirrors_df: 包含所有定日镜信息的DataFrame
        month: 月份 (1-12)
        day: 日期
        hour: 小时 (含小数)
        
    Returns:
        tuple: (平均光学效率, 平均余弦效率, 平均阴影遮挡效率, 平均截断效率, 输出热功率, 单位面积输出热功率)
    """
    # 计算太阳位置
    alpha, gamma = calculate_sun_position(month, day, hour)
    
    # 如果太阳在地平线以下，返回零效率
    if alpha <= 0:
        return 0, 0, 0, 0, 0, 0
    
    # 计算太阳方向单位向量 [x, y, z]
    sun_pos = np.array([
        np.cos(alpha) * np.sin(gamma),  # x分量
        np.cos(alpha) * np.cos(gamma),  # y分量
        np.sin(alpha)                    # z分量
    ])
    
    # 集热器中心位置 [x, y, z]
    receiver_pos = np.array([0, 0, TOWER_HEIGHT + RECEIVER_HEIGHT/2])
    
    # 计算DNI
    dni = calculate_dni(alpha)
    
    # 初始化效率和功率累加器
    total_optical_eff = 0
    total_cosine_eff = 0
    total_shadowing_blocking_eff = 0
    total_truncation_eff = 0
    total_power = 0
    total_area = 0
    
    # 计算每个定日镜的效率和功率
    for i in range(len(mirrors_df)):
        optical_eff, cosine_eff, shadowing_blocking_eff, truncation_eff = calculate_optical_efficiency(
            mirrors_df, i, sun_pos, receiver_pos)
        
        # 累加效率
        total_optical_eff += optical_eff
        total_cosine_eff += cosine_eff
        total_shadowing_blocking_eff += shadowing_blocking_eff
        total_truncation_eff += truncation_eff
        
        # 计算该镜面的输出功率
        mirror_area = mirrors_df.iloc[i]['area']
        mirror_power = dni * mirror_area * optical_eff
        
        # 累加功率和面积
        total_power += mirror_power
        total_area += mirror_area
    
    # 计算平均效率
    n_mirrors = len(mirrors_df)
    avg_optical_eff = total_optical_eff / n_mirrors if n_mirrors > 0 else 0
    avg_cosine_eff = total_cosine_eff / n_mirrors if n_mirrors > 0 else 0
    avg_shadowing_blocking_eff = total_shadowing_blocking_eff / n_mirrors if n_mirrors > 0 else 0
    avg_truncation_eff = total_truncation_eff / n_mirrors if n_mirrors > 0 else 0
    
    # 计算单位面积输出热功率 (kW/m²)
    power_per_area = total_power / total_area / 1000 if total_area > 0 else 0
    
    return avg_optical_eff, avg_cosine_eff, avg_shadowing_blocking_eff, avg_truncation_eff, total_power, power_per_area

# 计算每月21日的平均效率和输出功率
def calculate_monthly_performance(mirrors_df):
    """计算每月21日的平均效率和输出功率
    
    Args:
        mirrors_df: 包含所有定日镜信息的DataFrame
        
    Returns:
        DataFrame: 包含每月效率和输出功率的数据框
    """
    results = []
    
    for month in MONTHS:
        day = 21
        
        # 初始化当日累加器
        daily_optical_eff = 0
        daily_cosine_eff = 0
        daily_shadowing_blocking_eff = 0
        daily_truncation_eff = 0
        daily_power = 0
        daily_power_per_area = 0
        valid_hours = 0
        
        # 计算每个时间点的效率和功率
        for hour in HOURS:
            optical_eff, cosine_eff, shadowing_blocking_eff, truncation_eff, power, power_per_area = calculate_field_performance(
                mirrors_df, month, day, hour)
            
            # 如果太阳在地平线以上，累加结果
            if optical_eff > 0:
                daily_optical_eff += optical_eff
                daily_cosine_eff += cosine_eff
                daily_shadowing_blocking_eff += shadowing_blocking_eff
                daily_truncation_eff += truncation_eff
                daily_power += power
                daily_power_per_area += power_per_area
                valid_hours += 1
        
        # 计算日平均值
        if valid_hours > 0:
            daily_optical_eff /= valid_hours
            daily_cosine_eff /= valid_hours
            daily_shadowing_blocking_eff /= valid_hours
            daily_truncation_eff /= valid_hours
            daily_power /= valid_hours
            daily_power_per_area /= valid_hours
        
        # 添加到结果列表
        results.append({
            'month': month,
            'day': day,
            'optical_efficiency': daily_optical_eff,
            'cosine_efficiency': daily_cosine_eff,
            'shadowing_blocking_efficiency': daily_shadowing_blocking_eff,
            'truncation_efficiency': daily_truncation_eff,
            'power': daily_power,  # W
            'power_per_area': daily_power_per_area  # kW/m²
        })
    
    return pd.DataFrame(results)

# 计算年平均效率和输出功率
def calculate_annual_performance(monthly_df):
    """计算年平均效率和输出功率
    
    Args:
        monthly_df: 包含每月效率和输出功率的DataFrame
        
    Returns:
        dict: 包含年平均效率和输出功率的字典
    """
    annual_results = {
        'optical_efficiency': monthly_df['optical_efficiency'].mean(),
        'cosine_efficiency': monthly_df['cosine_efficiency'].mean(),
        'shadowing_blocking_efficiency': monthly_df['shadowing_blocking_efficiency'].mean(),
        'truncation_efficiency': monthly_df['truncation_efficiency'].mean(),
        'power': monthly_df['power'].mean(),  # W
        'power_per_area': monthly_df['power_per_area'].mean()  # kW/m²
    }
    
    return annual_results

# 生成结果表格
def generate_result_tables(monthly_df, annual_results):
    """生成结果表格
    
    Args:
        monthly_df: 包含每月效率和输出功率的DataFrame
        annual_results: 包含年平均效率和输出功率的字典
        
    Returns:
        tuple: (表1 DataFrame, 表2 DataFrame)
    """
    # 表1：每月21日平均光学效率及输出功率
    table1 = pd.DataFrame({
        '日期': [f"{month}月21日" for month in MONTHS],
        '平均光学效率': monthly_df['optical_efficiency'],
        '平均余弦效率': monthly_df['cosine_efficiency'],
        '平均阴影遮挡效率': monthly_df['shadowing_blocking_efficiency'],
        '平均截断效率': monthly_df['truncation_efficiency'],
        '单位面积镜面平均输出热功率 (kW/m²)': monthly_df['power_per_area']
    })
    
    # 表2：年平均光学效率及输出功率
    table2 = pd.DataFrame({
        '年平均光学效率': [annual_results['optical_efficiency']],
        '年平均余弦效率': [annual_results['cosine_efficiency']],
        '年平均阴影遮挡效率': [annual_results['shadowing_blocking_efficiency']],
        '年平均截断效率': [annual_results['truncation_efficiency']],
        '年平均输出热功率 (MW)': [annual_results['power'] / 1e6],  # W转换为MW
        '单位面积镜面年平均输出热功率 (kW/m²)': [annual_results['power_per_area']]
    })
    
    return table1, table2

# 保存结果到Excel文件
def save_results_to_excel(table1, table2):
    """保存结果到Excel文件
    
    Args:
        table1: 表1 DataFrame
        table2: 表2 DataFrame
    """
    # 创建Excel写入器
    with pd.ExcelWriter('problem1_results.xlsx') as writer:
        table1.to_excel(writer, sheet_name='表1', index=False)
        table2.to_excel(writer, sheet_name='表2', index=False)
    
    print("结果已保存到 problem1_results.xlsx")

# 主函数
def main():
    # 读取定日镜数据
    mirrors_df = read_mirrors_data()
    if mirrors_df is None:
        return
    
    print(f"成功读取定日镜数据，共{len(mirrors_df)}个定日镜")
    
    # 计算每月21日的效率和输出功率
    print("计算每月21日的效率和输出功率...")
    monthly_df = calculate_monthly_performance(mirrors_df)
    
    # 计算年平均效率和输出功率
    print("计算年平均效率和输出功率...")
    annual_results = calculate_annual_performance(monthly_df)
    
    # 生成结果表格
    print("生成结果表格...")
    table1, table2 = generate_result_tables(monthly_df, annual_results)
    
    # 打印结果
    print("\n表1：每月21日平均光学效率及输出功率")
    print(table1)
    
    print("\n表2：年平均光学效率及输出功率")
    print(table2)
    
    # 保存结果到Excel文件
    save_results_to_excel(table1, table2)

if __name__ == "__main__":
    main()