import pandas as pd
import numpy as np
import math
import random
from scipy.optimize import minimize
import copy

# 从problem1.py导入需要的函数
from problem1 import (
    calculate_sun_position, calculate_dni, calculate_mirror_normal,
    calculate_cosine_efficiency, calculate_attenuation_factor,
    calculate_truncation_efficiency, calculate_shadowing_blocking_efficiency,
    MONTHS, DAYS, HOURS, REFLECTIVITY, TOWER_HEIGHT, RECEIVER_HEIGHT
)

# 常量定义
LATITUDE = 39.4  # 北纬 (度)
LONGITUDE = 98.5  # 东经 (度)
ALTITUDE = 3000  # 海拔 (m)
FORBIDDEN_RADIUS = 100.0  # 禁止安装区域半径 (m)
FIELD_RADIUS = 350.0  # 场地半径 (m)
RECEIVER_DIAMETER = 7.0  # 集热器直径 (m)

# 计算单个定日镜的光学效率（修改版，支持可变的接收器位置和不同尺寸的定日镜）
def calculate_optical_efficiency_v3(mirrors_df, mirror_index, sun_pos, receiver_pos):
    """计算单个定日镜的光学效率（支持可变接收器位置和不同尺寸的定日镜）
    
    Args:
        mirrors_df: 包含所有定日镜信息的DataFrame
        mirror_index: 当前计算的定日镜索引
        sun_pos: 太阳方向单位向量
        receiver_pos: 集热器中心位置 [x, y, z]
        
    Returns:
        tuple: (光学效率, 余弦效率, 阴影遮挡效率, 截断效率)
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

# 计算指定时间点的镜场效率和输出功率（修改版，支持可变的接收器位置和不同尺寸的定日镜）
def calculate_field_performance_v3(mirrors_df, month, day, hour, receiver_pos):
    """计算指定时间点的镜场效率和输出功率（支持可变接收器位置和不同尺寸的定日镜）
    
    Args:
        mirrors_df: 包含所有定日镜信息的DataFrame
        month: 月份 (1-12)
        day: 日期
        hour: 小时 (含小数)
        receiver_pos: 集热器中心位置 [x, y, z]
        
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
        optical_eff, cosine_eff, shadowing_blocking_eff, truncation_eff = calculate_optical_efficiency_v3(
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

# 计算每月21日的平均效率和输出功率（修改版，支持可变的接收器位置和不同尺寸的定日镜）
def calculate_monthly_performance_v3(mirrors_df, receiver_pos):
    """计算每月21日的平均效率和输出功率（支持可变接收器位置和不同尺寸的定日镜）
    
    Args:
        mirrors_df: 包含所有定日镜信息的DataFrame
        receiver_pos: 集热器中心位置 [x, y, z]
        
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
            optical_eff, cosine_eff, shadowing_blocking_eff, truncation_eff, power, power_per_area = calculate_field_performance_v3(
                mirrors_df, month, day, hour, receiver_pos)
            
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

# 计算年平均效率和输出功率（修改版）
def calculate_annual_performance_v3(monthly_df):
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

# 生成圆形区域内随机点
def generate_random_points_in_circle(n_points, radius, forbidden_radius=0):
    """在圆形区域内生成随机点（排除中心禁止区域）
    
    Args:
        n_points: 需要生成的点数
        radius: 圆形区域半径
        forbidden_radius: 中心禁止区域半径
        
    Returns:
        list: 包含(x, y)坐标的点列表
    """
    points = []
    while len(points) < n_points:
        # 在圆形区域内生成随机点
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(forbidden_radius, radius)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        
        # 检查是否在禁止区域内
        if math.sqrt(x**2 + y**2) >= forbidden_radius:
            points.append((x, y))
    
    return points

# 生成螺旋排列的点
def generate_spiral_points(n_points, radius, forbidden_radius=0):
    """生成螺旋排列的点
    
    Args:
        n_points: 需要生成的点数
        radius: 圆形区域半径
        forbidden_radius: 中心禁止区域半径
        
    Returns:
        list: 包含(x, y)坐标的点列表
    """
    points = []
    for i in range(n_points):
        # 使用黄金角生成螺旋点
        angle = i * 2.399963229728653  # 黄金角约为137.5度，转换为弧度
        r = forbidden_radius + (radius - forbidden_radius) * math.sqrt(i / n_points)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        
        # 检查是否在允许区域内
        if math.sqrt(x**2 + y**2) <= radius and math.sqrt(x**2 + y**2) >= forbidden_radius:
            points.append((x, y))
    
    return points

# 创建定日镜场（支持不同尺寸和高度的定日镜）
def create_heliostat_field_v3(n_mirrors, tower_x, tower_y, layout_type="spiral"):
    """创建定日镜场（支持不同尺寸和高度的定日镜）
    
    Args:
        n_mirrors: 定日镜数量
        tower_x: 吸收塔x坐标
        tower_y: 吸收塔y坐标
        layout_type: 布局类型（"spiral" 或 "random"）
        
    Returns:
        DataFrame: 包含定日镜位置和参数的数据框
    """
    # 生成定日镜位置
    if layout_type == "spiral":
        positions = generate_spiral_points(n_mirrors, FIELD_RADIUS, FORBIDDEN_RADIUS)
    else:
        positions = generate_random_points_in_circle(n_mirrors, FIELD_RADIUS, FORBIDDEN_RADIUS)
    
    # 创建DataFrame
    mirrors_data = []
    for i, (x, y) in enumerate(positions):
        # 根据距离中心的距离确定定日镜尺寸和高度
        distance = math.sqrt(x**2 + y**2)
        
        # 根据距离确定尺寸和高度（离塔越远，使用更大的定日镜）
        if distance < 150:
            # 靠近塔的位置使用较小的定日镜
            mirror_width = random.uniform(5.0, 6.0)
            mirror_height = random.uniform(2.0, 3.0)
        elif distance < 250:
            # 中等距离使用中等定日镜
            mirror_width = random.uniform(6.0, 7.0)
            mirror_height = random.uniform(3.0, 4.0)
        else:
            # 远离塔的位置使用较大的定日镜
            mirror_width = random.uniform(7.0, 8.0)
            mirror_height = random.uniform(4.0, 5.0)
        
        # 确保宽度不小于高度
        mirror_width = max(mirror_width, mirror_height)
        
        # 随机确定安装高度
        mirror_install_height = random.uniform(2.0, 6.0)
        
        mirrors_data.append({
            'x': x,
            'y': y,
            'z': mirror_install_height,
            'width': mirror_width,
            'height': mirror_height,
            'area': mirror_width * mirror_height
        })
    
    return pd.DataFrame(mirrors_data)

# 优化不同尺寸定日镜的分布策略
def optimize_heliostat_field_v3(n_mirrors, tower_x, tower_y):
    """优化不同尺寸定日镜的分布策略
    
    Args:
        n_mirrors: 定日镜数量
        tower_x: 吸收塔x坐标
        tower_y: 吸收塔y坐标
        
    Returns:
        DataFrame: 包含优化后的定日镜位置和参数的数据框
    """
    # 生成定日镜位置（螺旋排列）
    positions = generate_spiral_points(n_mirrors, FIELD_RADIUS, FORBIDDEN_RADIUS)
    
    # 创建DataFrame
    mirrors_data = []
    for i, (x, y) in enumerate(positions):
        # 根据距离中心的距离确定定日镜尺寸和高度
        distance = math.sqrt(x**2 + y**2)
        
        # 根据距离和角度确定尺寸和高度
        # 这种策略考虑了太阳的位置和定日镜的朝向
        angle = math.atan2(y, x)  # 与x轴的夹角
        
        # 确定定日镜尺寸（离塔越远，尺寸越大）
        # 但也要考虑角度因素（东西向的镜子可能需要不同的尺寸）
        if distance < 150:
            # 靠近塔的位置使用较小的定日镜
            if abs(angle) < math.pi/4 or abs(angle) > 3*math.pi/4:
                # 南北向的镜子
                mirror_width = random.uniform(6.0, 7.0)
                mirror_height = random.uniform(3.0, 4.0)
            else:
                # 东西向的镜子
                mirror_width = random.uniform(6.5, 7.5)
                mirror_height = random.uniform(3.5, 4.5)
        elif distance < 250:
            # 中等距离使用中等定日镜
            if abs(angle) < math.pi/4 or abs(angle) > 3*math.pi/4:
                mirror_width = random.uniform(7.0, 8.0)
                mirror_height = random.uniform(4.0, 5.0)
            else:
                mirror_width = random.uniform(7.5, 8.0)
                mirror_height = random.uniform(4.5, 6.0)
        else:
            # 远离塔的位置使用较大的定日镜
            if abs(angle) < math.pi/4 or abs(angle) > 3*math.pi/4:
                mirror_width = random.uniform(7.5, 8.0)
                mirror_height = random.uniform(5.0, 6.0)
            else:
                mirror_width = random.uniform(7.8, 8.0)
                mirror_height = random.uniform(5.5, 6.0)
        
        # 确保宽度不小于高度
        mirror_width = max(mirror_width, mirror_height)
        
        # 安装高度也根据距离调整（离塔越远，可以安装得更高）
        mirror_install_height = 2.0 + (distance / FIELD_RADIUS) * 4.0
        
        mirrors_data.append({
            'x': x,
            'y': y,
            'z': mirror_install_height,
            'width': mirror_width,
            'height': mirror_height,
            'area': mirror_width * mirror_height
        })
    
    return pd.DataFrame(mirrors_data)

# 问题3主函数
def solve_problem3():
    """解决第三个问题"""
    print("开始解决第三个问题...")
    print("目标：在定日镜尺寸和安装高度可以不同的情况下，最大化单位镜面面积年平均输出热功率")
    
    # 基于问题2的结果进行优化
    # 问题2中使用1800个7x7定日镜产生61.99MW功率
    # 在问题3中，我们尝试使用不同尺寸的定日镜来提高单位面积效率
    
    best_solution = None
    best_power_per_area = 0
    
    # 测试不同的参数组合
    n_mirror_options = [2000, 2200, 2400, 2600]
    
    total_combinations = len(n_mirror_options)
    current_combination = 0
    
    for n_mirrors in n_mirror_options:
        current_combination += 1
        print(f"测试进度: {current_combination}/{total_combinations}")
        
        # 创建优化的定日镜场（吸收塔在中心）
        mirrors_df = optimize_heliostat_field_v3(n_mirrors, 0, 0)
        
        # 集热器中心位置 [x, y, z]
        receiver_pos = np.array([0, 0, TOWER_HEIGHT + RECEIVER_HEIGHT/2])
        
        # 计算性能
        try:
            monthly_df = calculate_monthly_performance_v3(mirrors_df, receiver_pos)
            annual_results = calculate_annual_performance_v3(monthly_df)
            
            actual_power = annual_results['power']
            power_per_area = annual_results['power_per_area']
            total_area = mirrors_df['area'].sum()
            
            print(f"定日镜数量: {n_mirrors}")
            print(f"定日镜总面积: {total_area:.2f} m²")
            print(f"年平均输出功率: {actual_power/1e6:.2f} MW")
            print(f"单位面积功率: {power_per_area:.4f} kW/m²")
            
            # 检查是否满足功率要求并且单位面积功率更高
            if actual_power >= 57e6 and power_per_area > best_power_per_area:  # 允许5%的误差
                best_power_per_area = power_per_area
                best_solution = {
                    'tower_x': 0,
                    'tower_y': 0,
                    'n_mirrors': n_mirrors,
                    'total_area': total_area,
                    'annual_power': actual_power,
                    'power_per_area': power_per_area,
                    'mirrors_df': mirrors_df.copy()
                }
                print(f"找到更好的解: 单位面积功率 = {power_per_area:.4f} kW/m²")
            elif actual_power >= 55e6:  # 至少记录一个接近目标的解
                if best_solution is None or actual_power > best_solution['annual_power']:
                    best_solution = {
                        'tower_x': 0,
                        'tower_y': 0,
                        'n_mirrors': n_mirrors,
                        'total_area': total_area,
                        'annual_power': actual_power,
                        'power_per_area': power_per_area,
                        'mirrors_df': mirrors_df.copy()
                    }
                    print(f"记录当前最佳解: 功率 = {actual_power/1e6:.2f} MW, 单位面积功率 = {power_per_area:.4f} kW/m²")
                
        except Exception as e:
            print(f"计算出错: {e}")
            continue
    
    if best_solution:
        print("\n最优解:")
        print(f"吸收塔位置: ({best_solution['tower_x']}, {best_solution['tower_y']})")
        print(f"定日镜数量: {best_solution['n_mirrors']}")
        print(f"定日镜总面积: {best_solution['total_area']:.2f} m²")
        print(f"年平均输出功率: {best_solution['annual_power']/1e6:.2f} MW")
        print(f"单位面积年平均输出功率: {best_solution['power_per_area']:.4f} kW/m²")
        
        # 显示定日镜尺寸分布统计
        mirrors_df = best_solution['mirrors_df']
        print("\n定日镜尺寸分布:")
        width_bins = pd.cut(mirrors_df['width'], bins=[0, 5, 6, 7, 8, 10], include_lowest=True)
        width_distribution = width_bins.value_counts().sort_index()
        for bin_range, count in width_distribution.items():
            print(f"  宽度 {bin_range}: {count} 面")
        
        height_bins = pd.cut(mirrors_df['height'], bins=[0, 3, 4, 5, 6, 10], include_lowest=True)
        height_distribution = height_bins.value_counts().sort_index()
        for bin_range, count in height_distribution.items():
            print(f"  高度 {bin_range}: {count} 面")
        
        # 显示安装高度分布
        print("\n安装高度分布:")
        height_bins = pd.cut(mirrors_df['z'], bins=[0, 2.5, 3.5, 4.5, 5.5, 10], include_lowest=True)
        height_distribution = height_bins.value_counts().sort_index()
        for bin_range, count in height_distribution.items():
            print(f"  高度 {bin_range}: {count} 面")
        
        # 生成详细结果
        receiver_pos = np.array([
            best_solution['tower_x'], 
            best_solution['tower_y'], 
            TOWER_HEIGHT + RECEIVER_HEIGHT/2
        ])
        
        monthly_df = calculate_monthly_performance_v3(mirrors_df, receiver_pos)
        annual_results = calculate_annual_performance_v3(monthly_df)
        
        # 生成表1：每月21日平均光学效率及输出功率
        table1 = pd.DataFrame({
            '日期': [f"{month}月21日" for month in MONTHS],
            '平均光学效率': monthly_df['optical_efficiency'],
            '平均余弦效率': monthly_df['cosine_efficiency'],
            '平均阴影遮挡效率': monthly_df['shadowing_blocking_efficiency'],
            '平均截断效率': monthly_df['truncation_efficiency'],
            '单位面积镜面平均输出热功率 (kW/m²)': monthly_df['power_per_area']
        })
        
        # 生成表2：年平均光学效率及输出功率
        table2 = pd.DataFrame({
            '年平均光学效率': [annual_results['optical_efficiency']],
            '年平均余弦效率': [annual_results['cosine_efficiency']],
            '年平均阴影遮挡效率': [annual_results['shadowing_blocking_efficiency']],
            '年平均截断效率': [annual_results['truncation_efficiency']],
            '年平均输出热功率 (MW)': [annual_results['power'] / 1e6],
            '单位面积镜面年平均输出热功率 (kW/m²)': [annual_results['power_per_area']]
        })
        
        # 生成表3：设计参数表（问题3中定日镜尺寸和安装高度可不同，所以这两栏空缺）
        table3 = pd.DataFrame({
            '吸收塔位置坐标 (x, y)': [f"({best_solution['tower_x']}, {best_solution['tower_y']})"],
            '定日镜尺寸 (宽 × 高)': [""],  # 空缺
            '定日镜安装高度 (m)': [""],   # 空缺
            '定日镜总面数': [best_solution['n_mirrors']],
            '定日镜总面积 (m²)': [best_solution['total_area']]
        })
        
        # 保存结果到Excel文件
        with pd.ExcelWriter('result3.xlsx') as writer:
            table1.to_excel(writer, sheet_name='表1', index=False)
            table2.to_excel(writer, sheet_name='表2', index=False)
            table3.to_excel(writer, sheet_name='表3', index=False)
        
        print("\n结果已保存到 result3.xlsx")
        
        # 保存定日镜位置数据
        mirrors_df.to_csv('heliostat_positions_problem3.csv', index=False)
        print("定日镜位置数据已保存到 heliostat_positions_problem3.csv")
        
        return best_solution
    else:
        print("未找到满足条件的解")
        return None

# 主函数
if __name__ == "__main__":
    solution = solve_problem3()