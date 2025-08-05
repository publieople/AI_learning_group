import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
from matplotlib.patches import Circle

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 常量定义
TOWER_HEIGHT = 80.0  # 吸收塔高度 (m)
RECEIVER_HEIGHT = 8.0  # 集热器高度 (m)
RECEIVER_DIAMETER = 7.0  # 集热器直径 (m)
FORBIDDEN_RADIUS = 100.0  # 禁止安装区域半径 (m)
FIELD_RADIUS = 350.0  # 场地半径 (m)

# 读取数据函数
def read_data():
    """
    读取定日镜位置数据和计算结果数据
    
    Returns:
        tuple: (mirrors_df, monthly_results, annual_results)
    """
    # 读取定日镜位置数据
    mirrors_df = pd.read_excel('../附件.xlsx')
    mirrors_df.rename(columns={'x坐标 (m)': 'x', 'y坐标 (m)': 'y'}, inplace=True)
    
    # 读取计算结果
    monthly_results = pd.read_excel('problem1_results.xlsx', sheet_name='表1')
    annual_results = pd.read_excel('problem1_results.xlsx', sheet_name='表2')
    
    return mirrors_df, monthly_results, annual_results

# 可视化函数
def visualize_heliostat_field(mirrors_df):
    """
    可视化定日镜场分布
    
    Args:
        mirrors_df: 包含定日镜位置的DataFrame
    """
    plt.figure(figsize=(10, 10))
    
    # 计算到原点的距离
    mirrors_df['distance'] = np.sqrt(mirrors_df['x']**2 + mirrors_df['y']**2)
    
    # 使用距离作为颜色映射
    norm = Normalize(vmin=mirrors_df['distance'].min(), vmax=mirrors_df['distance'].max())
    colors = cm.viridis(norm(mirrors_df['distance']))
    
    # 绘制定日镜位置散点图
    plt.scatter(mirrors_df['x'], mirrors_df['y'], c=colors, s=10, alpha=0.7)
    
    # 绘制吸收塔位置
    plt.scatter(0, 0, c='red', s=100, marker='*', label='吸收塔')
    
    # 绘制禁止区域
    forbidden_circle = Circle((0, 0), FORBIDDEN_RADIUS, fill=False, color='red', linestyle='--', label='禁止区域 (r=100m)')
    plt.gca().add_patch(forbidden_circle)
    
    # 绘制场地边界
    field_circle = Circle((0, 0), FIELD_RADIUS, fill=False, color='black', label='场地边界 (r=350m)')
    plt.gca().add_patch(field_circle)
    
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label('到吸收塔的距离 (m)')
    
    # 设置图表属性
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.title('定日镜场分布图')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('heliostat_field_distribution.png', dpi=300)
    plt.close()

def visualize_monthly_efficiency(monthly_results):
    """
    可视化月度效率变化
    
    Args:
        monthly_results: 包含月度效率的DataFrame
    """
    plt.figure(figsize=(12, 8))
    
    # 提取月份
    months = [int(month.split('月')[0]) for month in monthly_results['日期']]
    
    # 绘制各项效率的折线图
    plt.plot(months, monthly_results['平均光学效率'], 'o-', label='平均光学效率', linewidth=2)
    plt.plot(months, monthly_results['平均余弦效率'], 's-', label='平均余弦效率', linewidth=2)
    plt.plot(months, monthly_results['平均阴影遮挡效率'], '^-', label='平均阴影遮挡效率', linewidth=2)
    plt.plot(months, monthly_results['平均截断效率'], 'd-', label='平均截断效率', linewidth=2)
    
    # 设置图表属性
    plt.xlabel('月份')
    plt.ylabel('效率')
    plt.title('月度效率变化图')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower center')
    plt.xticks(months)
    
    # 添加数据标签
    for i, month in enumerate(months):
        plt.text(month, monthly_results['平均光学效率'][i] + 0.01, 
                 f"{monthly_results['平均光学效率'][i]:.3f}", 
                 ha='center')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('monthly_efficiency.png', dpi=300)
    plt.close()

def visualize_monthly_power(monthly_results):
    """
    可视化月度单位面积输出热功率
    
    Args:
        monthly_results: 包含月度输出功率的DataFrame
    """
    plt.figure(figsize=(12, 6))
    
    # 提取月份
    months = [int(month.split('月')[0]) for month in monthly_results['日期']]
    
    # 绘制单位面积输出热功率柱状图
    bars = plt.bar(months, monthly_results['单位面积镜面平均输出热功率 (kW/m²)'], 
             color=cm.viridis(np.linspace(0, 1, len(months))))
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f"{height:.3f}", ha='center', va='bottom')
    
    # 设置图表属性
    plt.xlabel('月份')
    plt.ylabel('单位面积镜面平均输出热功率 (kW/m²)')
    plt.title('月度单位面积输出热功率')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xticks(months)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('monthly_power.png', dpi=300)
    plt.close()

def visualize_annual_efficiency(annual_results):
    """
    可视化年平均效率
    
    Args:
        annual_results: 包含年平均效率的DataFrame
    """
    plt.figure(figsize=(10, 6))
    
    # 提取效率数据
    efficiency_labels = ['年平均光学效率', '年平均余弦效率', '年平均阴影遮挡效率', '年平均截断效率']
    efficiency_values = annual_results[efficiency_labels].values[0]
    
    # 绘制饼图
    plt.pie(efficiency_values, labels=efficiency_labels, autopct='%1.3f', 
            startangle=90, colors=cm.viridis(np.linspace(0, 1, len(efficiency_labels))))
    
    # 设置图表属性
    plt.axis('equal')
    plt.title('年平均效率对比')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('annual_efficiency_pie.png', dpi=300)
    plt.close()
    
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(efficiency_labels, efficiency_values, 
             color=cm.viridis(np.linspace(0, 1, len(efficiency_labels))))
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f"{height:.3f}", ha='center', va='bottom')
    
    # 设置图表属性
    plt.ylabel('效率值')
    plt.title('年平均效率对比')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xticks(rotation=15)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('annual_efficiency_bar.png', dpi=300)
    plt.close()

def visualize_annual_power(annual_results):
    """
    可视化年平均输出热功率
    
    Args:
        annual_results: 包含年平均输出功率的DataFrame
    """
    plt.figure(figsize=(8, 6))
    
    # 提取功率数据
    power_labels = ['年平均输出热功率 (MW)', '单位面积镜面年平均输出热功率 (kW/m²)']
    power_values = annual_results[power_labels].values[0]
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 第一个子图：年平均输出热功率
    ax1.bar(['年平均输出热功率'], power_values[0], color='orange')
    ax1.set_ylabel('功率 (MW)')
    ax1.set_title('年平均输出热功率')
    ax1.text(0, power_values[0] + 0.5, f"{power_values[0]:.3f} MW", ha='center')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 第二个子图：单位面积镜面年平均输出热功率
    ax2.bar(['单位面积镜面年平均输出热功率'], power_values[1], color='green')
    ax2.set_ylabel('功率 (kW/m²)')
    ax2.set_title('单位面积镜面年平均输出热功率')
    ax2.text(0, power_values[1] + 0.02, f"{power_values[1]:.3f} kW/m²", ha='center')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('annual_power.png', dpi=300)
    plt.close()

def visualize_distance_distribution(mirrors_df):
    """
    可视化定日镜到吸收塔的距离分布
    
    Args:
        mirrors_df: 包含定日镜位置的DataFrame
    """
    plt.figure(figsize=(10, 6))
    
    # 计算到原点的距离
    distances = np.sqrt(mirrors_df['x']**2 + mirrors_df['y']**2)
    
    # 绘制直方图
    plt.hist(distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 添加垂直线表示禁止区域和场地边界
    plt.axvline(x=FORBIDDEN_RADIUS, color='red', linestyle='--', label='禁止区域边界 (r=100m)')
    plt.axvline(x=FIELD_RADIUS, color='black', linestyle='-', label='场地边界 (r=350m)')
    
    # 设置图表属性
    plt.xlabel('到吸收塔的距离 (m)')
    plt.ylabel('定日镜数量')
    plt.title('定日镜到吸收塔的距离分布')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('distance_distribution.png', dpi=300)
    plt.close()

# 主函数
def main():
    # 读取数据
    mirrors_df, monthly_results, annual_results = read_data()
    
    print("正在生成可视化图表...")
    
    # 生成各种可视化图表
    visualize_heliostat_field(mirrors_df)
    visualize_monthly_efficiency(monthly_results)
    visualize_monthly_power(monthly_results)
    visualize_annual_efficiency(annual_results)
    visualize_annual_power(annual_results)
    visualize_distance_distribution(mirrors_df)
    
    print("可视化完成！生成的图表文件：")
    for file in os.listdir('.'):
        if file.endswith('.png'):
            print(f"- {file}")

if __name__ == "__main__":
    main()