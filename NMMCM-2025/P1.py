import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 确保输出目录存在
def ensure_dir_exists(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建目录: {dir_path}")

# 加载数据
def load_data():
    """加载清洗后的数据"""
    try:
        # 气象数据（附件1）
        weather_data = pd.read_csv('processed_data/cleaned/附件1_meteorological_data_cleaned.csv')
        # 轨道交通客运量（附件2）
        transit_data = pd.read_csv('processed_data/cleaned/附件2_subway_traffic_cleaned.csv')
        # 人口普查数据（附件3）- 使用城市级数据
        census_data = pd.read_csv('processed_data/cleaned/附件3_population_city_cleaned.csv')
        # 历史报名人数（附件12）
        registration_data = pd.read_csv('processed_data/cleaned/附件12_marathon_history_cleaned.csv')

        # 数据格式适配
        # 1. 气象数据
        if 'datetime' in weather_data.columns:
            weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])

        # 2. 轨道交通数据
        if '年份' in transit_data.columns:
            transit_data['年份'] = transit_data['年份'].astype(int)
        if '月份' in transit_data.columns:
            transit_data['月份'] = transit_data['月份'].astype(int)

        # 3. 人口普查数据
        if '普查年份' in census_data.columns:
            census_data['普查年份'] = census_data['普查年份'].astype(int)

        # 4. 马拉松历史数据
        if '比赛日期' in registration_data.columns:
            registration_data['比赛日期'] = pd.to_datetime(registration_data['比赛日期'])

        print("数据加载成功！")
        print(f"气象数据形状: {weather_data.shape}")
        print(f"轨道交通数据形状: {transit_data.shape}")
        print(f"人口普查数据形状: {census_data.shape}")
        print(f"马拉松历史数据形状: {registration_data.shape}")

        return weather_data, transit_data, census_data, registration_data

    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None, None, None, None

# 气象适宜性分析
def analyze_weather_suitability(weather_data):
    """分析气象数据，确定适宜举办马拉松的时间窗口"""
    # 添加城市列（从station_id映射，这里简化处理）
    weather_data['city'] = weather_data['station_id'].astype(str)

    # 添加年月信息
    weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
    weather_data['year'] = weather_data['datetime'].dt.year
    weather_data['month'] = weather_data['datetime'].dt.month

    # 添加湿度列（如果不存在）
    if 'humidity' not in weather_data.columns and 'dew_point' in weather_data.columns:
        # 利用温度和露点计算相对湿度
        weather_data['humidity'] = 100 * (np.exp((17.625 * weather_data['dew_point']) /
                                               (243.04 + weather_data['dew_point'])) /
                                       np.exp((17.625 * weather_data['temperature']) /
                                               (243.04 + weather_data['temperature'])))
    elif 'humidity' not in weather_data.columns:
        weather_data['humidity'] = 50  # 默认值

    # 确保数据中有precipitation列
    if 'precipitation' not in weather_data.columns and 'cloud_cover' in weather_data.columns:
        weather_data['precipitation'] = weather_data['cloud_cover']  # 使用cloud_cover作为替代
    elif 'precipitation' not in weather_data.columns:
        weather_data['precipitation'] = 0  # 默认值

    # 按城市和月份计算平均气象条件
    monthly_weather = weather_data.groupby(['city', 'month']).agg({
        'temperature': 'mean',
        'precipitation': 'mean',
        'wind_speed': 'mean',
        'humidity': 'mean',
        'pressure': 'mean'
    }).reset_index()

    # 定义气象舒适度指数计算函数
    def calculate_comfort_index(row):
        temp_score = 0
        if 5 <= row['temperature'] <= 28:  # 放宽温度范围
            # 最佳温度区间：10-22度，获得满分10分
            if 10 <= row['temperature'] <= 22:
                temp_score = 10
            # 5-10度和22-28度，得分9-5分，线性降低
            elif 5 <= row['temperature'] < 10:
                temp_score = 5 + (row['temperature'] - 5) * 0.8
            else:  # 22 < temp <= 28
                temp_score = 10 - (row['temperature'] - 22) * 0.833

        # 湿度评分：最佳40-60%，满分5分，放宽条件
        humidity_score = 0
        if 30 <= row['humidity'] <= 80:  # 放宽湿度范围
            if 40 <= row['humidity'] <= 65:
                humidity_score = 5
            elif 30 <= row['humidity'] < 40:
                humidity_score = 3 + (row['humidity'] - 30) * 0.2
            else:  # 65 < humidity <= 80
                humidity_score = 5 - (row['humidity'] - 65) * 0.333

        # 降水评分：无降水满分5分，降水越多分数越低，放宽条件
        precip_score = max(0, 5 - row['precipitation'] * 0.8)  # 降低降水的惩罚程度

        # 风速评分：最佳2-4m/s，满分5分，放宽条件
        wind_score = 0
        if row['wind_speed'] <= 6:  # 放宽风速范围
            if 2 <= row['wind_speed'] <= 4:
                wind_score = 5
            elif row['wind_speed'] < 2:
                wind_score = 3 + row['wind_speed']
            else:  # 4 < wind_speed <= 6
                wind_score = 5 - (row['wind_speed'] - 4) * 1.0

        # 总舒适度指数：温度占50%，湿度20%，降水20%，风速10%
        comfort_index = temp_score * 0.5 + humidity_score * 0.2 + precip_score * 0.2 + wind_score * 0.1
        return comfort_index

    # 计算每个城市每月的气象舒适度指数
    monthly_weather['comfort_index'] = monthly_weather.apply(calculate_comfort_index, axis=1)

    # 定义适宜条件，放宽判断标准
    suitable_conditions = (
        (monthly_weather['temperature'].between(5, 28)) &  # 放宽适宜气温范围
        (monthly_weather['precipitation'] < 10) &          # 放宽降水量标准
        (monthly_weather['wind_speed'] < 6) &              # 放宽风速标准
        (monthly_weather['humidity'].between(30, 80))      # 放宽适宜湿度范围
    )

    monthly_weather['is_suitable'] = suitable_conditions

    return monthly_weather

# 城市承载能力分析
def analyze_city_capacity(transit_data):
    """分析城市轨道交通运力"""
    if transit_data is None:
        print("轨道交通数据为空，无法进行分析")
        return None

    # 计算每个城市的月均客运量和峰值
    city_capacity = transit_data.groupby(['城市', '年份', '月份'])['客运量（万人次）'].mean().reset_index()
    city_capacity.rename(columns={'城市': 'city'}, inplace=True)

    # 分析交通负荷的月度变化
    monthly_capacity = city_capacity.groupby(['city', '月份'])['客运量（万人次）'].agg(['mean', 'std', 'max']).reset_index()
    monthly_capacity.rename(columns={'mean': 'avg_capacity', 'std': 'capacity_std', 'max': 'max_capacity'}, inplace=True)

    # 计算交通负荷评分：月均客运量越高，承载能力越强，但波动越大，稳定性越差
    monthly_capacity['capacity_score'] = monthly_capacity['avg_capacity'] / monthly_capacity['avg_capacity'].max() * 8
    monthly_capacity['stability_score'] = (1 - monthly_capacity['capacity_std'] / monthly_capacity['avg_capacity']) * 2
    monthly_capacity['stability_score'] = monthly_capacity['stability_score'].fillna(0).clip(0, 2)

    # 综合交通评分
    monthly_capacity['transport_score'] = monthly_capacity['capacity_score'] + monthly_capacity['stability_score']

    return monthly_capacity

# 人口规模分析
def analyze_population(population_data):
    """分析城市人口数据"""
    if population_data is None:
        print("人口数据为空，无法进行分析")
        return None

    # 计算总人口
    age_columns = [col for col in population_data.columns if any(x in col for x in ['岁_男', '岁_女'])]
    population_data['total_population'] = population_data[age_columns].sum(axis=1)

    # 计算适龄人口（18-45岁）比例
    running_age_columns = [col for col in population_data.columns if any(x in col for x in ['20-24岁', '25-29岁', '30-34岁', '35-39岁', '40-44岁', '15-19岁'])]
    population_data['running_age_population'] = population_data[running_age_columns].sum(axis=1)
    population_data['running_age_ratio'] = population_data['running_age_population'] / population_data['total_population']

    # 提取城市名称
    population_data['city'] = population_data['地名_Unnamed: 1_level_1']

    # 计算人口规模评分
    population_analysis = population_data[['city', 'total_population', 'running_age_population', 'running_age_ratio']].copy()

    # 人口规模评分：总人口越多越好，但有上限
    population_analysis.loc[:, 'population_size_score'] = population_analysis['total_population'] / 1000000  # 每百万人口1分
    population_analysis.loc[:, 'population_size_score'] = population_analysis['population_size_score'].clip(0, 7)  # 上限7分

    # 适龄人口评分：适龄人口比例越高越好
    population_analysis.loc[:, 'age_ratio_score'] = population_analysis['running_age_ratio'] * 3  # 最高3分

    # 总人口评分
    population_analysis.loc[:, 'total_score'] = population_analysis['population_size_score'] + population_analysis['age_ratio_score']

    return population_analysis

# 报名热度分析
def analyze_registration_trend(marathon_data):
    """分析马拉松报名趋势"""
    if marathon_data is None:
        print("马拉松历史数据为空，无法进行分析")
        return None

    # 打印列名以便调试
    print("马拉松历史数据列名:", marathon_data.columns.tolist())

    # 从raceTime列提取月份信息
    try:
        marathon_data['raceTime'] = pd.to_datetime(marathon_data['raceTime'], errors='coerce')
        marathon_data['month'] = marathon_data['raceTime'].dt.month
    except Exception as e:
        print(f"处理日期时出错: {e}")
        # 尝试从raceTime字符串中提取月份
        try:
            marathon_data['month'] = marathon_data['raceTime'].str.extract(r'(\d{4})-(\d{2})-\d{2}')[1].astype(int)
        except:
            print("无法从raceTime提取月份信息")
            return None

    # 从raceScale列提取报名人数
    try:
        # 尝试从raceScale中提取数字
        marathon_data['报名人数'] = marathon_data['raceScale'].str.extract(r'(\d+)').astype(float)
    except:
        print("无法从raceScale提取报名人数")
        return None

    # 计算每个城市的报名人数和完赛率
    city_stats = marathon_data.groupby('raceAddres').agg({
        '报名人数': 'mean',
        'month': 'count'  # 使用月份计数作为赛事数量
    }).reset_index()

    # 计算报名热度评分：报名人数越多，热度越高
    city_stats['报名热度'] = city_stats['报名人数'] / city_stats['报名人数'].max() * 10

    # 计算赛事频次评分：赛事数量越多，频次越高
    city_stats['频次评分'] = city_stats['month'] / city_stats['month'].max() * 10

    # 综合评分
    city_stats['综合评分'] = city_stats['报名热度'] + city_stats['频次评分']

    # 重命名列以保持一致性
    city_stats.rename(columns={
        'raceAddres': 'city',
        'month': '赛事数量'
    }, inplace=True)

    return city_stats

# 综合评分模型
def build_comprehensive_model(weather_data, city_capacity, population_data, registration_data):
    """构建综合评分模型，评估各城市各月份的马拉松适宜性"""
    # 初始化综合评分数据框
    cities = city_capacity['city'].unique()
    months = range(1, 13)

    comprehensive_scores = []

    for city in cities:
        city_weather = weather_data[weather_data['city'] == city]
        city_transport = city_capacity[city_capacity['city'] == city]

        # 查找人口数据（需要处理城市名称匹配问题）
        city_population = population_data[population_data['city'] == city]
        if len(city_population) == 0:
            # 尝试模糊匹配
            for pop_city in population_data['city'].unique():
                if city in pop_city or pop_city in city:
                    city_population = population_data[population_data['city'] == pop_city]
                    break

        # 如果仍然没找到匹配的人口数据，使用平均值
        if len(city_population) == 0:
            population_score = population_data['total_score'].mean()
        else:
            population_score = city_population['total_score'].iloc[0]

        # 查找报名热度数据
        city_registration = registration_data[registration_data['city'] == city]
        if len(city_registration) == 0:
            # 尝试模糊匹配
            for reg_city in registration_data['city'].unique():
                if city in reg_city or reg_city in city:
                    city_registration = registration_data[registration_data['city'] == reg_city]
                    break

        # 如果仍然没找到匹配的报名数据，使用平均值
        if len(city_registration) == 0:
            popularity_score = registration_data['综合评分'].mean()
        else:
            popularity_score = city_registration['综合评分'].iloc[0]

        for month in months:
            # 获取当月气象评分
            month_weather = city_weather[city_weather['month'] == month]
            if len(month_weather) > 0:
                weather_score = month_weather['comfort_index'].iloc[0]
                is_suitable = month_weather['is_suitable'].iloc[0]
            else:
                weather_score = 0
                is_suitable = False

            # 获取当月交通评分
            month_transport = city_transport[city_transport['月份'] == month]
            if len(month_transport) > 0:
                transport_score = month_transport['transport_score'].iloc[0]
            else:
                transport_score = 0

            # 计算综合评分：气象50%，交通30%，人口10%，报名热度10%
            total_score = (
                weather_score * 0.5 +
                transport_score * 0.3 +
                population_score * 0.1 +
                popularity_score * 0.1
            )

            comprehensive_scores.append({
                'city': city,
                'month': month,
                'weather_score': weather_score,
                'transport_score': transport_score,
                'population_score': population_score,
                'popularity_score': popularity_score,
                'total_score': total_score,
                'is_suitable': is_suitable
            })

    # 转换为DataFrame
    comprehensive_df = pd.DataFrame(comprehensive_scores)

    return comprehensive_df

# 优化赛事规模与频次
def optimize_event_parameters(comprehensive_scores, city_capacity, population_data):
    """为每个城市优化赛事规模与频次"""
    # 将城市按综合评分排序
    city_rankings = comprehensive_scores.groupby('city')['total_score'].max().sort_values(ascending=False)
    top_cities = city_rankings.index.tolist()

    event_recommendations = []

    for city in top_cities:
        # 获取该城市最适合的月份
        city_scores = comprehensive_scores[comprehensive_scores['city'] == city]
        best_months = city_scores.sort_values('total_score', ascending=False)

        # 筛选适宜举办的月份，如果没有适宜月份，则选择最佳的不适宜月份
        suitable_months = best_months[best_months['is_suitable']].copy()

        # 如果没有适宜月份，则使用总分最高的前3个月份
        if len(suitable_months) == 0:
            suitable_months = best_months.head(3).copy()
            if len(suitable_months) == 0:
                continue  # 如果仍然没有可用月份，则跳过该城市
            print(f"警告: 城市 {city} 没有完全适宜的月份，使用总分最高的月份")

        best_month = suitable_months['month'].iloc[0]
        second_best_month = suitable_months['month'].iloc[1] if len(suitable_months) > 1 else None

        # 计算赛事规模：基于城市交通承载能力和人口规模
        city_transport = city_capacity[city_capacity['city'] == city]

        # 默认规模
        event_size = 10000

        # 如果有交通数据，根据客运量调整规模
        if len(city_transport) > 0:
            max_capacity = city_transport['max_capacity'].max()
            if max_capacity > 0:
                # 假设每1万人次客运量可支持100名马拉松选手
                transport_based_size = max_capacity * 10
                event_size = max(event_size, min(transport_based_size, 40000))  # 上限4万人

        # 如果有人口数据，也考虑人口因素
        city_population = population_data[population_data['city'] == city]
        if len(city_population) > 0:
            population_size = city_population['total_population'].iloc[0]
            # 假设城市总人口的0.1%可参与马拉松
            population_based_size = population_size * 0.001
            event_size = min(event_size, population_based_size)

        # 确定赛事频次：基于适宜月份数量
        if len(suitable_months) >= 6:
            frequency = '每年2次'
            second_event = second_best_month
        elif len(suitable_months) >= 3:
            frequency = '每年1次'
            second_event = None
        else:
            frequency = '每两年1次'
            second_event = None

        event_recommendations.append({
            'city': city,
            'best_month': best_month,
            'second_best_month': second_event,
            'recommended_size': int(event_size),
            'frequency': frequency,
            'suitable_months': suitable_months['month'].tolist(),
            'total_score': suitable_months['total_score'].iloc[0]
        })

    # 如果still没有找到合适的城市，则放宽条件，选择top30的城市
    if len(event_recommendations) == 0:
        print("警告: 没有找到符合条件的城市，放宽条件重新筛选...")
        # 对所有城市按月份选择最高分的月份
        city_month_scores = comprehensive_scores.loc[comprehensive_scores.groupby('city')['total_score'].idxmax()]

        for _, row in city_month_scores.head(30).iterrows():
            city = row['city']
            month = row['month']

            # 设置默认规模
            event_size = 10000

            event_recommendations.append({
                'city': city,
                'best_month': month,
                'second_best_month': None,
                'recommended_size': int(event_size),
                'frequency': '每年1次',
                'suitable_months': [month],
                'total_score': row['total_score']
            })

    return pd.DataFrame(event_recommendations)

# 主函数
def main():
    # 加载数据
    weather_data, transit_data, census_data, registration_data = load_data()

    # 气象适宜性分析
    print("分析气象适宜性...")
    weather_suitability = analyze_weather_suitability(weather_data)

    # 城市承载能力分析
    print("分析城市承载能力...")
    city_capacity = analyze_city_capacity(transit_data)

    # 人口规模分析
    print("分析人口规模...")
    population_analysis = analyze_population(census_data)

    # 报名热度分析
    print("分析报名热度...")
    registration_trends = analyze_registration_trend(registration_data)

    # 构建综合评分模型
    print("构建综合评分模型...")
    comprehensive_scores = build_comprehensive_model(
        weather_suitability, city_capacity, population_analysis, registration_trends
    )

    # 优化赛事规模与频次
    print("优化赛事规模与频次...")
    event_recommendations = optimize_event_parameters(
        comprehensive_scores, city_capacity, population_analysis
    )

    # 检查结果是否为空
    if event_recommendations.empty:
        print("\n没有找到符合条件的城市。请检查筛选条件或数据。")
        return

    # 检查是否包含必要的列
    if 'best_month' not in event_recommendations.columns:
        print("\n结果数据中缺少'best_month'列。请检查optimize_event_parameters函数的实现。")
        # 打印数据框的列名以帮助调试
        print("可用的列名:", event_recommendations.columns.tolist())
        return

    # 输出结果
    print("\n推荐的马拉松赛事安排：")
    for _, rec in event_recommendations.head(10).iterrows():
        month_names = {1: '一月', 2: '二月', 3: '三月', 4: '四月', 5: '五月', 6: '六月',
                       7: '七月', 8: '八月', 9: '九月', 10: '十月', 11: '十一月', 12: '十二月'}

        print(f"\n城市：{rec['city']}")
        print(f"最适宜举办月份：{month_names[rec['best_month']]}")
        if pd.notna(rec['second_best_month']) and rec['second_best_month'] is not None:
            print(f"次适宜举办月份：{month_names[rec['second_best_month']]}")
        print(f"适宜月份列表：{[month_names[m] for m in rec['suitable_months']]}")
        print(f"建议参赛人数：{rec['recommended_size']:,}人")
        print(f"建议举办频次：{rec['frequency']}")
        print(f"综合评分：{rec['total_score']:.2f}")

    # 可视化部分添加异常处理
    try:
        # 创建保存图表的目录
        output_dir = './P1'
        ensure_dir_exists(output_dir)

        # 保存结果数据到CSV
        result_csv_path = f'{output_dir}/马拉松赛事推荐结果.csv'
        event_recommendations.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
        print(f"结果数据已保存至: {result_csv_path}")

        # 可视化：不同城市的最佳月份分布
        plt.figure(figsize=(12, 6))
        best_month_counts = event_recommendations['best_month'].value_counts().sort_index()
        if not best_month_counts.empty:
            best_month_counts.plot(kind='bar')
            plt.title('各城市最适宜举办马拉松的月份分布')
            plt.xlabel('月份')
            plt.ylabel('城市数量')
            plt.savefig(f'{output_dir}/马拉松最佳月份分布.png')
            print(f"已生成月份分布图：{output_dir}/马拉松最佳月份分布.png")
        else:
            print("没有足够数据生成月份分布图")

        # 可视化：城市评分前20名
        if len(event_recommendations) >= 5:  # 至少需要5个城市才生成排名图
            plt.figure(figsize=(12, 8))
            top_cities = event_recommendations.sort_values('total_score', ascending=False).head(20)
            top_cities.plot(x='city', y='total_score', kind='bar', figsize=(12, 6))
            plt.title('马拉松赛事综合评分前20城市')
            plt.xlabel('城市')
            plt.ylabel('综合评分')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/马拉松综合评分前20城市.png')
            print(f"已生成城市评分图：{output_dir}/马拉松综合评分前20城市.png")
        else:
            print("可用城市数量不足，无法生成城市评分图")
    except Exception as e:
        print(f"生成可视化图表时出错: {e}")

if __name__ == "__main__":
    main()