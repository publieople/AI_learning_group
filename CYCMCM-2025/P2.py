import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据准备与预处理
def load_and_preprocess_data():
    data = {
        '年份': [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014],
        '医疗卫生机构数(个)': [6514, 6404, 6308, 5897, 5597, 5293, 5144, 5016, 5016, 4984],
        '医院床位数(万张)': [17.5, 16.53, 16.04, 15.22, 14.65, 13.9, 13.46, 12.92, 12.28, 11.75],
        '公园个数(个)': [552, 473, 434, 386, 319, 250, 224, 217, 165, 161],
        '建成区绿化覆盖率(%)': [37.8, 38.1, 37.7, 37.3, 36.8, 36.2, 39.1, 38.6, 38.5, 38.4],
        '公园绿地面积(万公顷)': [2.35, 2.3, 2.25, 2.2, 2.12, 2.06, 1.98, 1.9, 1.84, 1.78],
        '文化机构(个)': [85, 116, 116, 107, 98, 100, 98, 99, 99, 103],
        '老龄人口占比(%)': [14.316, 13.782, 13.206, 12.645, 12.061, 11.516, 10.981, 10.491, 10.053, 9.666],
        '养老机构数量(个)': [700, 729, 730, 709, 724, 728, 703, 700, 699, 660],
        '养老床位(万张)': [517.2, 518.3, 503.6, 488.2, 438.8, 379.4, 383.5, 378.8, 358.2, 390.2],
        '常住人口(万人)': [2487.45, 2475.89, 2489.43, 2488.36, 2481.34, 2475.39, 2466.28, 2467.37, 2457.59, 2467.06],
        '人均寿命(岁)': [84.18, 83.66, 83.63, 82.55, 82.29, 81.75, 81.25, 80.98, 80.47, 80.26],
        '地区生产总值(亿元)': [47218.7, 44809.1, 43653.2, 38963.3, 37987.6, 36011.8, 32925, 29887, 26887, 25269.8],
        '人均生产总值(万元)': [19.03, 18.05, 17.54, 15.68, 15.33, 14.57, 13.35, 12.14, 10.92, 10.28],
        '用水总量(亿立方米)': [104.8, 105.7, 105.8, 97.5, 100.9, 103.4, 104.8, 104.8, 103.8, 105.9],
        '空气质量指数': [0.877, 0.871, 0.871, 0.872, 0.847, 0.811, 0.758, 0.754, 0.707, 0.77],
        '新生儿死亡率(%)': [2.14, 2.21, 2.3, 2.66, 3.98, 3.52, 3.71, 4.06, 4.58, 4.83],
        '进出口总额(千美元)': [574150000, 599680000, 604394651, 478776906, 473699895, 485863759, 447349338, 404613805,
                               423037090, 452602727]
    }

    df = pd.DataFrame(data)

    # 转换进出口总额单位（千美元转亿美元）
    df['进出口总额(亿美元)'] = df['进出口总额(千美元)'] / 10000000
    df.drop('进出口总额(千美元)', axis=1, inplace=True)

    # 计算人均指标
    df['每万人医疗床位数'] = df['医院床位数(万张)'] * 10000 / df['常住人口(万人)']
    df['每万人养老床位数'] = df['养老床位(万张)'] * 10000 / df['常住人口(万人)']
    df['每万人医疗卫生机构数'] = df['医疗卫生机构数(个)'] / df['常住人口(万人)'] * 10000
    df['每万人公园数'] = df['公园个数(个)'] / df['常住人口(万人)'] * 10000
    df['每万人文化机构数'] = df['文化机构(个)'] / df['常住人口(万人)'] * 10000
    df['人均公园绿地面积'] = df['公园绿地面积(万公顷)'] * 10000 / df['常住人口(万人)']
    return df

# 2. 构建评价指标体系
def define_indicators():
    indicators = {
        '康养资源丰富度': [
            '每万人医疗床位数',
            '每万人养老床位数',
            '每万人医疗卫生机构数',
            '每万人公园数',
            '每万人文化机构数',
            '养老机构数量(个)'
        ],
        '居民健康状况': [
            '人均寿命(岁)',
            '老龄人口占比(%)',  # 逆向指标
            '新生儿死亡率(%)'  # 逆向指标
        ],
        '环境质量': [
            '建成区绿化覆盖率(%)',
            '人均公园绿地面积',
            '空气质量指数'
        ],
        '经济发展水平': [
            '人均生产总值(万元)',
            '地区生产总值(亿元)',
            '进出口总额(亿美元)'
        ]
    }
    return indicators

# 3. 数据标准化处理
def normalize_data(df, indicators):
    scaler = MinMaxScaler()
    normalized_df = df.copy()

    # 正向指标标准化
    for dimension in indicators:
        for indicator in indicators[dimension]:
            if indicator not in ['老龄人口占比(%)', '新生儿死亡率(%)']:  # 正向指标
                normalized_df[indicator] = scaler.fit_transform(df[[indicator]])

    # 逆向指标处理
    for indicator in ['老龄人口占比(%)', '新生儿死亡率(%)']:
        if indicator in df.columns:
            normalized_df[indicator] = 1 - scaler.fit_transform(df[[indicator]])

    return normalized_df

# 4. 确定指标权重
def calculate_weights(indicators):
    # 维度比较矩阵
    dimension_matrix = np.array([
        [1, 3, 2, 2],
        [1 / 3, 1, 1 / 2, 1 / 2],
        [1 / 2, 2, 1, 1],
        [1 / 2, 2, 1, 1]
    ])

    # 计算特征向量
    eigenvalues, eigenvectors = np.linalg.eig(dimension_matrix)
    max_index = np.argmax(eigenvalues)
    weights = np.real(eigenvectors[:, max_index])
    dimension_weights = weights / np.sum(weights)

    # 定义各维度内指标的权重（自定义部分）
    indicator_weights = {
        '康养资源丰富度': {
            '每万人医疗床位数': 0.15,
            '每万人养老床位数': 0.25,
            '每万人医疗卫生机构数': 0.3,
            '每万人公园数': 0.1,
            '每万人文化机构数': 0.1,
            '养老机构数量(个)': 0.1
        },
        '居民健康状况': {
            '人均寿命(岁)': 0.7,
            '老龄人口占比(%)': 0.15,
            '新生儿死亡率(%)': 0.15
        },
        '环境质量': {
            '建成区绿化覆盖率(%)': 0.5,
            '人均公园绿地面积': 0.3,
            '空气质量指数': 0.2
        },
        '经济发展水平': {
            '人均生产总值(万元)': 0.5,
            '地区生产总值(亿元)': 0.3,
            '进出口总额(亿美元)': 0.2
        }
    }

    # 检查并归一化指标权重
    for dimension in indicator_weights:
        total_weight = sum(indicator_weights[dimension].values())
        for indicator in indicator_weights[dimension]:
            indicator_weights[dimension][indicator] = indicator_weights[dimension][indicator] / total_weight * \
                                                      dimension_weights[list(indicators.keys()).index(dimension)]

    return dimension_weights, indicator_weights

def calculate_scores(normalized_df, indicators, dimension_weights, indicator_weights):
    scores = pd.DataFrame()
    scores['年份'] = normalized_df['年份']

    # 计算各维度得分
    for dimension in indicators:
        dim_score = np.zeros(len(normalized_df))
        for indicator in indicators[dimension]:
            dim_score += normalized_df[indicator] * indicator_weights[dimension][indicator]
        scores[dimension] = dim_score

    # 计算综合得分
    scores['综合得分'] = 0
    for i, dimension in enumerate(indicators):
        scores['综合得分'] += scores[dimension] * dimension_weights[i]

    return scores

# 5. 结果可视化
def visualize_results(scores, indicators, df, output_dir='./P2'):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))

    # 各维度得分趋势
    for dimension in indicators:
        plt.plot(scores['年份'], scores[dimension], label=dimension, marker='o')

    plt.plot(scores['年份'], scores['综合得分'], label='综合得分', linewidth=2, color='black', marker='s')
    plt.title('上海市康养城市发展综合评价(2014-2023)')
    plt.xlabel('年份')
    plt.ylabel('标准化得分')
    plt.legend()
    plt.grid(True)
    plt.xticks(scores['年份'])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/上海市康养城市发展综合评价(2014-2023).png', dpi=300, bbox_inches='tight')

    # 2023年各维度得分雷达图
    categories = list(indicators.keys())
    values = scores[scores['年份'] == 2023][categories].values[0]
    values = np.append(values, values[0])
    categories = np.append(categories, categories[0])
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(values))

    plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)
    plt.plot(label_loc, values, label='2023年')
    plt.fill(label_loc, values, alpha=0.2)
    plt.title('2023年上海市康养城市各维度评价', y=1.1)
    lines, labels = plt.thetagrids(np.degrees(label_loc[:-1]), labels=categories[:-1])
    plt.legend(loc='upper right')
    plt.savefig(f'{output_dir}/2023年上海市康养城市各维度评价.png', dpi=300, bbox_inches='tight')

    # 3. 新增：各指标随时间变化的趋势图
    plt.figure(figsize=(14, 10))
    indicators_all = [ind for dim in indicators.values() for ind in dim]
    for i, indicator in enumerate(indicators_all):
        plt.subplot(4, 4, i + 1)
        plt.plot(df['年份'], df[indicator], marker='o')
        plt.title(indicator)
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/各指标随时间变化趋势.png', dpi=300, bbox_inches='tight')

    # 4. 2023年各维度得分占比饼图
    plt.figure(figsize=(8, 8))
    latest_scores = scores[scores['年份'] == 2023][categories[:-1]].iloc[0]
    plt.pie(latest_scores, labels=latest_scores.index, autopct='%1.1f%%', startangle=90)
    plt.title('2023年各维度得分占比')
    plt.savefig(f'{output_dir}/2023年各维度得分占比.png', dpi=300, bbox_inches='tight')

    # 5. 各维度得分与综合得分的相关性热力图
    plt.figure(figsize=(8, 6))
    correlation = scores.drop('年份', axis=1).corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('各维度得分相关性热力图')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/各维度得分相关性热力图.png', dpi=300, bbox_inches='tight')

    # 6. 各指标与综合得分的散点图矩阵
    selected_indicators = ['每万人医疗床位数', '人均寿命(岁)', '建成区绿化覆盖率(%)',
                           '人均生产总值(万元)', '综合得分']
    sns.pairplot(scores.join(df[selected_indicators[:-1]]), vars=selected_indicators)
    plt.suptitle('各指标与综合得分关系矩阵', y=1.02)
    plt.savefig(f'{output_dir}/各指标与综合得分关系矩阵.png', dpi=300, bbox_inches='tight')

# 6. 模型评价
def evaluate_model(scores, normalized_df, indicators, dimension_weights, indicator_weights, output_dir='./P2'):
    """
    模型评价函数，包括一致性检验、敏感性分析和稳定性测试
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. AHP一致性检验
    def check_consistency(matrix):
        n = matrix.shape[0]
        eigenvalues = np.linalg.eigvals(matrix)
        max_eigenvalue = max(np.real(eigenvalues))
        CI = (max_eigenvalue - n) / (n - 1)
        RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
        CR = CI / RI[n]
        return CR

    dimension_matrix = np.array([
        [1, 3, 2, 2],
        [1 / 3, 1, 1 / 2, 1 / 2],
        [1 / 2, 2, 1, 1],
        [1 / 2, 2, 1, 1]
    ])
    CR = check_consistency(dimension_matrix)
    consistency_result = f"模型一致性检验结果:\n一致性比率CR = {CR:.4f}\n"
    consistency_result += "CR < 0.1，通过一致性检验" if CR < 0.1 else "CR ≥ 0.1，未通过一致性检验"

    # 2. 敏感性分析 - 权重变化对结果的影响
    sensitivity_result = "\n敏感性分析:\n"
    original_scores = scores[scores['年份'] == 2023]['综合得分'].values[0]

    # 测试康养资源丰富度权重变化±20%的影响
    for change in [-0.2, 0.2]:
        temp_weights = dimension_weights.copy()
        temp_weights[0] = temp_weights[0] * (1 + change)
        temp_weights = temp_weights / temp_weights.sum()  # 重新归一化
        temp_scores = calculate_scores(normalized_df, indicators, temp_weights, indicator_weights)
        changed_score = temp_scores[temp_scores['年份'] == 2023]['综合得分'].values[0]
        sensitivity_result += f"康养资源权重变化{change * 100:.0f}% → 综合得分变化: {changed_score - original_scores:.4f} "
        sensitivity_result += f"({(changed_score - original_scores) / original_scores * 100:.2f}%)\n"

    # 3. 稳定性测试 - 指标增减测试
    stability_result = "\n稳定性测试:\n"
    # 测试移除一个指标的影响
    test_indicators = indicators.copy()
    removed_indicator = '每万人医疗床位数'
    test_indicators['康养资源丰富度'].remove(removed_indicator)
    temp_scores = calculate_scores(normalized_df, test_indicators, dimension_weights, indicator_weights)
    changed_score = temp_scores[temp_scores['年份'] == 2023]['综合得分'].values[0]
    stability_result += f"移除指标'{removed_indicator}' → 综合得分变化: {changed_score - original_scores:.4f} "
    stability_result += f"({(changed_score - original_scores) / original_scores * 100:.2f}%)\n"

    # 4. 模型拟合度评估 - 与实际数据的相关性
    actual_data = normalized_df[['人均寿命(岁)']].values.flatten()
    model_scores = scores['综合得分'].values
    correlation = np.corrcoef(actual_data, model_scores)[0, 1]
    fit_result = f"\n模型拟合度评估:\n综合得分与实际寿命数据的相关系数: {correlation:.4f}"

    # 5. 可视化模型评价结果
    plt.figure(figsize=(12, 5))

    # 5.1 权重敏感性分析可视化
    plt.subplot(1, 2, 1)
    changes = np.linspace(-0.3, 0.3, 7)
    score_changes = []
    for change in changes:
        temp_weights = dimension_weights.copy()
        temp_weights[0] = temp_weights[0] * (1 + change)
        temp_weights = temp_weights / temp_weights.sum()
        temp_scores = calculate_scores(normalized_df, indicators, temp_weights, indicator_weights)
        changed_score = temp_scores[temp_scores['年份'] == 2023]['综合得分'].values[0]
        score_changes.append((changed_score - original_scores) / original_scores * 100)

    plt.plot(changes * 100, score_changes, marker='o')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('权重敏感性分析(康养资源维度)')
    plt.xlabel('权重变化百分比(%)')
    plt.ylabel('综合得分变化百分比(%)')
    plt.grid(True)

    # 5.2 模型与实际数据相关性可视化
    plt.subplot(1, 2, 2)
    plt.scatter(model_scores, actual_data)
    plt.plot(np.unique(model_scores), np.poly1d(np.polyfit(model_scores, actual_data, 1))(np.unique(model_scores)),
             color='red')
    plt.title(f'模型与实际数据相关性(r={correlation:.2f})')
    plt.xlabel('模型综合得分')
    plt.ylabel('标准化人均寿命')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/模型评价结果.png', dpi=300, bbox_inches='tight')

    return consistency_result + sensitivity_result + stability_result + fit_result

# 7. 生成分析报告
# 新增函数：生成分析报告
def generate_report(scores, df, indicators, dimension_weights, indicator_weights, output_dir='./P2'):
    os.makedirs(output_dir, exist_ok=True)
    latest_year = scores['年份'].max()
    latest_score = scores[scores['年份'] == latest_year].iloc[0]

    # 找出优势与不足维度
    max_dim = max(indicators.keys(), key=lambda x: latest_score[x])
    min_dim = min(indicators.keys(), key=lambda x: latest_score[x])

    # 计算年均增长率
    growth_rates = {}
    for dimension in indicators:
        growth = (latest_score[dimension] - scores[scores['年份'] == 2014][dimension].values[0]) / 10
        growth_rates[dimension] = growth

    # 生成文本报告
    report = """
# 上海市康养城市发展综合评价报告 ({0})
=================================================

## 一、综合得分

综合得分: {1:.3f}

## 二、各维度得分


{2}

## 三、维度评价

1. 优势维度: {3} (得分: {4:.3f})
   该维度表现最佳，是上海市康养城市发展的主要优势领域。

2. 不足维度: {5} (得分: {6:.3f})
   该维度表现相对较弱，是需要重点改进的方向。

## 四、改进建议

{7}

## 五、详细指标权重

{8}

## 六、发展趋势分析 (2014-{0})

从长期发展趋势来看，各维度表现如下：

{9}

报告生成时间: {10}
""".format(
        latest_year,
        latest_score['综合得分'],
        '\n'.join([
            "- {0}: 得分{1:.3f}, 年均增长率{2:.2%}, 权重{3:.3f}".format(
                dim,
                latest_score[dim],
                growth_rates[dim],
                dimension_weights[list(indicators.keys()).index(dim)]
            ) for dim in indicators
        ]),
        max_dim,
        latest_score[max_dim],
        min_dim,
        latest_score[min_dim],
        '\n'.join([
            "- {0}".format(s) for s in get_improvement_suggestions(min_dim)
        ]),
        '\n'.join([
            "- {0} - {1}: 权重{2:.4f} ({3})".format(
                dim,
                ind,
                indicator_weights[dim][ind],
                "逆向" if ind in ['老龄人口占比(%)', '新生儿死亡率(%)'] else "正向"
            ) for dim in indicators for ind in indicators[dim]
        ]),
        '\n'.join([
            "- {0}年均增长率: {1:.2%}".format(dim, growth_rates[dim])
            for dim in indicators
        ]),
        pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    # 保存报告到文本文件
    with open(f'{output_dir}/康养城市评价报告.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("分析报告已生成: 康养城市评价报告.md")

def get_improvement_suggestions(min_dim):
    suggestions = {
        '康养资源丰富度': [
            "增加医疗和养老设施建设，特别是在人口密集区域",
            "提升公园绿地和文化设施的可达性",
            "优化养老机构布局，提高服务质量"
        ],
        '居民健康状况': [
            "加强慢性病管理和预防保健服务",
            "推广健康生活方式，提高居民健康素养",
            "完善老年健康服务体系"
        ],
        '环境质量': [
            "进一步提升绿化覆盖率，增加城市绿地",
            "加强空气污染治理，改善空气质量",
            "优化水资源管理，提高用水效率"
        ],
        '经济发展水平': [
            "促进经济高质量发展，提高居民收入",
            "增加对康养产业的财政投入",
            "推动康养产业与其他产业融合发展"
        ]
    }

    return suggestions[min_dim]

# 修改主程序部分
if __name__ == "__main__":
    # 1. 加载和预处理数据
    df = load_and_preprocess_data()
    # 2. 定义评价指标体系
    indicators = define_indicators()
    # 3. 数据标准化
    normalized_df = normalize_data(df, indicators)
    # 4. 计算权重
    dimension_weights, indicator_weights = calculate_weights(indicators)
    # 5. 计算得分
    scores = calculate_scores(normalized_df, indicators, dimension_weights, indicator_weights)
    # 6. 可视化结果
    visualize_results(scores, indicators, df)
    # 7. 模型评价
    evaluate_model(scores, normalized_df, indicators, dimension_weights, indicator_weights)
    # 8. 生成分析报告
    generate_report(scores, df, indicators, dimension_weights, indicator_weights)