import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
import os
warnings.filterwarnings('ignore')

class KangYangResourceAnalysis:
    def __init__(self):
        self.population_data = None
        self.shanghai_data = None
        self.city_ranking = None
        self.processed_data = None
        self.results = {}

    def load_data(self):
        """加载所需数据"""
        try:
            self.population_data = pd.read_csv(r'CYCMCM-2025\dataset\各地区分年龄、性别的人口(城市).csv')
            self.shanghai_data = pd.read_csv(r'CYCMCM-2025\dataset\上海数据.csv')
            self.city_ranking = pd.read_csv(r'CYCMCM-2025\dataset\2024 年中国康养城市100强名单.csv')
            print("数据加载成功")
        except Exception as e:
            print(f"数据加载错误: {e}")

    def preprocess_data(self):
        """数据预处理：处理缺失值、异常值、数据类型转换等"""
        # 处理上海数据
        if self.shanghai_data is not None:
            # 去除百分号并转为float
            if '空气质量指数（%）' in self.shanghai_data['指标'].values:
                idx = self.shanghai_data[self.shanghai_data['指标'] == '空气质量指数（%）'].index[0]
                self.shanghai_data.iloc[idx, 1:] = self.shanghai_data.iloc[idx, 1:].str.replace('%', '').astype(float)
            # 转换为宽表
            self.shanghai_data = self.shanghai_data.set_index('指标').T
            self.shanghai_data = self.shanghai_data.apply(pd.to_numeric, errors='ignore')
            self.shanghai_data.index.name = '年份'
            self.shanghai_data.reset_index(inplace=True)
        # 处理人口数据
        if self.population_data is not None:
            self.population_data = self.population_data.rename(columns=lambda x: x.strip())
            self.population_data = self.population_data.dropna(how='all')
            self.population_data = self.population_data[self.population_data['地    区'].notnull()]
        print("数据预处理完成")

    def analyze_resource_density(self):
        """分析康养资源空间分布密度"""
        # 1. 计算核密度估计
        # 2. 生成热力图
        # 3. 识别高密度和低密度区域
        pass

    def calculate_resource_coverage(self):
        """计算资源覆盖率"""
        # 1. 定义服务半径
        # 2. 计算覆盖区域
        # 3. 评估覆盖效果
        pass

    def identify_resource_gaps(self):
        """识别资源空白区"""
        # 1. 设定阈值
        # 2. 识别低密度区域
        # 3. 提出优化建议
        pass

    def cluster_analysis(self):
        """聚类分析"""
        if self.city_ranking is None:
            print("城市排名数据未加载")
            return

        # 1. 准备数据
        data = self.city_ranking[['康养指数']].values
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # 2. KMeans聚类分析
        n_clusters_range = range(2, 6)
        silhouette_scores = []

        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data_scaled)
            score = silhouette_score(data_scaled, cluster_labels)
            silhouette_scores.append(score)

        # 选择最佳聚类数
        best_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]

        # 使用最佳聚类数进行聚类
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
        self.city_ranking['cluster'] = kmeans.fit_predict(data_scaled)

        # 3. 分析聚类结果
        cluster_stats = self.city_ranking.groupby('cluster').agg({
            '康养指数': ['count', 'mean', 'std', 'min', 'max']
        }).round(2)

        # 4. 可视化聚类结果
        plt.figure(figsize=(10, 6))
        for i in range(best_n_clusters):
            cluster_data = self.city_ranking[self.city_ranking['cluster'] == i]
            plt.scatter(cluster_data.index, cluster_data['康养指数'],
                       label=f'聚类 {i+1}')

        plt.title('康养城市聚类分析结果')
        plt.xlabel('城市排名')
        plt.ylabel('康养指数')
        plt.legend()

        # 保存图表
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'city_clusters.png'))
        plt.close()

        # 5. 找出上海所属的聚类
        shanghai = self.city_ranking[self.city_ranking['城市'] == '上海市']
        if not shanghai.empty:
            shanghai_cluster = shanghai['cluster'].values[0]
            cluster_cities = self.city_ranking[self.city_ranking['cluster'] == shanghai_cluster]

            print("\n聚类分析结果：")
            print(f"最佳聚类数：{best_n_clusters}")
            print(f"\n上海市所属聚类：{shanghai_cluster + 1}")
            print(f"同类城市数量：{len(cluster_cities)}")
            print("\n同类城市康养指数统计：")
            print(cluster_stats.loc[shanghai_cluster])

            # 保存同类城市列表
            cluster_cities_list = cluster_cities[['城市', '康养指数']].sort_values(
                by='康养指数', ascending=False)
            cluster_cities_list.to_csv(
                os.path.join(output_dir, f'cluster_{shanghai_cluster + 1}_cities.csv'),
                encoding='utf-8-sig', index=False)

            self.results['cluster_analysis'] = {
                'best_n_clusters': best_n_clusters,
                'shanghai_cluster': shanghai_cluster,
                'cluster_stats': cluster_stats,
                'silhouette_scores': silhouette_scores
            }

    def accessibility_analysis(self):
        """可达性分析"""
        # 1. 计算服务范围
        # 2. 评估可达性
        # 3. 识别改进区域
        pass

    def sensitivity_analysis(self):
        """敏感性分析"""
        # 1. 参数敏感性
        # 2. 模型稳定性
        # 3. 结果可靠性
        pass

    def visualize_results(self):
        """可视化分析结果"""
        # 1. 创建交互式地图
        # 2. 绘制统计图表
        # 3. 生成分析报告
        pass

    def generate_report(self):
        """生成分析报告"""
        # 1. 汇总分析结果
        # 2. 生成建议
        # 3. 输出报告
        pass

    def validate_results(self):
        """验证分析结果"""
        # 1. 交叉验证
        # 2. 结果评估
        # 3. 可靠性分析
        pass

    def resource_population_ratio(self):
        """计算上海市主要康养资源的人均/老年人均占有量"""
        if self.shanghai_data is None:
            print("上海数据未加载")
            return None
        # 以2023年为例
        row = self.shanghai_data[self.shanghai_data['年份'] == '2023']
        if row.empty:
            row = self.shanghai_data.iloc[0:1]
        row = row.squeeze()
        # 获取老龄人口占比
        old_ratio = float(row['老龄人口占比（%）']) / 100 if '老龄人口占比（%）' in row else 0.14
        total_pop = float(row['常驻人口（万人）']) * 10000 if '常驻人口（万人）' in row else 24874500
        old_pop = total_pop * old_ratio
        # 主要资源
        hospital_num = float(row['医疗卫生机构数(个)'])
        park_num = float(row['公园个数(个)'])
        bed_num = float(row['养老床位（万张）']) * 10000
        green_area = float(row['公园绿地面积（万顷）']) * 10000  # 转为公顷
        # 计算人均
        result = {
            '人均医疗机构数': hospital_num / total_pop,
            '人均公园数': park_num / total_pop,
            '人均养老床位': bed_num / total_pop,
            '人均绿地面积(公顷)': green_area / total_pop,
            '老年人均养老床位': bed_num / old_pop if old_pop > 0 else None
        }
        print("上海市2023年主要康养资源人均/老年人均占有量：")
        for k, v in result.items():
            print(f"{k}: {v:.4f}")
        return result

    def time_series_analysis(self):
        """时间序列分析"""
        if self.shanghai_data is None:
            print("上海数据未加载")
            return

        # 1. 计算关键指标的增长率
        metrics = ['医疗卫生机构数(个)', '养老机构数量（个）', '公园个数(个)',
                  '养老床位（万张）', '老龄人口占比（%）']

        growth_rates = pd.DataFrame()

        for metric in metrics:
            if metric in self.shanghai_data.columns:
                # 计算年增长率
                values = self.shanghai_data[metric].astype(float)
                growth_rate = (values.shift(-1) - values) / values * 100
                growth_rates[f'{metric}_增长率'] = growth_rate

        # 2. 计算趋势指标
        trends = {}
        for metric in metrics:
            if metric in self.shanghai_data.columns:
                values = self.shanghai_data[metric].astype(float)
                trend = {
                    '总体增长率': ((values.iloc[0] - values.iloc[-1]) / values.iloc[-1] * 100),
                    '年均增长率': ((values.iloc[0] / values.iloc[-1]) ** (1/len(values)) - 1) * 100,
                    '5年增长率': ((values.iloc[0] - values.iloc[5]) / values.iloc[5] * 100) if len(values) > 5 else None
                }
                trends[metric] = trend

        # 3. 可视化分析
        plt.figure(figsize=(15, 8))

        # 3.1 资源增长趋势
        for metric in metrics:
            if metric in self.shanghai_data.columns:
                values = self.shanghai_data[metric].astype(float)
                # 标准化处理，便于比较
                normalized = (values - values.min()) / (values.max() - values.min())
                plt.plot(self.shanghai_data['年份'], normalized, marker='o', label=metric)

        plt.title('上海市康养资源发展趋势(标准化)')
        plt.xlabel('年份')
        plt.ylabel('标准化值')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存图表
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'resource_growth_trend.png'))
        plt.close()

        # 4. 输出分析结果
        print("\n时间序列分析结果：")
        print("\n各指标增长趋势：")
        for metric, trend in trends.items():
            print(f"\n{metric}:")
            for k, v in trend.items():
                if v is not None:
                    print(f"- {k}: {v:.2f}%")

        # 5. 存储结果
        self.results['time_series_analysis'] = {
            'growth_rates': growth_rates,
            'trends': trends
        }

        return trends

def main():
    # 创建分析器实例
    analyzer = KangYangResourceAnalysis()

    # 数据加载和预处理
    analyzer.load_data()
    analyzer.preprocess_data()

    # 基础分析
    print("\n计算资源人均占有量...")
    analyzer.resource_population_ratio()

    # 时间序列分析
    print("\n进行时间序列分析...")
    analyzer.time_series_analysis()

    # 聚类分析
    print("\n进行城市聚类分析...")
    analyzer.cluster_analysis()

    # 可视化结果
    print("\n生成可视化结果...")
    analyzer.visualize_results()

    # 生成报告
    print("\n生成分析报告...")
    analyzer.generate_report()

    print("\n分析完成！请查看output目录下的分析报告和可视化结果。")

if __name__ == "__main__":
    main()
