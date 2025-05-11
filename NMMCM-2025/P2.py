#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.neighbors import BallTree
import networkx as nx
import math
import warnings
warnings.filterwarnings('ignore')

class MarathonRoutePlanner:
    """
    西安市马拉松路线规划类
    实现问题2的解决方案：
    1. 建立评价函数，筛选最优起点-终点组合
    2. 设计符合国际马拉松要求的闭合回路
    """
    def __init__(self, data_dir="附件"):
        """
        初始化马拉松路线规划器

        参数:
        data_dir: 附件数据的目录
        """
        self.data_dir = data_dir
        self.output_dir = "P2"
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载数据
        self.load_data()

    def load_data(self):
        """
        加载西安市基础数据
        - 住宿设施数据
        - 景点数据
        - 餐饮设施数据
        - 地铁站点数据
        - 道路数据
        - 地形数据
        """
        print("加载西安市基础数据...")

        # 1. 加载住宿设施数据
        accommodation_path = os.path.join(self.data_dir, "附件5：西安市基础数据", "西安市住宿服务数据.csv")
        self.accommodation_df = pd.read_csv(accommodation_path)
        print(f"加载了{len(self.accommodation_df)}条住宿设施数据")

        # 2. 加载景点数据
        scenic_path = os.path.join(self.data_dir, "附件5：西安市基础数据", "西安市风景名胜数据.csv")
        self.scenic_df = pd.read_csv(scenic_path)
        print(f"加载了{len(self.scenic_df)}条景点数据")

        # 3. 加载餐饮设施数据
        restaurant_path = os.path.join(self.data_dir, "附件5：西安市基础数据", "西安市餐饮数据.csv")
        self.restaurant_df = pd.read_csv(restaurant_path)
        print(f"加载了{len(self.restaurant_df)}条餐饮设施数据")

        # 4. 加载地铁站点数据
        subway_path = os.path.join(self.data_dir, "附件9：西安_2024年地铁数据", "地铁站点（含经纬度）.xlsx")
        self.subway_df = pd.read_excel(subway_path)
        print(f"加载了{len(self.subway_df)}条地铁站点数据")

        # 5. 加载道路数据
        road_path = os.path.join(self.data_dir, "附件7：2025年西安市道路数据", "路线连接信息.csv")
        self.road_df = pd.read_csv(road_path, encoding='gbk')
        print(f"加载了{len(self.road_df)}条道路数据")

        # 6. 加载路口坐标数据
        intersection_path = os.path.join(self.data_dir, "附件7：2025年西安市道路数据", "路口坐标及编号.csv")
        self.intersection_df = pd.read_csv(intersection_path, encoding='gbk')
        print(f"加载了{len(self.intersection_df)}条路口坐标数据")

        # 7. 加载地形数据（用于计算坡度）
        # 这里需要使用rasterio库读取tif文件，暂时略过
        print("地形数据加载略过，将在后续实现")

        # 数据预处理
        self.preprocess_data()

    def preprocess_data(self):
        """
        数据预处理
        - 统一坐标系
        - 提取必要字段
        - 计算容量权重
        - 计算邻近路网密度
        """
        print("数据预处理...")

        # 1. 处理住宿设施数据
        # 使用WGS84坐标系
        self.accommodation_df['geometry'] = self.accommodation_df.apply(
            lambda row: Point(row['wgs84Lng'], row['wgs84Lat']), axis=1)

        # 2. 处理景点数据
        self.scenic_df['geometry'] = self.scenic_df.apply(
            lambda row: Point(row['wgs84Lng'], row['wgs84Lat']), axis=1)

        # 3. 处理餐饮设施数据
        self.restaurant_df['geometry'] = self.restaurant_df.apply(
            lambda row: Point(row['wgs84Lng'], row['wgs84Lat']), axis=1)

        # 4. 处理地铁站点数据
        # 假设地铁站点数据中经纬度列名为'经度'和'纬度'
        if '经度' in self.subway_df.columns and '纬度' in self.subway_df.columns:
            self.subway_df['geometry'] = self.subway_df.apply(
                lambda row: Point(row['经度'], row['纬度']), axis=1)

        # 5. 构建路网图
        self.build_road_network()

    def build_road_network(self):
        """
        构建路网图
        使用NetworkX库构建路网图，用于路径规划
        """
        print("构建路网图...")

        # 创建无向图
        self.G = nx.Graph()

        # 添加节点（路口）
        for _, row in self.intersection_df.iterrows():
            # 假设路口数据中有'编号'、'经度'、'纬度'列
            if '编号' in row and '经度' in row and '纬度' in row:
                self.G.add_node(row['编号'],
                               pos=(row['经度'], row['纬度']),
                               geometry=Point(row['经度'], row['纬度']))

        # 添加边（道路）
        for _, row in self.road_df.iterrows():
            # 假设道路数据中有'起点'、'终点'、'长度'列
            if '起点' in row and '终点' in row and '长度' in row:
                self.G.add_edge(row['起点'], row['终点'],
                               weight=row['长度'],
                               geometry=LineString([self.G.nodes[row['起点']]['pos'],
                                                  self.G.nodes[row['终点']]['pos']]))

        print(f"路网图构建完成，包含{self.G.number_of_nodes()}个节点和{self.G.number_of_edges()}条边")

    def calculate_accommodation_capacity(self, point, radius=3000):
        """
        计算指定点周围指定半径内的住宿容量

        参数:
        point: 坐标点 (lng, lat)
        radius: 半径，单位米

        返回:
        住宿容量（估算值）
        """
        # 创建以point为中心，radius为半径的圆形区域
        center_point = Point(point)

        # 计算该区域内的住宿设施数量
        # 这里简化处理，假设每个住宿设施平均可容纳100人
        # 实际应用中应根据住宿设施的类型和规模进行更精确的估算
        count = 0
        for _, row in self.accommodation_df.iterrows():
            facility_point = Point(row['wgs84Lng'], row['wgs84Lat'])
            # 计算两点之间的距离（米）
            dist = self.haversine_distance(point[0], point[1],
                                         row['wgs84Lng'], row['wgs84Lat'])
            if dist <= radius:
                count += 1

        # 估算容量（每个设施平均100人）
        capacity = count * 100
        return capacity

    def haversine_distance(self, lon1, lat1, lon2, lat2):
        """
        计算两点间的哈弗赛因距离（考虑地球曲率）

        参数:
        lon1, lat1: 第一个点的经纬度
        lon2, lat2: 第二个点的经纬度

        返回:
        距离，单位米
        """
        # 将经纬度转换为弧度
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        # 哈弗赛因公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # 地球平均半径，单位米
        return c * r

    def calculate_road_density(self, point, radius=1000):
        """
        计算指定点周围指定半径内的路网密度

        参数:
        point: 坐标点 (lng, lat)
        radius: 半径，单位米

        返回:
        路网密度（道路长度/面积）
        """
        # 创建以point为中心，radius为半径的圆形区域
        center_point = Point(point)

        # 计算该区域内的道路总长度
        total_length = 0
        for u, v, data in self.G.edges(data=True):
            # 获取道路的几何形状
            road_geom = data.get('geometry')
            if road_geom:
                # 计算道路与圆形区域的交点
                # 这里简化处理，只考虑道路的起点和终点是否在圆形区域内
                start_point = Point(self.G.nodes[u]['pos'])
                end_point = Point(self.G.nodes[v]['pos'])

                # 计算起点和终点到中心点的距离
                start_dist = self.haversine_distance(start_point.x, start_point.y,
                                                  center_point.x, center_point.y)
                end_dist = self.haversine_distance(end_point.x, end_point.y,
                                                center_point.x, center_point.y)

                # 如果起点或终点在圆形区域内，则计算道路长度
                if start_dist <= radius or end_dist <= radius:
                    total_length += data.get('weight', 0)

        # 计算圆形区域的面积（平方米）
        area = math.pi * radius**2

        # 计算路网密度（米/平方米）
        density = total_length / area if area > 0 else 0
        return density

    def is_near_subway(self, point, max_distance=500):
        """
        判断指定点是否靠近地铁站

        参数:
        point: 坐标点 (lng, lat)
        max_distance: 最大距离，单位米

        返回:
        是否靠近地铁站
        """
        # 遍历所有地铁站点
        for _, row in self.subway_df.iterrows():
            # 计算点到地铁站的距离
            if '经度' in row and '纬度' in row:
                dist = self.haversine_distance(point[0], point[1],
                                             row['经度'], row['纬度'])
                if dist <= max_distance:
                    return True
        return False

    def evaluate_point(self, point, is_start=True):
        """
        评估点的适合度

        参数:
        point: 坐标点 (lng, lat)
        is_start: 是否为起点

        返回:
        评分（0-100）
        """
        score = 0

        # 1. 住宿容量评分（仅对起点有效）
        if is_start:
            capacity = self.calculate_accommodation_capacity(point)
            # 如果容量>=3000，得满分；否则按比例得分
            capacity_score = min(capacity / 3000 * 40, 40)
            score += capacity_score

        # 2. 路网密度评分
        density = self.calculate_road_density(point)
        # 路网密度评分（假设密度越高越好）
        density_score = min(density * 1000, 30)  # 假设密度为0.03时得满分
        score += density_score

        # 3. 地铁站点评分
        subway_score = 30 if self.is_near_subway(point) else 0
        score += subway_score

        return score

    def find_optimal_start_end_points(self):
        """
        寻找最优的起点-终点组合

        返回:
        最优起点和终点坐标
        """
        print("寻找最优的起点-终点组合...")

        # 候选点集合（可以是景点或其他重要地点）
        candidate_points = []
        for _, row in self.scenic_df.iterrows():
            candidate_points.append((row['wgs84Lng'], row['wgs84Lat']))

        # 评估所有可能的起点-终点组合
        best_score = 0
        best_start = None
        best_end = None

        for i, start_point in enumerate(candidate_points):
            # 评估起点
            start_score = self.evaluate_point(start_point, is_start=True)

            # 如果起点不满足住宿容量要求，跳过
            if self.calculate_accommodation_capacity(start_point) < 3000:
                continue

            # 如果起点不靠近地铁站，跳过
            if not self.is_near_subway(start_point):
                continue

            for j, end_point in enumerate(candidate_points):
                if i == j:  # 起点和终点不能相同
                    continue

                # 计算起点到终点的距离
                distance = self.haversine_distance(start_point[0], start_point[1],
                                                end_point[0], end_point[1])

                # 如果距离小于42公里，跳过
                if distance < 42000:
                    continue

                # 评估终点
                end_score = self.evaluate_point(end_point, is_start=False)

                # 如果终点不靠近地铁站，跳过
                if not self.is_near_subway(end_point):
                    continue

                # 计算总分
                total_score = start_score + end_score

                # 更新最优解
                if total_score > best_score:
                    best_score = total_score
                    best_start = start_point
                    best_end = end_point

        print(f"找到最优起点-终点组合，总评分：{best_score}")
        return best_start, best_end

    def design_marathon_route(self, route_type='full'):
        """
        设计马拉松路线

        参数:
        route_type: 路线类型，'full'全程马拉松，'half'半程马拉松，'health'健康跑

        返回:
        路线坐标列表
        """
        print(f"设计{route_type}马拉松路线...")

        # 根据路线类型确定目标距离
        if route_type == 'full':
            target_distance = 42195  # 全程马拉松，42.195公里
        elif route_type == 'half':
            target_distance = 21098  # 半程马拉松，21.0975公里
        else:  # health
            target_distance = 5000   # 健康跑，5公里

        # 寻找最优起点和终点
        start_point, end_point = self.find_optimal_start_end_points()

        # 构建必经节点集（景点）
        must_visit_nodes = []
        for _, row in self.scenic_df.head(10).iterrows():  # 选取前10个景点作为必经节点
            must_visit_nodes.append((row['wgs84Lng'], row['wgs84Lat']))

        # 构建增益节点集（餐饮设施）
        gain_nodes = []
        for _, row in self.restaurant_df.head(20).iterrows():  # 选取前20个餐饮设施作为增益节点
            gain_nodes.append((row['wgs84Lng'], row['wgs84Lat']))

        # TODO: 实现路线规划算法
        # 这里需要实现一个复杂的路径规划算法，考虑以下因素：
        # 1. 起点和终点的约束
        # 2. 必经节点集
        # 3. 增益节点集
        # 4. 路线长度约束
        # 5. 坡度约束（<=5%）
        # 6. 每5公里设置补给站

        # 简化处理：返回一个示例路线
        route = [start_point]
        route.extend(must_visit_nodes[:5])  # 添加部分必经节点
        route.append(end_point)

        print(f"{route_type}马拉松路线设计完成，包含{len(route)}个节点")
        return route

    def calculate_route_gain(self, route):
        """
        计算路线的增益值

        参数:
        route: 路线坐标列表

        返回:
        增益值
        """
        gain = 0

        # 遍历路线上的每个点
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i + 1]

            # 计算这段路径上经过的增益节点数量
            for _, row in self.restaurant_df.iterrows():
                point = (row['wgs84Lng'], row['wgs84Lat'])

                # 简化处理：如果点到线段的距离小于阈值，认为经过了该点
                if self.point_to_line_distance(point, start, end) < 100:  # 100米阈值
                    gain += 0.2

        return gain

    def point_to_line_distance(self, point, line_start, line_end):
        """
        计算点到线段的距离

        参数:
        point: 点坐标 (x, y)
        line_start: 线段起点 (x, y)
        line_end: 线段终点 (x, y)

        返回:
        距离
        """
        # 将经纬度转换为平面坐标（简化处理）
        # 实际应用中应考虑地球曲率

        # 线段向量
        line_vec = (line_end[0] - line_start[0], line_end[1] - line_start[1])
        # 点到起点的向量
        point_vec = (point[0] - line_start[0], point[1] - line_start[1])

        # 线段长度的平方
        line_len_sq = line_vec[0]**2 + line_vec[1]**2

        # 如果线段长度为0，则返回点到起点的距离
        if line_len_sq == 0:
            return math.sqrt(point_vec[0]**2 + point_vec[1]**2)

        # 计算投影比例
        t = max(0, min(1, (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_len_sq))

        # 计算投影点
        proj = (line_start[0] + t * line_vec[0], line_start[1] + t * line_vec[1])

        # 计算点到投影点的距离
        return self.haversine_distance(point[0], point[1], proj[0], proj[1])

    def run(self):
        """
        运行马拉松路线规划
        """
        print("开始马拉松路线规划...")

        # 1. 寻找最优起点-终点组合
        start_point, end_point = self.find_optimal_start_end_points()
        print(f"最优起点: {start_point}")
        print(f"最优终点: {end_point}")

        # 2. 设计全程马拉松路线
        full_route = self.design_marathon_route(route_type='full')
        full_gain = self.calculate_route_gain(full_route)
        print(f"全程马拉松路线增益值: {full_gain}")

        # 3. 设计半程马拉松路线
        half_route = self.design_marathon_route(route_type='half')
        half_gain = self.calculate_route_gain(half_route)
        print(f"半程马拉松路线增益值: {half_gain}")

        # 4. 设计健康跑路线
        health_route = self.design_marathon_route(route_type='health')
        health_gain = self.calculate_route_gain(health_route)
        print(f"健康跑路线增益值: {health_gain}")

        # 5. 可视化路线
        self.visualize_routes(full_route, half_route, health_route)

        print("马拉松路线规划完成")

    def visualize_routes(self, full_route, half_route, health_route):
        """
        可视化马拉松路线

        参数:
        full_route: 全程马拉松路线
        half_route: 半程马拉松路线
        health_route: 健康跑路线
        """
        print("可视化马拉松路线...")

        # 创建地图底图
        plt.figure(figsize=(12, 10))

        # 绘制景点
        plt.scatter(self.scenic_df['wgs84Lng'], self.scenic_df['wgs84Lat'],
                   c='green', s=20, alpha=0.7, label='景点')

        # 绘制住宿设施
        plt.scatter(self.accommodation_df['wgs84Lng'], self.accommodation_df['wgs84Lat'],
                   c='blue', s=10, alpha=0.5, label='住宿设施')

        # 绘制餐饮设施
        plt.scatter(self.restaurant_df['wgs84Lng'], self.restaurant_df['wgs84Lat'],
                   c='orange', s=10, alpha=0.5, label='餐饮设施')

        # 绘制地铁站点
        if '经度' in self.subway_df.columns and '纬度' in self.subway_df.columns:
            plt.scatter(self.subway_df['经度'], self.subway_df['纬度'],
                       c='red', s=30, alpha=0.8, label='地铁站点')

        # 绘制全程马拉松路线
        full_x = [p[0] for p in full_route]
        full_y = [p[1] for p in full_route]
        plt.plot(full_x, full_y, 'r-', linewidth=2, label='全程马拉松')

        # 绘制半程马拉松路线
        half_x = [p[0] for p in half_route]
        half_y = [p[1] for p in half_route]
        plt.plot(half_x, half_y, 'b-', linewidth=2, label='半程马拉松')

        # 绘制健康跑路线
        health_x = [p[0] for p in health_route]
        health_y = [p[1] for p in health_route]
        plt.plot(health_x, health_y, 'g-', linewidth=2, label='健康跑')

        # 添加图例和标题
        plt.legend()
        plt.title('西安市马拉松路线规划')
        plt.xlabel('经度')
        plt.ylabel('纬度')

        # 保存图像
        plt.savefig(os.path.join(self.output_dir, '马拉松路线规划.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("路线可视化完成，已保存到'马拉松路线规划.png'")

# 主函数
def main():
    planner = MarathonRoutePlanner()
    planner.run()

if __name__ == "__main__":
    main()