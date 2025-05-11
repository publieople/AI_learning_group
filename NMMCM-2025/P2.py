import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
from math import radians, sin, cos, sqrt, atan2
import folium
from folium.plugins import MarkerCluster
import os

# 高德地图API密钥
API_KEY = "8ca1e4d717a1f7c095f78b2a127c96ea" # 请替换为您的实际密钥

# 数据路径
ACCOMMODATION_PATH = "processed_data/cleaned/附件5_xian_hotels_cleaned.csv"
ATTRACTIONS_PATH = "processed_data/cleaned/附件5_xian_attractions_cleaned.csv"
RESTAURANTS_PATH = "processed_data/cleaned/附件5_xian_restaurants_cleaned.csv"
SUBWAY_STATIONS_PATH = "processed_data/附件9_xian_subway_stations.csv"
ROAD_PATH = "processed_data/附件7_xian_road_connections.csv"

# 加载和处理数据
def load_data():
    """加载西安市的基础数据"""
    try:
        # 加载住宿设施数据
        accommodations = pd.read_csv(ACCOMMODATION_PATH)
        print(f"住宿设施数据加载成功，共 {len(accommodations)} 条记录")

        # 确保住宿数据有经纬度列
        # 根据实际数据，应该使用gcjLng/gcjLat作为经纬度
        if 'longitude' not in accommodations.columns:
            if 'gcjLng' in accommodations.columns:
                accommodations['longitude'] = accommodations['gcjLng']
            elif 'location' in accommodations.columns:
                # 如果有location列且格式为"lng,lat"，则分割
                try:
                    accommodations[['longitude', 'latitude']] = accommodations['location'].str.split(',', expand=True).astype(float)
                except:
                    print("无法从location列提取经纬度")

        if 'latitude' not in accommodations.columns and 'gcjLat' in accommodations.columns:
            accommodations['latitude'] = accommodations['gcjLat']

        # 确保有容量信息
        if 'capacity' not in accommodations.columns:
            if 'rooms' in accommodations.columns:
                # 估算容量：假设每个房间可以住2人
                accommodations['capacity'] = accommodations['rooms'] * 2
            else:
                # 默认容量为50人
                print("住宿设施数据中没有容量信息，使用默认值100")
                accommodations['capacity'] = 100

        # 加载景点数据
        attractions = pd.read_csv(ATTRACTIONS_PATH)
        print(f"景点数据加载成功，共 {len(attractions)} 条记录")

        # 确保景点数据有经纬度列
        if 'longitude' not in attractions.columns:
            if 'gcjLng' in attractions.columns:
                attractions['longitude'] = attractions['gcjLng']
            elif 'location' in attractions.columns:
                try:
                    attractions[['longitude', 'latitude']] = attractions['location'].str.split(',', expand=True).astype(float)
                except:
                    print("无法从location列提取景点经纬度")

        if 'latitude' not in attractions.columns and 'gcjLat' in attractions.columns:
            attractions['latitude'] = attractions['gcjLat']

        # 加载餐饮设施数据
        restaurants = pd.read_csv(RESTAURANTS_PATH)
        print(f"餐饮设施数据加载成功，共 {len(restaurants)} 条记录")

        # 确保餐饮数据有经纬度列
        if 'longitude' not in restaurants.columns:
            if 'gcjLng' in restaurants.columns:
                restaurants['longitude'] = restaurants['gcjLng']
            elif 'location' in restaurants.columns:
                try:
                    restaurants[['longitude', 'latitude']] = restaurants['location'].str.split(',', expand=True).astype(float)
                except:
                    print("无法从location列提取餐饮设施经纬度")

        if 'latitude' not in restaurants.columns and 'gcjLat' in restaurants.columns:
            restaurants['latitude'] = restaurants['gcjLat']

        # 加载地铁站数据
        subway_stations = pd.read_csv(SUBWAY_STATIONS_PATH)
        print(f"地铁站数据加载成功，共 {len(subway_stations)} 条记录")

        # 确保地铁站数据有经纬度列
        if 'longitude' not in subway_stations.columns and '经度' in subway_stations.columns:
            subway_stations['longitude'] = subway_stations['经度']
        if 'latitude' not in subway_stations.columns and '纬度' in subway_stations.columns:
            subway_stations['latitude'] = subway_stations['纬度']

        # 加载道路连接数据
        try:
            road_connections = pd.read_csv(ROAD_PATH)
            print(f"道路连接数据加载成功，共 {len(road_connections)} 条记录")
        except Exception as e:
            print(f"道路连接数据加载失败: {e}")
            road_connections = None

        return {
            'accommodations': accommodations,
            'attractions': attractions,
            'restaurants': restaurants,
            'subway_stations': subway_stations,
            'road_connections': road_connections
        }
    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# 坐标转换为高德坐标
def convert_to_amap_coordinates(coordinates, from_type="gps"):
    """
    将坐标转换为高德坐标系
    coordinates: 字符串，格式为"lng1,lat1|lng2,lat2|..."
    from_type: 原坐标类型，可选值: gps, mapbar, baidu
    """
    url = "https://restapi.amap.com/v3/assistant/coordinate/convert"
    params = {
        "key": API_KEY,
        "locations": coordinates,
        "coordsys": from_type
    }

    try:
        response = requests.get(url, params=params)
        result = response.json()

        if result["status"] == "1":
            return result["locations"]
        else:
            print(f"坐标转换失败: {result['info']}")
            return None
    except Exception as e:
        print(f"坐标转换请求失败: {e}")
        return None

# 计算两点之间的直线距离
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    使用Haversine公式计算两点间的直线距离（单位：公里）
    """
    # 将经纬度转换为弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = 6371 * c  # 地球平均半径，单位为公里

    return distance

# 检查位置是否毗邻轨道交通站点
def is_near_subway(location, subway_stations_data, max_distance=0.8):  # 增加到800米
    """
    检查位置是否在指定距离内有地铁站
    location: [lng, lat]
    subway_stations_data: 地铁站数据
    max_distance: 最大距离（公里），默认800米
    """
    for _, station in subway_stations_data.iterrows():
        station_loc = None

        # 尝试不同的列名获取地铁站坐标
        if 'longitude' in station and 'latitude' in station:
            station_loc = [float(station['longitude']), float(station['latitude'])]
        elif 'gcjLng' in station and 'gcjLat' in station:
            station_loc = [float(station['gcjLng']), float(station['gcjLat'])]
        elif '经度' in station and '纬度' in station:
            station_loc = [float(station['经度']), float(station['纬度'])]

        # 如果无法获取坐标，跳过此地铁站
        if not station_loc:
            continue

        try:
            distance = haversine_distance(location[1], location[0], station_loc[1], station_loc[0])
            if distance <= max_distance:
                return True
        except Exception as e:
            print(f"计算与地铁站距离时出错: {e}")
            continue

    return False

# 使用高德API计算路径规划
def calculate_route(origin, destination, route_type="walk"):
    """
    使用高德路径规划API计算两点之间的路线
    origin: 起点坐标，格式为"lng,lat"
    destination: 终点坐标，格式为"lng,lat"
    route_type: 路线类型，可选值: "walk"(步行), "drive"(驾车)
    返回路线长度（米）和路线信息
    """
    if route_type == "walk":
        url = "https://restapi.amap.com/v3/direction/walking"
        params = {
            "key": API_KEY,
            "origin": origin,
            "destination": destination,
            "output": "JSON"
        }
    else:  # 驾车路线
        url = "https://restapi.amap.com/v3/direction/driving"
        params = {
            "key": API_KEY,
            "origin": origin,
            "destination": destination,
            "output": "JSON",
            "strategy": 5,  # 使用步行与公交综合模式，最适合跑步路线规划
            "extensions": "all"  # 返回详细信息
        }

    try:
        response = requests.get(url, params=params)
        result = response.json()

        if result["status"] == "1" and "route" in result:
            # 获取路线长度（米）
            if "paths" in result["route"] and len(result["route"]["paths"]) > 0:
                distance = float(result["route"]["paths"][0]["distance"])  # 确保转换为浮点数
                # 路线信息
                route_info = result["route"]["paths"][0]
                return distance, route_info
            else:
                print(f"未找到有效路径")
                return None, None
        else:
            error_info = result.get("info", "未知错误")
            print(f"路径规划失败: {error_info}")
            return None, None
    except Exception as e:
        print(f"路径规划请求失败: {e}")
        return None, None

# 查找住宿设施周围3000米内的住宿容量
def get_accommodation_capacity(location, accommodations_data):
    """
    计算给定位置3000米内的住宿总容量
    location: [lng, lat]
    accommodations_data: 住宿设施数据
    返回住宿总容量
    """
    total_capacity = 0
    for _, acc in accommodations_data.iterrows():
        # 获取住宿设施坐标
        acc_loc = None

        # 尝试不同的列名获取坐标
        if 'longitude' in acc and 'latitude' in acc:
            acc_loc = [float(acc['longitude']), float(acc['latitude'])]
        elif 'gcjLng' in acc and 'gcjLat' in acc:
            acc_loc = [float(acc['gcjLng']), float(acc['gcjLat'])]
        elif 'location' in acc and isinstance(acc['location'], str):
            try:
                coords = acc['location'].split(',')
                if len(coords) == 2:
                    acc_loc = [float(coords[0]), float(coords[1])]
            except:
                continue

        # 如果无法获取坐标，跳过此住宿设施
        if not acc_loc:
            continue

        try:
            distance = haversine_distance(location[1], location[0], acc_loc[1], acc_loc[0])
            if distance <= 3.0:  # 3000米 = 3.0公里
                # 获取容量信息
                capacity = acc.get('capacity', 0)
                if capacity == 0 and 'rooms' in acc:
                    # 如果有房间数但没有容量，估算容量
                    capacity = float(acc['rooms']) * 2

                # 确保容量值为有效数字
                if isinstance(capacity, (int, float)) and not np.isnan(capacity):
                    total_capacity += capacity
        except Exception as e:
            print(f"计算住宿容量时出错: {e}")
            continue

    return total_capacity

# 评价函数
def evaluate_route(origin, destination, route_info, accommodations_data, subway_stations_data):
    """
    评价起点-终点组合的质量
    考虑因素：住宿容量、邻近路网密度、交通便利性等
    返回评分（越高越好）
    """
    # 获取起点周围的住宿容量
    accommodation_capacity = get_accommodation_capacity(origin, accommodations_data)

    # 检查起点和终点是否毗邻轨道交通站点
    origin_near_subway = is_near_subway(origin, subway_stations_data)
    destination_near_subway = is_near_subway(destination, subway_stations_data)

    # 路线长度需大于等于42公里
    route_length = float(route_info.get("distance", 0)) / 1000  # 转换为公里，确保先转为浮点数

    # 评分计算
    score = 0

    # 住宿容量评分（满足条件才计算后续评分）
    if accommodation_capacity >= 3000 and origin_near_subway and destination_near_subway and route_length >= 42:
        # 住宿容量分数（超过3000的部分给予奖励）
        capacity_score = min(1.0, (accommodation_capacity - 3000) / 7000) * 30

        # 路线质量分数（考虑路网密度、交通便利性等）
        route_score = min(1.0, (route_length - 42) / 8) * 20  # 适度奖励更长的路线，但不超过50公里

        # 交通便利性分数
        transport_score = 20 if origin_near_subway and destination_near_subway else 0

        # 景观价值分数（这部分需要根据实际数据补充）
        scenery_score = 0  # 待完善

        score = capacity_score + route_score + transport_score + scenery_score

    return score

# 找出最优起点-终点组合
def find_optimal_start_end_points(data):
    """
    寻找最优的起点-终点组合
    """
    accommodations = data['accommodations']
    subway_stations = data['subway_stations']

    print("正在优化数据集大小...")
    # 限制处理的数据量
    max_accommodations = 10000
    if len(accommodations) > max_accommodations:
        print(f"住宿设施数据量过大，随机抽样{max_accommodations}条记录进行处理")
        accommodations = accommodations.sample(max_accommodations)

    # 预计算所有住宿设施的容量
    print("预计算住宿容量...")
    accommodation_capacities = {}
    for _, acc in accommodations.iterrows():
        # 获取住宿设施ID
        acc_id = acc.get('id', str(acc.name))
        # 获取容量
        capacity = acc.get('capacity', 0)
        if capacity == 0 and 'rooms' in acc:
            capacity = float(acc['rooms']) * 2
        accommodation_capacities[acc_id] = capacity

    # 预先计算所有地铁站的坐标，用于快速计算距离
    subway_coords = []
    for _, station in subway_stations.iterrows():
        station_loc = None
        if 'longitude' in station and 'latitude' in station:
            station_loc = [float(station['longitude']), float(station['latitude'])]
        elif 'gcjLng' in station and 'gcjLat' in station:
            station_loc = [float(station['gcjLng']), float(station['gcjLat'])]
        elif '经度' in station and '纬度' in station:
            station_loc = [float(station['经度']), float(station['纬度'])]

        if station_loc:
            subway_coords.append(station_loc)

    print(f"有效地铁站点数量: {len(subway_coords)}")

    # 筛选可能的起点（住宿容量≥1000且毗邻轨道交通站点）
    candidate_starts = []

    # 快速筛选：先用直线距离初步筛选毗邻地铁站的住宿设施
    potential_starts = []
    for _, acc in tqdm(accommodations.iterrows(), desc="快速筛选起点", total=len(accommodations)):
        acc_loc = None
        # 获取住宿设施坐标
        if 'longitude' in acc and 'latitude' in acc:
            acc_loc = [float(acc['longitude']), float(acc['latitude'])]
        elif 'gcjLng' in acc and 'gcjLat' in acc:
            acc_loc = [float(acc['gcjLng']), float(acc['gcjLat'])]
        elif 'location' in acc and isinstance(acc['location'], str):
            try:
                coords = acc['location'].split(',')
                if len(coords) == 2:
                    acc_loc = [float(coords[0]), float(coords[1])]
            except:
                continue

        if not acc_loc:
            continue

        # 检查是否毗邻地铁站（快速估计）
        near_subway = False
        for subway_loc in subway_coords:
            distance = haversine_distance(acc_loc[1], acc_loc[0], subway_loc[1], subway_loc[0])
            if distance <= 0.8:  # 800米内，放宽条件
                near_subway = True
                break

        if near_subway:
            # 获取容量信息
            acc_id = acc.get('id', str(acc.name))
            capacity = accommodation_capacities.get(acc_id, 0)

            # 降低容量阈值，放宽条件
            if capacity >= 300:  # 降低到300人的容量要求
                potential_starts.append((acc_loc, capacity))

    print(f"初步筛选后的潜在起点数量: {len(potential_starts)}")

    # 如果潜在起点太少，直接将它们加入候选起点
    if len(potential_starts) < 10:
        for acc_loc, _ in potential_starts:
            candidate_starts.append(acc_loc)
    else:
        # 对潜在起点进行详细评估
        for acc_loc, acc_capacity in tqdm(potential_starts, desc="详细评估潜在起点"):
            # 进一步评估住宿容量
            if acc_capacity >= 500:  # 进一步降低到500人
                candidate_starts.append(acc_loc)

    # 如果仍然没有找到足够的起点，继续降低标准
    if len(candidate_starts) < 5:
        print("找不到足够的起点，继续降低容量要求...")
        for acc_loc, acc_capacity in potential_starts:
            if acc_capacity >= 200 and acc_loc not in candidate_starts:  # 继续降低到200人
                candidate_starts.append(acc_loc)

    print(f"符合条件的起点数量: {len(candidate_starts)}")

    # 如果起点太多，限制数量以加快评估速度
    max_starts = 20
    if len(candidate_starts) > max_starts:
        print(f"起点数量过多，随机选择{max_starts}个进行评估")
        candidate_starts = random.sample(candidate_starts, max_starts)
    elif len(candidate_starts) == 0:
        print("警告：找不到符合条件的起点，将使用住宿设施中靠近地铁站的前10个点作为起点")
        # 找出所有靠近地铁站的住宿设施
        near_subway_accommodations = []
        for _, acc in accommodations.iterrows():
            acc_loc = None
            if 'longitude' in acc and 'latitude' in acc:
                acc_loc = [float(acc['longitude']), float(acc['latitude'])]
            elif 'gcjLng' in acc and 'gcjLat' in acc:
                acc_loc = [float(acc['gcjLng']), float(acc['gcjLat'])]
            elif 'location' in acc and isinstance(acc['location'], str):
                try:
                    coords = acc['location'].split(',')
                    if len(coords) == 2:
                        acc_loc = [float(coords[0]), float(coords[1])]
                except:
                    continue

            if not acc_loc:
                continue

            # 检查是否靠近地铁站
            for subway_loc in subway_coords:
                distance = haversine_distance(acc_loc[1], acc_loc[0], subway_loc[1], subway_loc[0])
                if distance <= 1.0:  # 1公里内
                    near_subway_accommodations.append(acc_loc)
                    break

        # 选择最多10个作为起点
        if len(near_subway_accommodations) > 0:
            candidate_starts = near_subway_accommodations[:min(10, len(near_subway_accommodations))]
            print(f"找到{len(candidate_starts)}个靠近地铁站的住宿设施作为起点")

    # 筛选可能的终点（毗邻轨道交通站点）
    candidate_ends = []
    for subway_loc in subway_coords:
        candidate_ends.append(subway_loc)

    # 如果终点太多，限制数量以加快评估速度
    max_ends = 30
    if len(candidate_ends) > max_ends:
        print(f"终点数量过多，随机选择{max_ends}个进行评估")
        candidate_ends = random.sample(candidate_ends, max_ends)

    print(f"可能的终点数量: {len(candidate_ends)}")

    # 评估所有可能的组合
    best_combination = None
    best_score = -1

    combos_count = 0
    max_combos = 50  # 限制组合数量，避免API调用过多

    for start in tqdm(candidate_starts, desc="评估起点-终点组合"):
        if combos_count >= max_combos:
            break

        for end in candidate_ends:
            if combos_count >= max_combos:
                break

            # 先计算直线距离，过滤明显不符合条件的组合
            direct_distance = haversine_distance(start[1], start[0], end[1], end[0])
            if direct_distance < 25:  # 降低要求，只要直线距离不小于25公里
                continue

            # 使用高德API计算路线
            origin_str = f"{start[0]},{start[1]}"
            destination_str = f"{end[0]},{end[1]}"
            distance, route_info = calculate_route(origin_str, destination_str, route_type="drive")

            combos_count += 1

            # 如果无法获取路线或距离小于40公里，跳过（稍微降低要求）
            if not distance or distance < 40000:
                continue

            # 评价此组合
            score = evaluate_route(start, end, route_info, accommodations, subway_stations)

            # 更新最佳组合
            if score > best_score:
                best_score = score
                best_combination = {
                    'start': start,
                    'end': end,
                    'route_info': route_info,
                    'distance': distance / 1000,  # 转换为公里
                    'score': score
                }

            # 限制API调用频率
            time.sleep(0.2)

    print(f"共评估了 {combos_count} 个起点-终点组合")
    return best_combination

# 设计闭合回路
def design_closed_loop(data, route_type="full"):
    """
    设计符合马拉松要求的闭合回路
    route_type: 路线类型，可选值: "full"(全马)、"half"(半马)、"health"(健康跑)
    """
    attractions = data['attractions']
    restaurants = data['restaurants']

    # 限制处理的数据量
    max_restaurants = 5000
    if len(restaurants) > max_restaurants:
        print(f"餐饮设施数据量过大，随机抽样{max_restaurants}条记录进行处理")
        restaurants = restaurants.sample(max_restaurants)

    max_attractions = 1000
    if len(attractions) > max_attractions:
        print(f"景点数据量过大，随机抽样{max_attractions}条记录进行处理")
        attractions = attractions.sample(max_attractions)

    # 根据路线类型确定距离要求
    if route_type == "full":
        target_distance = 42.195  # 全马 42.195公里
        print(f"正在设计全程马拉松路线 ({target_distance}公里)...")
    elif route_type == "half":
        target_distance = 21.0975  # 半马 21.0975公里
        print(f"正在设计半程马拉松路线 ({target_distance}公里)...")
    else:  # 健康跑
        target_distance = 10.0  # 健康跑 10公里
        print(f"正在设计健康跑路线 ({target_distance}公里)...")

    # 选择景点作为必经节点，根据马拉松类型选择合适数量
    if route_type == "full":
        node_count = min(8, len(attractions))
    elif route_type == "half":
        node_count = min(5, len(attractions))
    else:
        node_count = min(3, len(attractions))

    required_nodes = attractions.sample(node_count).to_dict('records')
    print(f"已选择{len(required_nodes)}个景点作为必经节点")

    # 寻找合适的起点（优先选择景点中评分最高或知名度最高的）
    start_node = None
    if 'rating' in attractions.columns:
        # 按评分选择
        start_node_candidate = attractions.nlargest(1, 'rating').to_dict('records')[0]
    else:
        # 如果没有评分，随机选择第一个景点
        start_node_candidate = required_nodes[0]

    # 确保起点毗邻地铁站
    subway_stations = data['subway_stations']
    if is_near_subway([start_node_candidate['longitude'], start_node_candidate['latitude']], subway_stations):
        start_node = start_node_candidate
    else:
        # 寻找毗邻地铁站的必经节点作为起点
        for node in required_nodes:
            if is_near_subway([node['longitude'], node['latitude']], subway_stations):
                start_node = node
                break

    # 如果仍未找到合适起点，则选择毗邻地铁站的景点
    if start_node is None:
        for _, attraction in attractions.iterrows():
            if is_near_subway([attraction['longitude'], attraction['latitude']], subway_stations):
                start_node = attraction.to_dict()
                break

    # 如果还是找不到，就用第一个必经节点
    if start_node is None:
        start_node = required_nodes[0]

    print(f"选择 {start_node.get('name', '未命名景点')} 作为起点")

    current_node = start_node
    route = [current_node]
    total_distance = 0
    total_gain = 0

    # 补给站位置
    supply_stations = []
    last_supply_distance = 0

    # 记录已使用的节点，避免重复
    used_nodes = {tuple([current_node['longitude'], current_node['latitude']])}
    remaining_required = [node for node in required_nodes if node != start_node]

    max_iterations = 20  # 限制最大迭代次数
    iterations = 0

    while (len(remaining_required) > 0 or total_distance < target_distance) and iterations < max_iterations:
        iterations += 1
        print(f"迭代 {iterations}: 当前路线长度 {total_distance:.2f}公里, 剩余必经节点 {len(remaining_required)}")

        # 决定下一个节点选择策略
        if len(remaining_required) > 0:
            # 优先选择必经节点
            candidates = remaining_required
            print("优先选择剩余的必经节点...")
        elif total_distance < target_distance:
            # 选择未使用过的餐饮设施来增加增益
            candidates = []
            # 从餐饮设施中随机选取最多10个作为候选
            rand_restaurants = restaurants.sample(min(10, len(restaurants))).to_dict('records')
            for node in rand_restaurants:
                node_coord = tuple([node['longitude'], node['latitude']])
                if node_coord not in used_nodes:
                    candidates.append(node)
            print(f"选择餐饮设施来增加路线长度，候选数量: {len(candidates)}")
        else:
            # 已经满足条件，尝试回到起点
            break

        # 如果没有候选节点，尝试返回起点
        if not candidates:
            break

        best_next_node = None
        best_route_score = float('-inf')

        # 评估每个候选节点
        for node in candidates:
            origin_str = f"{current_node['longitude']},{current_node['latitude']}"
            destination_str = f"{node['longitude']},{node['latitude']}"

            # 使用驾车模式获取路线
            segment_distance, segment_info = calculate_route(origin_str, destination_str, route_type="drive")

            if segment_distance:
                # 计算此段路线的增益（经过的餐饮设施）
                segment_gain = calculate_segment_gain(segment_info, restaurants)

                # 计算路线评分（综合考虑距离和增益）
                route_score = segment_gain * 50 - segment_distance / 1000  # 增益优先，但也要考虑距离

                # 如果这是必经节点，给予额外加分
                if node in remaining_required:
                    route_score += 200

                if best_next_node is None or route_score > best_route_score:
                    best_next_node = node
                    best_route_score = route_score
                    best_distance = segment_distance
                    best_gain = segment_gain
                    best_segment_info = segment_info

        # 如果找到了下一个节点
        if best_next_node:
            # 添加到路线
            route.append(best_next_node)
            total_distance += best_distance / 1000  # 转换为公里
            total_gain += best_gain

            # 标记为已使用
            used_nodes.add(tuple([best_next_node['longitude'], best_next_node['latitude']]))

            # 如果是必经节点，从剩余列表中移除
            if best_next_node in remaining_required:
                remaining_required.remove(best_next_node)

            # 检查是否需要设置补给站
            check_and_place_supply_stations(supply_stations, last_supply_distance, total_distance, best_segment_info, restaurants)
            last_supply_distance = total_distance

            # 更新当前节点
            current_node = best_next_node
        else:
            print("找不到下一个合适的节点，尝试返回起点")
            break

    # 完成路线后，添加回到起点的路段
    if current_node != start_node:
        print("添加返回起点的路段，形成闭合回路")
        origin_str = f"{current_node['longitude']},{current_node['latitude']}"
        destination_str = f"{start_node['longitude']},{start_node['latitude']}"

        segment_distance, segment_info = calculate_route(origin_str, destination_str, route_type="drive")

        if segment_distance:
            route.append(start_node)
            total_distance += segment_distance / 1000
            segment_gain = calculate_segment_gain(segment_info, restaurants)
            total_gain += segment_gain

            # 检查是否需要设置补给站
            check_and_place_supply_stations(supply_stations, last_supply_distance, total_distance, segment_info, restaurants)

    # 检查是否符合距离要求
    distance_diff = abs(total_distance - target_distance)
    if distance_diff > target_distance * 0.1:  # 如果偏差超过10%
        print(f"警告: 规划的路线距离({total_distance:.2f}公里)与目标距离({target_distance}公里)相差较大")

    return {
        'route': route,
        'distance': total_distance,
        'gain': total_gain,
        'supply_stations': supply_stations
    }

# 计算路段的增益值（经过的餐饮设施数量）
def calculate_segment_gain(segment_info, restaurants_data):
    """
    计算路段经过的餐饮设施增益
    每经过1个餐饮设施+0.2
    """
    # 如果餐饮设施太多，随机抽样减少计算量
    max_restaurants = 1000
    if len(restaurants_data) > max_restaurants:
        restaurants_sample = restaurants_data.sample(max_restaurants)
    else:
        restaurants_sample = restaurants_data

    gain = 0
    # 提取路段的所有坐标点
    steps = segment_info.get("steps", [])
    route_coords = []

    # 只取每隔几个点，减少计算量
    step_interval = 5
    point_count = 0

    for step in steps:
        # 提取每一步的坐标点
        polyline = step.get("polyline", "")
        if polyline:
            points = polyline.split(";")
            for point in points:
                point_count += 1
                if point_count % step_interval != 0:  # 每隔step_interval个点取一个
                    continue

                coords = point.split(",")
                if len(coords) == 2:
                    try:
                        route_coords.append([float(coords[0]), float(coords[1])])
                    except ValueError:
                        continue

    # 检查路段是否经过餐饮设施（简化为坐标点靠近餐饮设施）
    passed_restaurants = set()  # 用于记录已经计算过的餐饮设施ID

    for _, restaurant in restaurants_sample.iterrows():
        # 获取餐饮设施ID或生成唯一标识
        rest_id = str(restaurant.get('id', restaurant.name))  # 使用ID或行索引

        # 跳过已经计算过的餐饮设施
        if rest_id in passed_restaurants:
            continue

        # 获取餐饮设施坐标
        rest_loc = None

        # 尝试不同的坐标字段
        if 'longitude' in restaurant and 'latitude' in restaurant:
            rest_loc = [float(restaurant['longitude']), float(restaurant['latitude'])]
        elif 'gcjLng' in restaurant and 'gcjLat' in restaurant:
            rest_loc = [float(restaurant['gcjLng']), float(restaurant['gcjLat'])]
        elif 'location' in restaurant and isinstance(restaurant['location'], str):
            try:
                coords = restaurant['location'].split(',')
                if len(coords) == 2:
                    rest_loc = [float(coords[0]), float(coords[1])]
            except:
                continue

        # 如果无法获取坐标，跳过此餐饮设施
        if not rest_loc:
            continue

        # 检查路线是否经过此餐饮设施
        for coord in route_coords:
            try:
                distance = haversine_distance(coord[1], coord[0], rest_loc[1], rest_loc[0])
                if distance <= 0.1:  # 假设100米内视为经过
                    gain += 0.2
                    passed_restaurants.add(rest_id)
                    break  # 每个餐饮设施只计算一次
            except Exception as e:
                # print(f"计算餐饮设施增益时出错: {e}")
                continue

    return gain

# 检查并设置补给站
def check_and_place_supply_stations(supply_stations, last_distance, current_distance, segment_info, restaurants_data):
    """
    检查是否需要设置补给站，确保每5公里设置一个补给站
    并确保补给站邻近餐饮设施
    """
    # 计算从上一个补给站到当前位置的距离区间
    start_km = int(last_distance / 5) * 5 + 5
    end_km = int(current_distance / 5) * 5 + 5

    # 如果区间内有5的倍数公里点，则需要设置补给站
    for km in range(start_km, end_km + 1, 5):
        if km <= current_distance and km > last_distance:
            # 在路线上找到最接近此公里数的点
            target_point = find_point_at_distance(segment_info, km - last_distance)

            if target_point:
                # 找到邻近的餐饮设施
                nearby_restaurant = find_nearest_restaurant(target_point, restaurants_data)

                if nearby_restaurant:
                    supply_stations.append({
                        'distance': km,
                        'location': target_point,
                        'nearby_restaurant': nearby_restaurant
                    })

# 在路线上找到特定距离的点
def find_point_at_distance(segment_info, target_distance_km):
    """
    在路段上找到距离起点特定公里数的点
    """
    target_distance_m = target_distance_km * 1000
    accumulated_distance = 0

    steps = segment_info.get("steps", [])
    last_point = None

    for step in steps:
        step_distance = float(step.get("distance", 0))

        if accumulated_distance + step_distance >= target_distance_m:
            # 目标点在当前step内
            polyline = step.get("polyline", "")
            points = polyline.split(";")

            # 简化处理：假设step内距离均匀分布
            point_index = int(len(points) * (target_distance_m - accumulated_distance) / step_distance)
            point_index = min(point_index, len(points) - 1)

            coords = points[point_index].split(",")
            if len(coords) == 2:
                return [float(coords[0]), float(coords[1])]

        accumulated_distance += step_distance

        # 记录最后一个点作为备用
        if step.get("polyline"):
            last_points = step["polyline"].split(";")
            if last_points:
                last_coords = last_points[-1].split(",")
                if len(last_coords) == 2:
                    last_point = [float(last_coords[0]), float(last_coords[1])]

    # 如果没有找到精确的点，返回最后一个点
    return last_point

# 找到最近的餐饮设施
def find_nearest_restaurant(location, restaurants_data):
    """
    找到距离给定位置最近的餐饮设施
    """
    nearest_restaurant = None
    min_distance = float('inf')

    for _, restaurant in restaurants_data.iterrows():
        # 获取餐饮设施位置坐标
        rest_loc = None

        # 尝试不同的列名格式获取坐标
        if 'longitude' in restaurant and 'latitude' in restaurant:
            rest_loc = [restaurant['longitude'], restaurant['latitude']]
        elif 'gcjLng' in restaurant and 'gcjLat' in restaurant:
            rest_loc = [restaurant['gcjLng'], restaurant['gcjLat']]
        elif 'location' in restaurant and isinstance(restaurant['location'], str):
            # 尝试从location字符串解析坐标
            try:
                lng_lat = restaurant['location'].split(',')
                if len(lng_lat) == 2:
                    rest_loc = [float(lng_lat[0]), float(lng_lat[1])]
            except:
                # 解析失败，跳过此餐饮设施
                continue

        # 如果无法获取坐标，跳过此餐饮设施
        if not rest_loc:
            continue

        # 计算距离
        try:
            distance = haversine_distance(location[1], location[0], rest_loc[1], rest_loc[0])

            if distance < min_distance:
                min_distance = distance

                # 获取餐饮设施名称和ID
                name = restaurant.get('name', '未命名餐饮设施')
                rest_id = restaurant.get('id', f"{rest_loc[0]}_{rest_loc[1]}")

                nearest_restaurant = {
                    'id': rest_id,
                    'name': name,
                    'location': rest_loc,
                    'distance': distance
                }
        except Exception as e:
            # 计算距离时出错，跳过此餐饮设施
            continue

    return nearest_restaurant

# 主函数
def main():
    # 创建输出目录
    output_dir = "NMMCM-2025/P2"
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("正在加载数据...")
    data = load_data()
    if not data:
        print("数据加载失败，请确保数据文件路径正确")
        return

    # 检查数据有效性
    valid_data = True
    if len(data['accommodations']) == 0:
        print("警告: 住宿设施数据为空")
        valid_data = False
    if len(data['attractions']) == 0:
        print("警告: 景点数据为空")
        valid_data = False
    if len(data['restaurants']) == 0:
        print("警告: 餐饮设施数据为空")
        valid_data = False
    if len(data['subway_stations']) == 0:
        print("警告: 地铁站数据为空")
        valid_data = False

    if not valid_data:
        print("数据无效，请检查数据文件")
        return

    # 选择要计算的内容
    print("\n请选择要计算的内容:")
    print("1. 最优起点-终点组合")
    print("2. 全马闭合回路")
    print("3. 半马闭合回路")
    print("4. 健康跑闭合回路")
    print("5. 所有内容")
    choice = input("请输入选项(1-5): ")

    optimal_combination = None
    closed_loop_full = None
    closed_loop_half = None
    closed_loop_health = None

    # 根据用户选择执行相应计算
    if choice in ["1", "5"]:
        # 找出最优起点-终点组合
        print("\n正在寻找最优起点-终点组合...")
        optimal_combination = find_optimal_start_end_points(data)

        if optimal_combination:
            print("\n最优起点-终点组合:")
            print(f"起点: {optimal_combination['start']}")
            print(f"终点: {optimal_combination['end']}")
            print(f"距离: {optimal_combination['distance']:.2f}公里")
            print(f"评分: {optimal_combination['score']:.2f}")
        else:
            print("未找到满足条件的起点-终点组合")

    if choice in ["2", "5"]:
        # 设计全马闭合回路
        print("\n正在设计闭合回路...")
        closed_loop_full = design_closed_loop(data, "full")

        if closed_loop_full:
            print("\n全马闭合回路:")
            print(f"总距离: {closed_loop_full['distance']:.2f}公里")
            print(f"总增益: {closed_loop_full['gain']:.2f}")
            print(f"节点数量: {len(closed_loop_full['route'])}")
            print(f"补给站数量: {len(closed_loop_full['supply_stations'])}")

    if choice in ["3", "5"]:
        # 设计半马闭合回路
        closed_loop_half = design_closed_loop(data, "half")

        if closed_loop_half:
            print("\n半马闭合回路:")
            print(f"总距离: {closed_loop_half['distance']:.2f}公里")
            print(f"总增益: {closed_loop_half['gain']:.2f}")
            print(f"节点数量: {len(closed_loop_half['route'])}")
            print(f"补给站数量: {len(closed_loop_half['supply_stations'])}")

    if choice in ["4", "5"]:
        # 设计健康跑闭合回路
        closed_loop_health = design_closed_loop(data, "health")

        if closed_loop_health:
            print("\n健康跑闭合回路:")
            print(f"总距离: {closed_loop_health['distance']:.2f}公里")
            print(f"总增益: {closed_loop_health['gain']:.2f}")
            print(f"节点数量: {len(closed_loop_health['route'])}")
            print(f"补给站数量: {len(closed_loop_health['supply_stations'])}")

    # 可视化结果
    print("\n正在生成可视化结果...")
    visualize_results(optimal_combination, closed_loop_full, closed_loop_half, closed_loop_health)

    print("\n程序运行完成!")

# 可视化结果
def visualize_results(optimal_combination, closed_loop_full, closed_loop_half, closed_loop_health):
    """
    可视化最优路线和闭合回路
    """
    # 如果所有结果都为None，则不生成可视化
    if not optimal_combination and not closed_loop_full and not closed_loop_half and not closed_loop_health:
        print("没有可视化的结果")
        return

    # 创建西安市中心的地图
    xian_center = [34.2655, 108.9541]  # 西安市中心坐标
    m = folium.Map(location=xian_center, zoom_start=11)

    # 可视化最优起点-终点组合
    if optimal_combination:
        # 添加起点
        start = optimal_combination['start']
        folium.Marker(
            location=[start[1], start[0]],
            popup='起点',
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)

        # 添加终点
        end = optimal_combination['end']
        folium.Marker(
            location=[end[1], end[0]],
            popup='终点',
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(m)

        # 添加路线
        if 'route_info' in optimal_combination and optimal_combination['route_info'].get('steps'):
            route_points = []
            for step in optimal_combination['route_info']['steps']:
                polyline = step.get('polyline', '')
                points = polyline.split(';')
                for point in points:
                    coords = point.split(',')
                    if len(coords) == 2:
                        route_points.append([float(coords[1]), float(coords[0])])

            folium.PolyLine(
                route_points,
                color='blue',
                weight=3,
                opacity=0.8,
                popup=f"距离: {optimal_combination['distance']:.2f}公里"
            ).add_to(m)

    # 可视化闭合回路
    routes = {
        'full': closed_loop_full,
        'half': closed_loop_half,
        'health': closed_loop_health
    }

    colors = {
        'full': 'red',
        'half': 'purple',
        'health': 'orange'
    }

    for route_type, route_data in routes.items():
        if route_data:
            route_points = []
            for i in range(len(route_data['route']) - 1):
                start_node = route_data['route'][i]
                end_node = route_data['route'][i + 1]

                # 获取起点坐标
                start_loc = None
                if 'longitude' in start_node and 'latitude' in start_node:
                    start_loc = [float(start_node['longitude']), float(start_node['latitude'])]
                elif 'gcjLng' in start_node and 'gcjLat' in start_node:
                    start_loc = [float(start_node['gcjLng']), float(start_node['gcjLat'])]
                else:
                    continue

                # 获取终点坐标
                end_loc = None
                if 'longitude' in end_node and 'latitude' in end_node:
                    end_loc = [float(end_node['longitude']), float(end_node['latitude'])]
                elif 'gcjLng' in end_node and 'gcjLat' in end_node:
                    end_loc = [float(end_node['gcjLng']), float(end_node['gcjLat'])]
                else:
                    continue

                # 添加节点标记
                folium.CircleMarker(
                    location=[start_loc[1], start_loc[0]],
                    radius=5,
                    color=colors[route_type],
                    fill=True,
                    fill_color=colors[route_type],
                    popup=start_node.get('name', f'节点{i}')
                ).add_to(m)

                # 添加路段
                # 此处可以添加起点到终点的路线，但由于我们没有具体的路线数据，只用直线连接
                folium.PolyLine(
                    [[start_loc[1], start_loc[0]], [end_loc[1], end_loc[0]]],
                    color=colors[route_type],
                    weight=2,
                    opacity=0.8
                ).add_to(m)

            # 添加最后一个节点
            last_node = route_data['route'][-1]

            # 获取最后节点坐标
            last_loc = None
            if 'longitude' in last_node and 'latitude' in last_node:
                last_loc = [float(last_node['longitude']), float(last_node['latitude'])]
            elif 'gcjLng' in last_node and 'gcjLat' in last_node:
                last_loc = [float(last_node['gcjLng']), float(last_node['gcjLat'])]

            if last_loc:
                folium.CircleMarker(
                    location=[last_loc[1], last_loc[0]],
                    radius=5,
                    color=colors[route_type],
                    fill=True,
                    fill_color=colors[route_type],
                    popup=last_node.get('name', f'节点{len(route_data["route"]) - 1}')
                ).add_to(m)

            # 添加补给站
            for station in route_data['supply_stations']:
                if 'location' in station and station['location']:
                    folium.Marker(
                        location=[station['location'][1], station['location'][0]],
                        popup=f"补给站({station['distance']}公里)",
                        icon=folium.Icon(color='blue', icon='tint', prefix='fa')
                    ).add_to(m)

    # 确保目录存在
    os.makedirs("./P2", exist_ok=True)

    # 保存地图
    map_path = './P2/marathon_routes.html'
    m.save(map_path)
    print(f"可视化结果已保存到 {map_path}")

if __name__ == "__main__":
    main()