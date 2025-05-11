import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
from math import radians, sin, cos, sqrt, atan2
import folium
from folium.plugins import MarkerCluster
import rasterio
from rasterio import windows
from rasterio.plot import show
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping
import pyproj
from pyproj import Transformer
import requests
import json
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution, NonlinearConstraint
import nsga2
from functools import partial
from datetime import datetime

# 设置环境变量，降低内存占用
os.environ["OMP_NUM_THREADS"] = "2"  # OpenMP线程
os.environ["OPENBLAS_NUM_THREADS"] = "2"  # OpenBLAS线程
os.environ["MKL_NUM_THREADS"] = "2"  # MKL线程
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"  # Accelerate线程
os.environ["NUMEXPR_NUM_THREADS"] = "2"  # Numexpr线程
os.environ["GDAL_CACHEMAX"] = "256"  # 设置GDAL缓存大小为256MB

# 高德地图API密钥
API_KEY = "8ca1e4d717a1f7c095f78b2a127c96ea"  # 请替换为您的实际密钥

# 数据路径
ACCOMMODATION_PATH = "processed_data/cleaned/附件5_xian_hotels_cleaned.csv"
ATTRACTIONS_PATH = "processed_data/cleaned/附件5_xian_attractions_cleaned.csv"
RESTAURANTS_PATH = "processed_data/cleaned/附件5_xian_restaurants_cleaned.csv"
SUBWAY_STATIONS_PATH = "processed_data/附件9_xian_subway_stations.csv"
ROAD_PATH = "processed_data/附件7_xian_road_connections.csv"
GREEN_SPACE_PATH = "附件/附件10：西安市的绿地数据/xian.tif"
TERRAIN_PATH = "附件/附件6：陕西省12.5分辨率地形数据/陕西WGS84.tif"

# 创建输出目录
OUTPUT_DIR = "./P3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 从P2.py导入的实用函数
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

def calculate_route(origin, destination, route_type="walk", max_retries=3, retry_delay=1):
    """
    使用高德路径规划API计算两点之间的路线
    origin: 起点坐标，格式为"lng,lat"
    destination: 终点坐标，格式为"lng,lat"
    route_type: 路线类型，可选值: "walk"(步行), "drive"(驾车)
    max_retries: 最大重试次数
    retry_delay: 重试间隔时间（秒）
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

    for attempt in range(max_retries):
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
            else:
                error_info = result.get("info", "未知错误")
                # 如果是配额超限问题，等待更长时间后重试
                if "EXCEED" in error_info or "LIMIT" in error_info:
                    print(f"API限额超出: {error_info}，等待重试 ({attempt+1}/{max_retries})")
                    time.sleep(retry_delay * (attempt + 1))  # 逐次增加等待时间
                    continue
                else:
                    print(f"路径规划失败: {error_info}")
        except Exception as e:
            print(f"路径规划请求失败: {e}")

        # 如果不是配额问题导致的失败，短暂等待后重试
        time.sleep(retry_delay)

    # 所有重试都失败，返回None
    return None, None

# 加载和处理数据
def load_data():
    """加载西安市的基础数据"""
    try:
        # 加载住宿设施数据
        accommodations = pd.read_csv(ACCOMMODATION_PATH)
        print(f"住宿设施数据加载成功，共 {len(accommodations)} 条记录")

        # 确保住宿数据有经纬度列
        if 'longitude' not in accommodations.columns and 'gcjLng' in accommodations.columns:
            accommodations['longitude'] = accommodations['gcjLng']
        if 'latitude' not in accommodations.columns and 'gcjLat' in accommodations.columns:
            accommodations['latitude'] = accommodations['gcjLat']

        # 加载景点数据
        attractions = pd.read_csv(ATTRACTIONS_PATH)
        print(f"景点数据加载成功，共 {len(attractions)} 条记录")

        # 确保景点数据有经纬度列
        if 'longitude' not in attractions.columns and 'gcjLng' in attractions.columns:
            attractions['longitude'] = attractions['gcjLng']
        if 'latitude' not in attractions.columns and 'gcjLat' in attractions.columns:
            attractions['latitude'] = attractions['gcjLat']

        # 加载餐饮设施数据
        restaurants = pd.read_csv(RESTAURANTS_PATH)
        print(f"餐饮设施数据加载成功，共 {len(restaurants)} 条记录")

        # 确保餐饮数据有经纬度列
        if 'longitude' not in restaurants.columns and 'gcjLng' in restaurants.columns:
            restaurants['longitude'] = restaurants['gcjLng']
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

# 优化绿地数据加载函数
def load_green_space_data():
    """
    优化后的西安市绿地数据加载函数
    仅加载西安市中心区域数据，使用更小的内存窗口
    """
    try:
        # 西安市的地理范围（经纬度）- 仅加载中心区域
        xian_bounds = {
            'left': 108.9,    # 最小经度
            'bottom': 34.2,   # 最小纬度
            'right': 109.0,   # 最大经度
            'top': 34.3       # 最大纬度
        }

        print(f"正在加载西安市中心区域的绿地数据...")

        # 打开栅格数据文件，但不立即加载所有数据
        with rasterio.open(GREEN_SPACE_PATH) as src:
            print(f"绿地数据元信息: 宽度={src.width}, 高度={src.height}, 波段数={src.count}")

            try:
                # 计算西安市中心区域对应的像素窗口
                window = windows.from_bounds(
                    xian_bounds['left'],
                    xian_bounds['bottom'],
                    xian_bounds['right'],
                    xian_bounds['top'],
                    src.transform
                )

                # 将窗口取整，确保是有效的像素窗口
                window = windows.Window(
                    int(window.col_off),
                    int(window.row_off),
                    int(window.width),
                    int(window.height)
                )

                print(f"读取窗口大小: {window.width}x{window.height} 像素")

                # 如果窗口太大，进一步缩小
                max_size = 1000  # 最大窗口大小
                if window.width > max_size or window.height > max_size:
                    scale = max(window.width / max_size, window.height / max_size)
                    new_width = int(window.width / scale)
                    new_height = int(window.height / scale)

                    # 缩小窗口，保持中心不变
                    col_off = window.col_off + (window.width - new_width) // 2
                    row_off = window.row_off + (window.height - new_height) // 2

                    window = windows.Window(
                        int(col_off),
                        int(row_off),
                        int(new_width),
                        int(new_height)
                    )
                    print(f"缩小窗口到: {window.width}x{window.height} 像素")

                # 读取窗口内的数据
                data = src.read(1, window=window)

                # 获取对应区域的变换参数
                transform = windows.transform(window, src.transform)

                print(f"绿地数据加载完成，数据形状: {data.shape}")

                # 返回处理过的数据
                return {
                    'data': data,
                    'meta': src.meta.copy(),
                    'bounds': type('obj', (object,), xian_bounds),
                    'transform': transform,
                    'window': window
                }

            except Exception as e:
                print(f"绿地数据窗口处理错误: {e}")
                # 如果精确窗口失败，尝试读取一个固定大小的中心区域
                center_x = src.width // 2
                center_y = src.height // 2
                size = 500  # 读取500x500像素的区域

                window = windows.Window(
                    center_x - size // 2,
                    center_y - size // 2,
                    size,
                    size
                )

                data = src.read(1, window=window)
                transform = windows.transform(window, src.transform)

                print(f"使用中心区域代替，数据形状: {data.shape}")

                return {
                    'data': data,
                    'meta': src.meta.copy(),
                    'bounds': type('obj', (object,), xian_bounds),
                    'transform': transform,
                    'window': window
                }

    except Exception as e:
        print(f"绿地数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        # 返回一个随机数据作为替代
        print("使用随机数据代替绿地数据")
        return {
            'data': np.random.randint(0, 2, size=(500, 500)),
            'meta': {'crs': 'EPSG:4326'},
            'bounds': type('obj', (object,), {
                'left': 108.9, 'right': 109.0,
                'bottom': 34.2, 'top': 34.3
            }),
            'transform': None,
            'window': None
        }

# 优化地形数据加载函数
def load_terrain_data():
    """
    优化后的陕西省地形数据加载函数
    仅加载西安市中心区域数据，使用更小的内存窗口
    """
    try:
        # 西安市的地理范围（经纬度）- 仅加载中心区域
        xian_bounds = {
            'left': 108.9,    # 最小经度
            'bottom': 34.2,   # 最小纬度
            'right': 109.0,   # 最大经度
            'top': 34.3       # 最大纬度
        }

        print(f"正在加载西安市中心区域的地形数据...")

        # 打开栅格数据文件，但不立即加载所有数据
        with rasterio.open(TERRAIN_PATH) as src:
            print(f"地形数据元信息: 宽度={src.width}, 高度={src.height}, 波段数={src.count}")

            try:
                # 计算西安市中心区域对应的像素窗口
                window = windows.from_bounds(
                    xian_bounds['left'],
                    xian_bounds['bottom'],
                    xian_bounds['right'],
                    xian_bounds['top'],
                    src.transform
                )

                # 将窗口取整，确保是有效的像素窗口
                window = windows.Window(
                    int(window.col_off),
                    int(window.row_off),
                    int(window.width),
                    int(window.height)
                )

                print(f"读取窗口大小: {window.width}x{window.height} 像素")

                # 如果窗口太大，进一步缩小
                max_size = 1000  # 最大窗口大小
                if window.width > max_size or window.height > max_size:
                    scale = max(window.width / max_size, window.height / max_size)
                    new_width = int(window.width / scale)
                    new_height = int(window.height / scale)

                    # 缩小窗口，保持中心不变
                    col_off = window.col_off + (window.width - new_width) // 2
                    row_off = window.row_off + (window.height - new_height) // 2

                    window = windows.Window(
                        int(col_off),
                        int(row_off),
                        int(new_width),
                        int(new_height)
                    )
                    print(f"缩小窗口到: {window.width}x{window.height} 像素")

                # 读取窗口内的数据
                data = src.read(1, window=window)

                # 获取对应区域的变换参数
                transform = windows.transform(window, src.transform)

                print(f"地形数据加载完成，数据形状: {data.shape}")

                # 返回处理过的数据
                return {
                    'data': data,
                    'meta': src.meta.copy(),
                    'bounds': type('obj', (object,), xian_bounds),
                    'transform': transform,
                    'window': window
                }

            except Exception as e:
                print(f"地形数据窗口处理错误: {e}")
                # 如果精确窗口失败，尝试读取一个固定大小的中心区域
                center_x = src.width // 2
                center_y = src.height // 2
                size = 500  # 读取500x500像素的区域

                window = windows.Window(
                    center_x - size // 2,
                    center_y - size // 2,
                    size,
                    size
                )

                data = src.read(1, window=window)
                transform = windows.transform(window, src.transform)

                print(f"使用中心区域代替，数据形状: {data.shape}")

                return {
                    'data': data,
                    'meta': src.meta.copy(),
                    'bounds': type('obj', (object,), xian_bounds),
                    'transform': transform,
                    'window': window
                }

    except Exception as e:
        print(f"地形数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        # 返回一个随机数据作为替代
        print("使用随机数据代替地形数据")
        return {
            'data': np.random.randint(350, 450, size=(500, 500)),
            'meta': {'crs': 'EPSG:4326', 'nodata': -9999},
            'bounds': type('obj', (object,), {
                'left': 108.9, 'right': 109.0,
                'bottom': 34.2, 'top': 34.3
            }),
            'transform': None,
            'window': None
        }

# 优化树荫覆盖率计算函数
def calculate_green_coverage(route_points, green_space_data):
    """
    优化后的树荫覆盖率计算函数
    使用抽样方式减少计算量，提高性能

    route_points: 路线上的点集合 [(lng1, lat1), (lng2, lat2), ...]
    green_space_data: 绿地数据
    返回树荫覆盖率（0-1之间的值）
    """
    if not route_points or not green_space_data or 'data' not in green_space_data:
        return 0.0

    # 获取绿地数据参数
    transform = green_space_data.get('transform')
    data = green_space_data.get('data')
    bounds = green_space_data.get('bounds')

    if transform is None or data is None or bounds is None:
        return 0.0

    # 计算路线上点的树荫覆盖情况
    covered_points = 0
    total_points = len(route_points)

    # 对路线进行抽样分析，减少计算量
    # 对于短路线(<1000点)，每个点都检查；对于长路线，抽样检查
    sample_interval = max(1, total_points // 500)  # 最多检查500个点
    valid_samples = 0

    for i in range(0, total_points, sample_interval):
        if i >= len(route_points):
            break

        point = route_points[i]
        lng, lat = point

        # 检查点是否在绿地数据的边界内
        if not (bounds.left <= lng <= bounds.right and bounds.bottom <= lat <= bounds.top):
            continue

        try:
            # 计算在栅格中的行列索引
            col, row = ~transform * (lng, lat)

            # 转为整数索引
            row, col = int(row), int(col)

            # 检查索引是否在有效范围内
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                valid_samples += 1

                # 检查是否有树荫覆盖（值大于0的区域为绿地）
                if data[row, col] > 0:
                    covered_points += 1
        except Exception as e:
            # 忽略计算错误，继续处理下一个点
            continue

    # 计算覆盖率
    if valid_samples > 0:
        coverage_rate = covered_points / valid_samples
    else:
        coverage_rate = 0.0

    return coverage_rate

# 优化路线坡度计算函数
def calculate_slope(route_points, terrain_data):
    """
    优化后的路线坡度计算函数
    使用抽样方式减少计算量，提高性能

    route_points: 路线上的点集合 [(lng1, lat1), (lng2, lat2), ...]
    terrain_data: 地形数据
    返回最大坡度和平均坡度
    """
    if not route_points or not terrain_data or 'data' not in terrain_data or len(route_points) < 2:
        return 0.0, 0.0

    # 获取地形数据参数
    transform = terrain_data.get('transform')
    data = terrain_data.get('data')
    bounds = terrain_data.get('bounds')
    nodata_value = terrain_data.get('meta', {}).get('nodata', -9999)

    if transform is None or data is None or bounds is None:
        return 0.0, 0.0

    # 对路线进行抽样分析，减少计算量
    total_points = len(route_points)
    sample_interval = max(1, total_points // 200)  # 最多采样200个点

    # 采样的点和对应的高程值
    sampled_points = []
    elevations = []

    for i in range(0, total_points, sample_interval):
        if i >= len(route_points):
            break

        point = route_points[i]
        lng, lat = point

        # 检查点是否在地形数据的边界内
        if not (bounds.left <= lng <= bounds.right and bounds.bottom <= lat <= bounds.top):
            # 边界外使用默认高程
            sampled_points.append(point)
            elevations.append(400)  # 西安市平均海拔约400米
            continue

        try:
            # 计算在栅格中的行列索引
            col, row = ~transform * (lng, lat)

            # 转为整数索引
            row, col = int(row), int(col)

            # 检查索引是否在有效范围内
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                elevation = data[row, col]

                # 检查是否是有效高程值
                if elevation != nodata_value:
                    sampled_points.append(point)
                    elevations.append(float(elevation))
                else:
                    # 无效高程，使用默认值
                    sampled_points.append(point)
                    elevations.append(400)
            else:
                # 索引超出范围，使用默认值
                sampled_points.append(point)
                elevations.append(400)
        except Exception as e:
            # 忽略计算错误，使用默认值
            sampled_points.append(point)
            elevations.append(400)

    # 计算相邻点之间的坡度
    slopes = []
    for i in range(len(sampled_points) - 1):
        point1 = sampled_points[i]
        point2 = sampled_points[i + 1]

        # 计算两点间的水平距离（米）
        distance = haversine_distance(point1[1], point1[0], point2[1], point2[0]) * 1000

        # 计算高度差（米）
        height_diff = abs(elevations[i + 1] - elevations[i])

        # 计算坡度（百分比）
        if distance > 0:
            slope = (height_diff / distance) * 100
            slopes.append(slope)

    # 计算最大坡度和平均坡度
    if slopes:
        max_slope = max(slopes)
        avg_slope = sum(slopes) / len(slopes)
    else:
        max_slope = 0.0
        avg_slope = 0.0

    return max_slope, avg_slope

# 评估路线对交通的影响
def evaluate_traffic_impact(route_points, road_connections, time_period="morning_peak"):
    """
    评估路线对城市交通的影响
    route_points: 路线上的点集合 [(lng1, lat1), (lng2, lat2), ...]
    road_connections: 道路连接数据
    time_period: 时间段，可选值: "morning_peak"(早高峰), "evening_peak"(晚高峰), "normal"(平时)
    返回交通影响指数（越低越好）
    """
    if not route_points or road_connections is None or len(road_connections) == 0:
        return 0.0

    # 样本点数量限制，避免过度计算
    max_sample_points = 300

    # 对路线进行抽样
    if len(route_points) > max_sample_points:
        sample_interval = len(route_points) // max_sample_points
        sampled_route_points = [route_points[i] for i in range(0, len(route_points), sample_interval)]
    else:
        sampled_route_points = route_points

    # 将道路连接数据转换为坐标点
    road_points = []
    road_weights = []  # 道路权重，表示交通流量/重要性

    # 检查road_connections的列信息
    required_columns = ['start_lng', 'start_lat', 'end_lng', 'end_lat']

    # 如果道路数据不包含所需列，尝试不同的列名
    if not all(col in road_connections.columns for col in required_columns):
        # 尝试其他可能的列名格式
        if 'startLng' in road_connections.columns:
            road_connections['start_lng'] = road_connections['startLng']
            road_connections['start_lat'] = road_connections['startLat']
            road_connections['end_lng'] = road_connections['endLng']
            road_connections['end_lat'] = road_connections['endLat']
        elif 'from_lng' in road_connections.columns:
            road_connections['start_lng'] = road_connections['from_lng']
            road_connections['start_lat'] = road_connections['from_lat']
            road_connections['end_lng'] = road_connections['to_lng']
            road_connections['end_lat'] = road_connections['to_lat']

    # 仅使用部分道路数据，避免内存过载
    sample_size = min(5000, len(road_connections))
    sampled_roads = road_connections.sample(sample_size) if len(road_connections) > sample_size else road_connections

    # 提取道路坐标和权重
    for _, road in sampled_roads.iterrows():
        try:
            start_point = (float(road['start_lng']), float(road['start_lat']))
            end_point = (float(road['end_lng']), float(road['end_lat']))

            road_points.append(start_point)
            road_points.append(end_point)

            # 根据道路等级设置权重
            weight = 1.0
            if 'road_class' in road:
                road_class = str(road['road_class']).lower()
                if '主干道' in road_class or 'primary' in road_class:
                    weight = 3.0
                elif '次干道' in road_class or 'secondary' in road_class:
                    weight = 2.0
                elif '快速路' in road_class or 'expressway' in road_class:
                    weight = 4.0

            # 根据时间段调整权重
            if time_period == "morning_peak":
                weight *= 1.5
            elif time_period == "evening_peak":
                weight *= 1.3

            road_weights.append(weight)
            road_weights.append(weight)
        except:
            continue

    # 如果没有有效的道路数据，返回0
    if not road_points:
        return 0.0

    # 创建KD树加速最近邻搜索
    road_points_array = np.array(road_points)
    road_kdtree = cKDTree(road_points_array)

    # 计算路线上每个点到最近道路的距离，以及该道路的权重
    traffic_impact = 0.0
    count = 0

    for point in sampled_route_points:
        # 计算点到最近道路的距离
        try:
            distance, idx = road_kdtree.query([point[0], point[1]], k=1)

            # 如果距离很小，表示路线经过该道路，增加交通影响
            if distance < 0.001:  # 约100米
                traffic_impact += road_weights[idx]
                count += 1
        except:
            continue

    # 归一化交通影响值
    if count > 0:
        normalized_impact = traffic_impact / count
    else:
        normalized_impact = 0.0

    return normalized_impact

# 优化后的可视化函数
def visualize_routes(candidate_routes, data, green_space_data, terrain_data):
    """
    可视化多个候选路线
    candidate_routes: 候选路线列表
    data: 基础数据
    green_space_data: 绿地数据
    terrain_data: 地形数据
    """
    # 创建西安市中心的地图
    xian_center = [34.2655, 108.9541]  # 西安市中心坐标
    m = folium.Map(location=xian_center, zoom_start=12, tiles='OpenStreetMap')

    # 为每个候选路线设置不同颜色
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    # 可视化每个候选路线
    for i, route in enumerate(candidate_routes):
        color = colors[i % len(colors)]
        route_name = f"路线 {i+1}"

        # 创建路线信息
        popup_text = f"路线 {i+1}<br>"
        popup_text += f"树荫覆盖率: {route['green_coverage']:.2f}<br>"
        popup_text += f"最大坡度: {route['max_slope']:.2f}%<br>"
        popup_text += f"平均坡度: {route['avg_slope']:.2f}%<br>"
        popup_text += f"交通影响: {route['traffic_impact']:.2f}<br>"
        popup_text += f"总距离: {route['total_distance']:.2f}公里"

        # 添加路线
        if 'route_points' in route and route['route_points']:
            route_latlngs = [[p[1], p[0]] for p in route['route_points']]

            folium.PolyLine(
                route_latlngs,
                color=color,
                weight=4,
                opacity=0.8,
                popup=popup_text,
                tooltip=route_name
            ).add_to(m)

        # 添加途经点标记
        for j, point in enumerate(route['waypoints']):
            marker_text = f"路线 {i+1} - 点 {j}"
            if j == 0:
                marker_text = f"路线 {i+1} - 起点"
                folium.Marker(
                    location=[point[1], point[0]],
                    icon=folium.Icon(color=color, icon='play', prefix='fa'),
                    popup=marker_text
                ).add_to(m)
            elif j == len(route['waypoints']) - 1:
                marker_text = f"路线 {i+1} - 终点"
                folium.Marker(
                    location=[point[1], point[0]],
                    icon=folium.Icon(color=color, icon='stop', prefix='fa'),
                    popup=marker_text
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[point[1], point[0]],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=marker_text
                ).add_to(m)

    # 添加地铁站点作为参考 - 分组减少视觉干扰
    if 'subway_stations' in data and not data['subway_stations'].empty:
        subway_group = folium.FeatureGroup(name="地铁站", show=False)
        for _, station in data['subway_stations'].iterrows():
            station_loc = None

            # 尝试不同的列名获取地铁站坐标
            if 'longitude' in station and 'latitude' in station:
                station_loc = [float(station['latitude']), float(station['longitude'])]
            elif 'gcjLng' in station and 'gcjLat' in station:
                station_loc = [float(station['gcjLat']), float(station['gcjLng'])]
            elif '经度' in station and '纬度' in station:
                station_loc = [float(station['纬度']), float(station['经度'])]

            if station_loc:
                station_name = station.get('name', '地铁站')
                folium.CircleMarker(
                    location=station_loc,
                    radius=3,
                    color='black',
                    fill=True,
                    fill_color='black',
                    popup=station_name,
                    tooltip=station_name
                ).add_to(subway_group)

        subway_group.add_to(m)

    # 添加图层控制
    folium.LayerControl().add_to(m)

    # 保存地图
    output_path = os.path.join(OUTPUT_DIR, "optimized_routes.html")
    m.save(output_path)
    print(f"可视化结果已保存到 {output_path}")

    return output_path

# 生成评估报告
def generate_report(candidate_routes):
    """
    生成路线评估报告
    candidate_routes: 候选路线列表
    """
    report = "# 马拉松路线综合评估报告\n\n"
    report += "## 评估指标\n\n"
    report += "1. **树荫覆盖率**: 路线上有树荫覆盖的比例，越高越好\n"
    report += "2. **坡度控制**: 路线的最大坡度和平均坡度，要求最大坡度 ≤ 5%\n"
    report += "3. **交通影响**: 路线对城市交通的干扰程度，越低越好\n"
    report += "4. **路线长度**: 路线总长度，应符合马拉松标准\n\n"

    report += "## 候选路线评估\n\n"

    # 创建评估表格
    report += "| 路线编号 | 树荫覆盖率 | 最大坡度 | 平均坡度 | 交通影响 | 总距离(公里) | 综合评分 |\n"
    report += "| -------- | ---------- | -------- | -------- | -------- | ------------ | -------- |\n"

    for i, route in enumerate(candidate_routes):
        # 计算综合评分
        # 树荫覆盖率（越高越好）
        green_score = route['green_coverage'] * 10
        # 坡度控制（最大坡度不超过5%）
        slope_score = max(0, 5 - route['max_slope']) * 2
        # 交通影响（越低越好）
        traffic_score = max(0, 5 - route['traffic_impact']) * 2
        # 距离偏差（与标准马拉松距离42.195公里的接近程度）
        distance_score = max(0, 5 - abs(route['total_distance'] - 42.195) / 2)

        # 综合评分
        total_score = green_score + slope_score + traffic_score + distance_score

        # 添加到表格
        report += f"| {i+1} | {route['green_coverage']:.2f} | {route['max_slope']:.2f}% | "
        report += f"{route['avg_slope']:.2f}% | {route['traffic_impact']:.2f} | "
        report += f"{route['total_distance']:.2f} | {total_score:.2f} |\n"

    report += "\n## 最优路线分析\n\n"

    # 找出综合评分最高的路线
    best_route_idx = 0
    best_score = 0

    for i, route in enumerate(candidate_routes):
        green_score = route['green_coverage'] * 10
        slope_score = max(0, 5 - route['max_slope']) * 2
        traffic_score = max(0, 5 - route['traffic_impact']) * 2
        distance_score = max(0, 5 - abs(route['total_distance'] - 42.195) / 2)
        total_score = green_score + slope_score + traffic_score + distance_score

        if total_score > best_score:
            best_score = total_score
            best_route_idx = i

    if candidate_routes:
        best_route = candidate_routes[best_route_idx]

        report += f"**最优路线编号**: {best_route_idx + 1}\n\n"
        report += f"**树荫覆盖率**: {best_route['green_coverage']:.2f}\n"
        report += f"**最大坡度**: {best_route['max_slope']:.2f}%\n"
        report += f"**平均坡度**: {best_route['avg_slope']:.2f}%\n"
        report += f"**交通影响**: {best_route['traffic_impact']:.2f}\n"
        report += f"**总距离**: {best_route['total_distance']:.2f}公里\n\n"

        report += "### 优势分析\n\n"

        # 分析最优路线的优势
        if best_route['green_coverage'] > 0.3:
            report += "- 树荫覆盖率较高，有利于选手在高温天气下比赛\n"
        if best_route['max_slope'] <= 5.0:
            report += "- 最大坡度满足要求，不超过5%，确保赛道坡度适中\n"
        if best_route['avg_slope'] < 2.0:
            report += "- 平均坡度较小，整体赛道较为平缓\n"
        if best_route['traffic_impact'] < 2.0:
            report += "- 对城市交通影响较小，减少交通拥堵\n"
        if abs(best_route['total_distance'] - 42.195) < 1.0:
            report += "- 总距离接近标准马拉松距离(42.195公里)，符合国际标准\n"

        report += "\n### 建议改进\n\n"

        # 分析最优路线的不足和改进建议
        if best_route['green_coverage'] < 0.3:
            report += "- 建议增加树荫覆盖率，可考虑调整路线经过更多公园或绿地\n"
        if best_route['max_slope'] > 4.0:
            report += "- 最大坡度接近限制，建议微调路线降低最大坡度\n"
        if best_route['avg_slope'] > 2.0:
            report += "- 平均坡度略高，可考虑优化路线使整体更加平缓\n"
        if best_route['traffic_impact'] > 2.0:
            report += "- 交通影响较大，建议调整路线避开交通密集区域\n"
        if abs(best_route['total_distance'] - 42.195) > 1.0:
            report += "- 总距离与标准马拉松距离有偏差，建议微调路线使距离更接近42.195公里\n"

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, "route_evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"评估报告已保存到 {report_path}")

    return report_path

# 优化后的路线评估器
class RouteEvaluator:
    def __init__(self, data, green_space_data, terrain_data):
        self.data = data
        self.green_space_data = green_space_data
        self.terrain_data = terrain_data

        # 缓存已评估的路线，减少重复计算
        self.evaluated_routes = {}

        # 控制API调用频率，避免超出限制
        self.api_call_count = 0
        self.last_api_call_time = time.time()

    def generate_route_from_waypoints(self, waypoints):
        """
        根据途经点生成完整路线
        waypoints: 关键途经点列表 [(lng1, lat1), (lng2, lat2), ...]
        返回完整路线点集合
        """
        complete_route = []

        # 连接所有途经点
        for i in range(len(waypoints) - 1):
            origin = f"{waypoints[i][0]},{waypoints[i][1]}"
            destination = f"{waypoints[i+1][0]},{waypoints[i+1][1]}"

            # 控制API调用频率
            current_time = time.time()
            if self.api_call_count > 0 and current_time - self.last_api_call_time < 1.0:
                wait_time = 1.0 - (current_time - self.last_api_call_time)
                if wait_time > 0:
                    time.sleep(wait_time)

            self.api_call_count += 1
            self.last_api_call_time = time.time()

            # 使用高德API计算路线
            distance, route_info = calculate_route(origin, destination, route_type="drive", max_retries=2)

            if distance and route_info and "steps" in route_info:
                # 提取路线上的点
                for step in route_info["steps"]:
                    polyline = step.get("polyline", "")
                    if polyline:
                        points = polyline.split(";")
                        for point in points:
                            coords = point.split(",")
                            if len(coords) == 2:
                                try:
                                    lng, lat = float(coords[0]), float(coords[1])
                                    complete_route.append((lng, lat))
                                except:
                                    continue

            # 如果没有获取到路线，使用直线连接
            if not complete_route or (i > 0 and complete_route[-1] != waypoints[i]):
                # 添加起点
                complete_route.append(waypoints[i])
                # 添加终点
                complete_route.append(waypoints[i+1])

        return complete_route

    def generate_simple_route(self, waypoints, points_per_segment=10):
        """
        生成简化的路线（直线连接途经点）
        避免过度依赖API调用，用于快速评估
        """
        complete_route = []

        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i+1]

            # 添加起点
            complete_route.append(start)

            # 在起点和终点之间插入若干中间点
            for j in range(1, points_per_segment):
                ratio = j / points_per_segment
                lng = start[0] + (end[0] - start[0]) * ratio
                lat = start[1] + (end[1] - start[1]) * ratio
                complete_route.append((lng, lat))

            # 添加终点
            complete_route.append(end)

        return complete_route

    def evaluate_route(self, waypoints, cache=True, use_api=True):
        """
        评估路线的各项指标
        waypoints: 关键途经点列表 [(lng1, lat1), (lng2, lat2), ...]
        cache: 是否使用缓存结果
        use_api: 是否使用API获取详细路线，False时使用直线连接
        返回评估结果字典
        """
        # 生成路线缓存键
        cache_key = str(waypoints)

        # 检查是否有缓存结果
        if cache and cache_key in self.evaluated_routes:
            return self.evaluated_routes[cache_key]

        # 生成完整路线
        if use_api:
            route_points = self.generate_route_from_waypoints(waypoints)
        else:
            route_points = self.generate_simple_route(waypoints)

        # 计算各指标
        green_coverage = calculate_green_coverage(route_points, self.green_space_data)
        max_slope, avg_slope = calculate_slope(route_points, self.terrain_data)
        traffic_impact = evaluate_traffic_impact(route_points, self.data.get('road_connections'))

        # 计算路线总长度
        total_distance = 0
        for i in range(len(waypoints) - 1):
            distance = haversine_distance(
                waypoints[i][1], waypoints[i][0],
                waypoints[i+1][1], waypoints[i+1][0]
            )
            total_distance += distance

        # 评估结果
        result = {
            'green_coverage': green_coverage,
            'max_slope': max_slope,
            'avg_slope': avg_slope,
            'traffic_impact': traffic_impact,
            'total_distance': total_distance,
            'route_points': route_points
        }

        # 缓存结果
        if cache:
            self.evaluated_routes[cache_key] = result

        return result

    def objective_function(self, x, fixed_points=None):
        """
        多目标优化的目标函数
        x: 优化变量，表示途经点的经纬度 [lng1, lat1, lng2, lat2, ...]
        fixed_points: 固定不变的点（起点终点等）
        返回多个目标值: [-树荫覆盖率, 最大坡度, 交通影响]
        """
        # 将优化变量转换为途经点列表
        waypoints = []

        # 添加固定点（如起点）
        if fixed_points and fixed_points[0] is not None:
            waypoints.append(fixed_points[0])

        # 添加可变途经点
        for i in range(0, len(x), 2):
            if i+1 < len(x):
                waypoints.append((x[i], x[i+1]))

        # 添加固定点（如终点）
        if fixed_points and fixed_points[1] is not None:
            waypoints.append(fixed_points[1])

        # 在优化过程中使用简化路线进行快速评估
        # 只在最终结果中使用API获取详细路线
        result = self.evaluate_route(waypoints, use_api=False)

        # 返回多个目标值（需要最小化的值）
        return [
            -result['green_coverage'],  # 最大化树荫覆盖，所以取负
            result['max_slope'],        # 最小化最大坡度
            result['traffic_impact']    # 最小化交通影响
        ]

    def constraint_distance(self, x, fixed_points=None, target_distance=42.0, tolerance=5.0):
        """
        路线距离约束函数
        约束总距离在目标距离附近
        """
        # 将优化变量转换为途经点列表
        waypoints = []

        # 添加固定点（如起点）
        if fixed_points and fixed_points[0] is not None:
            waypoints.append(fixed_points[0])

        # 添加可变途经点
        for i in range(0, len(x), 2):
            if i+1 < len(x):
                waypoints.append((x[i], x[i+1]))

        # 添加固定点（如终点）
        if fixed_points and fixed_points[1] is not None:
            waypoints.append(fixed_points[1])

        # 计算总距离
        total_distance = 0
        for i in range(len(waypoints) - 1):
            distance = haversine_distance(
                waypoints[i][1], waypoints[i][0],
                waypoints[i+1][1], waypoints[i+1][0]
            )
            total_distance += distance

        # 返回约束值：总距离应在目标范围内
        # 返回值 >= 0 表示满足约束
        # 下界约束：总距离 >= target_distance - tolerance
        lower_bound = total_distance - (target_distance - tolerance)
        # 上界约束：总距离 <= target_distance + tolerance
        upper_bound = (target_distance + tolerance) - total_distance

        return min(lower_bound, upper_bound)

    def constraint_slope(self, x, fixed_points=None, max_allowed_slope=5.0):
        """
        坡度约束函数
        约束最大坡度不超过限制
        """
        # 将优化变量转换为途经点列表
        waypoints = []

        # 添加固定点（如起点）
        if fixed_points and fixed_points[0] is not None:
            waypoints.append(fixed_points[0])

        # 添加可变途经点
        for i in range(0, len(x), 2):
            if i+1 < len(x):
                waypoints.append((x[i], x[i+1]))

        # 添加固定点（如终点）
        if fixed_points and fixed_points[1] is not None:
            waypoints.append(fixed_points[1])

        # 在优化过程中使用简化路线进行快速评估
        result = self.evaluate_route(waypoints, use_api=False)

        # 返回约束值：最大坡度应小于等于允许值
        # 返回值 >= 0 表示满足约束
        return max_allowed_slope - result['max_slope']

# 优化后的多目标优化算法
def optimize_route(data, green_space_data, terrain_data, start_point=None, end_point=None,
                  target_distance=42.0, num_waypoints=5, population_size=20, max_iterations=30):
    """
    使用多目标优化算法优化路线
    data: 基础数据集
    green_space_data: 绿地数据
    terrain_data: 地形数据
    start_point: 起点坐标 (lng, lat)，如果为None则可变
    end_point: 终点坐标 (lng, lat)，如果为None则可变
    target_distance: 目标距离（公里）
    num_waypoints: 途经点数量
    population_size: 种群大小
    max_iterations: 最大迭代次数
    返回多个候选路线
    """
    print(f"开始优化路线，目标距离: {target_distance}公里，途经点数量: {num_waypoints}")

    # 创建路线评估器
    evaluator = RouteEvaluator(data, green_space_data, terrain_data)

    # 西安市中心坐标和边界
    xian_center = (108.9541, 34.2655)  # (lng, lat)

    # 减小搜索范围，聚焦于西安市中心区域
    lng_range = (108.9, 109.0)  # 西安市中心经度范围
    lat_range = (34.2, 34.3)   # 西安市中心纬度范围

    # 如果未指定起点终点，则在城市范围内随机选择
    if start_point is None:
        attractions = data['attractions']
        if len(attractions) > 0:
            # 随机选择一个景点作为起点
            start_idx = random.randint(0, len(attractions) - 1)
            start_point = (float(attractions.iloc[start_idx]['longitude']),
                          float(attractions.iloc[start_idx]['latitude']))
        else:
            # 使用西安市中心附近的点
            start_point = (xian_center[0] - 0.05, xian_center[1] - 0.05)

    if end_point is None:
        # 选择一个离起点足够远的景点作为终点
        candidates = []
        min_distance = target_distance / 3  # 起终点直线距离至少为目标距离的1/3

        for _, attraction in data['attractions'].iterrows():
            try:
                point = (float(attraction['longitude']), float(attraction['latitude']))
                distance = haversine_distance(start_point[1], start_point[0], point[1], point[0])
                if distance >= min_distance:
                    candidates.append(point)
            except:
                continue

        if candidates:
            # 随机选择一个候选点作为终点
            end_point = random.choice(candidates)
        else:
            # 使用西安市中心附近的点
            end_point = (xian_center[0] + 0.05, xian_center[1] + 0.05)

    # 固定起点和终点
    fixed_points = [start_point, end_point]
    print(f"起点: {start_point}, 终点: {end_point}")

    # 定义优化问题
    # 变量数量 = 途经点数量 * 2（每个点有经度和纬度）
    n_vars = num_waypoints * 2

    # 变量边界（经纬度范围）
    bounds = []
    for _ in range(num_waypoints):
        bounds.append((lng_range[0], lng_range[1]))  # 经度范围
        bounds.append((lat_range[0], lat_range[1]))  # 纬度范围

    # 创建目标函数和约束条件
    objective = partial(evaluator.objective_function, fixed_points=fixed_points)

    constraint_dist = NonlinearConstraint(
        partial(evaluator.constraint_distance, fixed_points=fixed_points,
                target_distance=target_distance, tolerance=5.0),
        0, float('inf')
    )

    constraint_slope = NonlinearConstraint(
        partial(evaluator.constraint_slope, fixed_points=fixed_points,
                max_allowed_slope=5.0),
        0, float('inf')
    )

    # 减少生成的候选解数量，降低计算量
    num_candidates = 3
    candidate_routes = []

    for i in range(num_candidates):
        print(f"生成候选路线 {i+1}/{num_candidates}...")

        # 随机初始化
        x0 = []
        for _ in range(num_waypoints):
            x0.append(random.uniform(lng_range[0], lng_range[1]))
            x0.append(random.uniform(lat_range[0], lat_range[1]))

        # 运行差分进化算法，减少迭代次数和种群大小
        try:
            result = differential_evolution(
                objective,
                bounds,
                constraints=(constraint_dist, constraint_slope),
                strategy='best1bin',
                maxiter=max_iterations,  # 减少最大迭代次数
                popsize=population_size, # 减少种群大小
                tol=0.05,                # 增大收敛容差
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=i,                  # 使用不同的随机种子
                disp=True,
                polish=False
            )

            # 将优化结果转换为途经点
            waypoints = [fixed_points[0]]  # 起点

            for j in range(0, len(result.x), 2):
                if j+1 < len(result.x):
                    waypoints.append((result.x[j], result.x[j+1]))

            waypoints.append(fixed_points[1])  # 终点

            # 最终评估使用API获取详细路线
            route_result = evaluator.evaluate_route(waypoints, use_api=True)

            # 添加到候选路线列表
            candidate_routes.append({
                'waypoints': waypoints,
                'route_points': route_result['route_points'],
                'green_coverage': route_result['green_coverage'],
                'max_slope': route_result['max_slope'],
                'avg_slope': route_result['avg_slope'],
                'traffic_impact': route_result['traffic_impact'],
                'total_distance': route_result['total_distance']
            })

            print(f"候选路线 {i+1} 生成完成:")
            print(f"  树荫覆盖率: {route_result['green_coverage']:.2f}")
            print(f"  最大坡度: {route_result['max_slope']:.2f}%")
            print(f"  平均坡度: {route_result['avg_slope']:.2f}%")
            print(f"  交通影响: {route_result['traffic_impact']:.2f}")
            print(f"  总距离: {route_result['total_distance']:.2f}公里")

        except Exception as e:
            print(f"优化算法运行出错: {e}")
            import traceback
            traceback.print_exc()

    return candidate_routes

# 主函数
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 打印系统信息
    print("\n===== 系统信息 =====")
    print(f"Python版本: {os.sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"GDAL缓存大小: {os.environ.get('GDAL_CACHEMAX', '未设置')}MB")
    print(f"OpenMP线程: {os.environ.get('OMP_NUM_THREADS', '未设置')}")
    print("==================\n")

    # 加载数据
    print("正在加载基础数据...")
    data = load_data()
    if not data:
        print("基础数据加载失败，请确保数据文件路径正确")
        return

    # 加载绿地数据
    print("\n正在加载绿地数据...")
    green_space_data = load_green_space_data()
    if not green_space_data:
        print("绿地数据加载失败，将使用随机数据代替")
        green_space_data = {
            'data': np.random.randint(0, 2, size=(500, 500)),
            'meta': {'crs': 'EPSG:4326'},
            'bounds': type('obj', (object,), {
                'left': 108.9, 'right': 109.0,
                'bottom': 34.2, 'top': 34.3
            }),
            'transform': None
        }

    # 加载地形数据
    print("\n正在加载地形数据...")
    terrain_data = load_terrain_data()
    if not terrain_data:
        print("地形数据加载失败，将使用随机数据代替")
        terrain_data = {
            'data': np.random.randint(350, 450, size=(500, 500)),
            'meta': {'crs': 'EPSG:4326', 'nodata': -9999},
            'bounds': type('obj', (object,), {
                'left': 108.9, 'right': 109.0,
                'bottom': 34.2, 'top': 34.3
            }),
            'transform': None
        }

    # 选择优化路线的类型
    print("\n请选择要优化的马拉松路线类型:")
    print("1. 全程马拉松 (42.195公里)")
    print("2. 半程马拉松 (21.0975公里)")
    print("3. 健康跑 (10公里)")

    try:
        choice = input("请输入选项(1-3): ")

        if choice == "1":
            target_distance = 42.195
            route_type = "全程马拉松"
            num_waypoints = 5
        elif choice == "2":
            target_distance = 21.0975
            route_type = "半程马拉松"
            num_waypoints = 3
        else:
            target_distance = 10.0
            route_type = "健康跑"
            num_waypoints = 2

        print(f"\n开始优化{route_type}路线 (目标距离: {target_distance}公里)...")

        # 根据路线类型调整优化参数
        if choice == "1":  # 全程马拉松
            population_size = 15
            max_iterations = 25
        elif choice == "2":  # 半程马拉松
            population_size = 12
            max_iterations = 20
        else:  # 健康跑
            population_size = 10
            max_iterations = 15

        # 使用多目标优化算法优化路线
        candidate_routes = optimize_route(
            data,
            green_space_data,
            terrain_data,
            target_distance=target_distance,
            num_waypoints=num_waypoints,
            population_size=population_size,
            max_iterations=max_iterations
        )

        if candidate_routes:
            # 可视化结果
            print("\n正在生成可视化结果...")
            map_path = visualize_routes(candidate_routes, data, green_space_data, terrain_data)

            # 生成评估报告
            print("\n正在生成评估报告...")
            report_path = generate_report(candidate_routes)

            print(f"\n结果已保存:")
            print(f"- 可视化地图: {map_path}")
            print(f"- 评估报告: {report_path}")
        else:
            print("未找到符合条件的路线")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n程序运行完成!")

if __name__ == "__main__":
    main()