#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """
    数据清洗类，专门用于处理各个附件数据的缺失值和异常值
    """
    def __init__(self, data_dir="processed_data"):
        """
        初始化数据清洗器

        参数:
        data_dir: 预处理数据的目录
        """
        self.data_dir = data_dir
        self.output_dir = os.path.join(data_dir, "cleaned")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self, file_name):
        """
        加载数据文件

        参数:
        file_name: 数据文件名

        返回:
        加载的数据框
        """
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None

        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                print(f"不支持的文件格式: {file_path}")
                return None

            print(f"成功加载数据: {file_name}, 形状: {df.shape}")
            return df
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None

    def detect_outliers(self, series, method='zscore', threshold=3):
        """
        检测异常值

        参数:
        series: 数据系列
        method: 检测方法，可选 'zscore', 'iqr', 'mad'
        threshold: 异常值阈值

        返回:
        异常值索引的布尔数组
        """
        if method == 'zscore':
            # Z分数法
            z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
            return z_scores > threshold

        elif method == 'iqr':
            # 四分位数法
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return (series < lower_bound) | (series > upper_bound)

        elif method == 'mad':
            # 中位数绝对偏差法
            median = series.median()
            mad = np.median(np.abs(series - median))
            mad_scores = np.abs(series - median) / mad
            return mad_scores > threshold

        else:
            print(f"不支持的异常值检测方法: {method}")
            return np.zeros(len(series), dtype=bool)

    def smooth_fill(self, df, col, group_cols=['station_id', 'month']):
        # 1. 先用同站点的时序线性插值
        df[col] = df.groupby('station_id')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
        # 2. 再用分组中位数填充剩余缺失
        df[col] = df.groupby(group_cols)[col].transform(lambda x: x.fillna(x.median()))
        # 3. 最后用全局中位数兜底
        df[col] = df[col].fillna(df[col].median())
        return df

    def fill_precip_by_distribution(self, df):
        # 针对降水量，缺失时用同站点同月的经验分布采样填充
        # 处理1小时降水量
        for col in ['precipitation_1h', 'precipitation_6h']:
            if col in df.columns:
                for (station, month), group in df.groupby(['station_id', 'month']):
                    valid = group[col].dropna()
                    if len(valid) > 0:
                        sampled = np.random.choice(valid, size=group[col].isna().sum(), replace=True)
                        df.loc[group.index[group[col].isna()], col] = sampled
        return df

    def handle_missing_values(self, df, numeric_cols=None, categorical_cols=None, date_cols=None, smooth_fill_cols=None):
        """
        处理缺失值，支持气象数据的平滑填充
        新增参数smooth_fill_cols：需要平滑填充的列
        """
        df_cleaned = df.copy()

        # 打印缺失值统计
        missing_stats = df.isnull().sum()
        missing_stats = missing_stats[missing_stats > 0]
        if len(missing_stats) > 0:
            print("缺失值统计:")
            print(missing_stats)
            print(f"总缺失值比例: {df.isnull().sum().sum() / df.size:.4f}")

        # 如果未指定列类型，自动检测
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            # 排除可能的日期列
            if date_cols:
                categorical_cols = [col for col in categorical_cols if col not in date_cols]

        # --- 新增：气象数据平滑填充 ---
        if smooth_fill_cols is not None:
            for col in smooth_fill_cols:
                if col in df_cleaned.columns and df_cleaned[col].isnull().any():
                    print(f"对列 {col} 采用平滑插值+分组中位数填充...")
                    df_cleaned = self.smooth_fill(df_cleaned, col)
            # 降水量分布采样
            precip_cols = [col for col in smooth_fill_cols if 'precipitation' in col]
            if precip_cols and any(col in df_cleaned.columns for col in precip_cols):
                has_missing = any(df_cleaned[col].isnull().any() for col in precip_cols if col in df_cleaned.columns)
                if has_missing:
                    print("对降水量采用分布采样填充...")
                    df_cleaned = self.fill_precip_by_distribution(df_cleaned)

        # --- 其余数值列用中位数填充 ---
        for col in numeric_cols:
            if col in df_cleaned.columns and df_cleaned[col].isnull().any():
                if smooth_fill_cols and col in smooth_fill_cols:
                    continue  # 已处理
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)
                print(f"列 {col} 的缺失值已用中位数 {median_val} 填充")

        # 处理分类列缺失值
        for col in categorical_cols:
            if col in df_cleaned.columns and df_cleaned[col].isnull().any():
                mode_val = df_cleaned[col].mode()[0]
                df_cleaned[col].fillna(mode_val, inplace=True)
                print(f"列 {col} 的缺失值已用众数 {mode_val} 填充")

        # 处理日期列缺失值
        if date_cols:
            for col in date_cols:
                if col in df_cleaned.columns and df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(method='ffill', inplace=True)
                    df_cleaned[col].fillna(method='bfill', inplace=True)
                    print(f"列 {col} 的缺失值已用相邻日期填充")

        return df_cleaned

    def handle_outliers(self, df, numeric_cols=None, method='winsorize'):
        """
        处理异常值

        参数:
        df: 数据框
        numeric_cols: 数值列列表
        method: 处理方法，可选 'winsorize', 'remove', 'cap'

        返回:
        处理后的数据框
        """
        df_cleaned = df.copy()

        # 如果未指定数值列，自动检测
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        for col in numeric_cols:
            if col not in df.columns:
                continue

            # 检测异常值
            is_outlier = self.detect_outliers(df[col].dropna(), method='iqr', threshold=3)
            outlier_indices = df[col].dropna().index[is_outlier]

            if len(outlier_indices) > 0:
                print(f"列 {col} 检测到 {len(outlier_indices)} 个异常值")

                if method == 'winsorize':
                    # Winsorizing: 将异常值替换为分位数
                    lower_bound = df[col].quantile(0.05)
                    upper_bound = df[col].quantile(0.95)
                    df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                    print(f"  已将异常值限制在 [{lower_bound:.2f}, {upper_bound:.2f}] 范围内")

                elif method == 'remove':
                    # 移除异常值所在行
                    df_cleaned = df_cleaned.drop(outlier_indices)
                    print(f"  已移除包含异常值的 {len(outlier_indices)} 行")

                elif method == 'cap':
                    # 使用阈值替换异常值
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr

                    # 替换异常值
                    df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                    df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                    print(f"  已将异常值截断在 [{lower_bound:.2f}, {upper_bound:.2f}] 范围内")

        return df_cleaned

    def clean_meteorological_data(self):
        """
        清洗气象数据，采用平滑插值+分组中位数+分布采样填充
        """
        print("\n开始清洗气象数据...")
        df = self.load_data("附件1_meteorological_data.csv")

        if df is None:
            return None

        # 转换日期列
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 处理缺失值 - 平滑填充气象变量
        numeric_cols = ['temperature', 'dew_point', 'pressure', 'wind_direction',
                       'wind_speed', 'cloud_cover', 'precipitation_1h', 'precipitation_6h']
        date_cols = ['datetime']
        smooth_fill_cols = ['temperature', 'dew_point', 'pressure', 'wind_speed', 'cloud_cover', 'precipitation_1h', 'precipitation_6h']

        # 若无station_id/month则补充
        if 'month' not in df.columns:
            df['month'] = df['datetime'].dt.month
        if 'station_id' not in df.columns:
            print('警告：缺少station_id，无法分组平滑填充')
            smooth_fill_cols = None

        df_cleaned = self.handle_missing_values(df, numeric_cols=numeric_cols, date_cols=date_cols, smooth_fill_cols=smooth_fill_cols)

        # 处理异常值 - 排除降水量列
        outlier_cols = [c for c in numeric_cols if 'precipitation' not in c]
        df_cleaned = self.handle_outliers(df_cleaned, numeric_cols=outlier_cols)

        # 保存清洗后的数据
        output_file = os.path.join(self.output_dir, "附件1_meteorological_data_cleaned.csv")
        df_cleaned.to_csv(output_file, index=False)
        print(f"气象数据清洗完成，已保存到 {output_file}")

        return df_cleaned

    def clean_subway_traffic_data(self):
        """
        清洗轨道交通客运量数据

        返回:
        清洗后的轨道交通数据
        """
        print("\n开始清洗轨道交通客运量数据...")
        df = self.load_data("附件2_subway_traffic.csv")

        if df is None:
            return None

        # 处理缺失值
        numeric_cols = ['运营线路条数', '运营里程（公里）', '客运量（万人次）',
                        '进站量（万人次）', '客运强度（万人次每公里日）']
        categorical_cols = ['城市']

        df_cleaned = self.handle_missing_values(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

        # 处理异常值
        df_cleaned = self.handle_outliers(df_cleaned, numeric_cols=numeric_cols)

        # 保存清洗后的数据
        output_file = os.path.join(self.output_dir, "附件2_subway_traffic_cleaned.csv")
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"轨道交通客运量数据清洗完成，已保存到 {output_file}")

        return df_cleaned

    def clean_population_data(self):
        """
        清洗人口普查数据

        返回:
        清洗后的人口数据
        """
        print("\n开始清洗人口普查数据...")
        # 分别处理省级和市级数据
        province_df = self.load_data("附件3_population_province.csv")
        city_df = self.load_data("附件3_population_city.csv")

        if province_df is None and city_df is None:
            return None, None

        # 处理省级数据
        if province_df is not None:
            # 确定数值列和分类列
            numeric_cols = [col for col in province_df.columns if any(x in col for x in ['岁_男', '岁_女'])]
            categorical_cols = ['地名_Unnamed: 1_level_1', '普查名称']

            # 处理缺失值
            province_cleaned = self.handle_missing_values(province_df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

            # 处理异常值
            province_cleaned = self.handle_outliers(province_cleaned, numeric_cols=numeric_cols)

            # 保存清洗后的数据
            output_file = os.path.join(self.output_dir, "附件3_population_province_cleaned.csv")
            province_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"省级人口数据清洗完成，已保存到 {output_file}")
        else:
            province_cleaned = None

        # 处理市级数据
        if city_df is not None:
            # 确定数值列和分类列
            numeric_cols = [col for col in city_df.columns if any(x in col for x in ['岁_男', '岁_女'])]
            categorical_cols = ['地名_Unnamed: 1_level_1', '普查名称']

            # 处理缺失值
            city_cleaned = self.handle_missing_values(city_df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

            # 处理异常值
            city_cleaned = self.handle_outliers(city_cleaned, numeric_cols=numeric_cols)

            # 保存清洗后的数据
            output_file = os.path.join(self.output_dir, "附件3_population_city_cleaned.csv")
            city_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"市级人口数据清洗完成，已保存到 {output_file}")
        else:
            city_cleaned = None

        return province_cleaned, city_cleaned

    def clean_marathon_history_data(self):
        """
        清洗马拉松赛历数据

        返回:
        清洗后的马拉松赛历数据
        """
        print("\n开始清洗马拉松赛历数据...")
        df = self.load_data("附件12_marathon_history.csv")

        if df is None:
            return None

        # 打印列名，以便调试
        print(f"数据列名: {df.columns.tolist()}")

        # 检查数据类型
        print("数据类型:")
        for col in df.columns:
            print(f"{col}: {df[col].dtype}")

        # 确定数值列、分类列和日期列
        numeric_cols = []
        categorical_cols = []
        date_cols = []

        # 尝试根据列名和数据类型确定列类型
        for col in df.columns:
            # 尝试找出数值列
            if df[col].dtype in ['int64', 'float64'] or any(x in col.lower() for x in ['人数', '费', '率']):
                numeric_cols.append(col)
            # 尝试找出日期列
            elif any(x in col.lower() for x in ['日期', 'time', 'date']):
                date_cols.append(col)
                # 尝试转换为日期格式
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    print(f"警告: 无法将列 {col} 转换为日期格式")
            # 其他列作为分类列
            else:
                categorical_cols.append(col)

        print(f"识别的数值列: {numeric_cols}")
        print(f"识别的日期列: {date_cols}")
        print(f"识别的分类列: {categorical_cols}")

        # 处理缺失值
        df_cleaned = self.handle_missing_values(df, numeric_cols=numeric_cols,
                                            categorical_cols=categorical_cols,
                                            date_cols=date_cols)

        # 处理异常值（只处理数值列）
        df_cleaned = self.handle_outliers(df_cleaned, numeric_cols=numeric_cols)

        # 处理完赛率异常值（范围应在0-1之间）
        for col in numeric_cols:
            if '完赛率' in col:
                df_cleaned.loc[df_cleaned[col] > 1, col] = 1
                df_cleaned.loc[df_cleaned[col] < 0, col] = 0
                print(f"已修正完赛率列 {col} 的范围为0-1")

        # 重新计算缺失的完赛率
        registration_col = None
        completion_col = None
        ratio_col = None

        for col in numeric_cols:
            if '报名人数' in col:
                registration_col = col
            elif '完赛人数' in col:
                completion_col = col
            elif '完赛率' in col:
                ratio_col = col

        if registration_col and completion_col and ratio_col:
            mask = (df_cleaned[ratio_col].isnull()) & (df_cleaned[registration_col] > 0) & (df_cleaned[completion_col] > 0)
            df_cleaned.loc[mask, ratio_col] = df_cleaned.loc[mask, completion_col] / df_cleaned.loc[mask, registration_col]
            print(f"已计算 {mask.sum()} 条缺失的完赛率数据")

        # 保存清洗后的数据
        output_file = os.path.join(self.output_dir, "附件12_marathon_history_cleaned.csv")
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"马拉松赛历数据清洗完成，已保存到 {output_file}")

        return df_cleaned

    def clean_xian_basic_data(self, data_type):
        """
        清洗西安市基础数据

        参数:
        data_type: 数据类型，如'住宿'、'餐饮'、'景点'等

        返回:
        清洗后的数据
        """
        print(f"\n开始清洗西安市{data_type}数据...")

        # 确定文件名
        file_mapping = {
            '住宿': '西安市住宿服务数据.csv',
            '餐饮': '西安市餐饮数据.csv',
            '景点': '西安市风景名胜数据.csv',
            '道路': '西安市道路附属设施数据.csv'
        }

        if data_type not in file_mapping:
            print(f"不支持的数据类型: {data_type}")
            return None

        # 从原始附件加载数据
        file_path = os.path.join("附件", "附件5：西安市基础数据", file_mapping[data_type])

        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None

        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"成功加载数据: {file_mapping[data_type]}, 形状: {df.shape}")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='gbk')
                print(f"成功加载数据: {file_mapping[data_type]}, 形状: {df.shape}")
            except Exception as e:
                print(f"加载数据时出错: {e}")
                return None

        # 打印列名，以便调试
        print(f"数据列名: {df.columns.tolist()}")

        # 找出经纬度列
        lat_col = None
        lon_col = None

        # 尝试识别经纬度列
        possible_lat_cols = ['lat', 'latitude', '纬度', 'Lat', 'LAT']
        possible_lon_cols = ['lon', 'longitude', '经度', 'Lon', 'LON', 'lng', 'LNG']

        for col in df.columns:
            col_lower = col.lower()
            if any(lat_name in col_lower for lat_name in possible_lat_cols):
                lat_col = col
            if any(lon_name in col_lower for lon_name in possible_lon_cols):
                lon_col = col

        if lat_col is None or lon_col is None:
            print(f"警告: 无法精确匹配经纬度列名。尝试使用模糊匹配")
            # 尝试模糊匹配
            for col in df.columns:
                if 'lat' in col.lower():
                    lat_col = col
                if 'lon' in col.lower() or 'lng' in col.lower():
                    lon_col = col

        if lat_col is None or lon_col is None:
            print(f"错误: 无法找到经纬度列，跳过地理位置清洗")
            # 继续处理其他列，不进行地理位置清洗
            has_geo = False
        else:
            print(f"找到经纬度列: {lat_col}, {lon_col}")
            has_geo = True

        # 确定评分列
        rating_col = None
        for col in df.columns:
            if 'rating' in col.lower() or '评分' in col or 'rate' in col.lower():
                rating_col = col
                break

        # 处理缺失值和异常值
        # 根据数据类型确定数值列和分类列
        numeric_cols = []
        categorical_cols = []

        # 添加评分列（如果存在）
        if rating_col:
            numeric_cols.append(rating_col)

        # 添加经纬度列（如果存在）
        if has_geo:
            numeric_cols.extend([lat_col, lon_col])

        # 确定分类列（所有对象类型的列）
        for col in df.columns:
            if col not in numeric_cols and df[col].dtype == 'object':
                categorical_cols.append(col)

        # 处理缺失值
        df_cleaned = self.handle_missing_values(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

        # 处理异常值（只处理数值列）
        df_cleaned = self.handle_outliers(df_cleaned, numeric_cols=numeric_cols)

        # 地理位置清洗（如果存在经纬度列）
        if has_geo:
            # 删除没有地理位置的记录
            df_cleaned = df_cleaned.dropna(subset=[lat_col, lon_col])
            print(f"移除了 {len(df) - len(df_cleaned)} 条缺失经纬度的记录")

            # 确保经纬度在合理范围内（西安大致范围：经度108-109.5，纬度33.5-35）
            initial_len = len(df_cleaned)
            df_cleaned = df_cleaned[(df_cleaned[lon_col] > 108) & (df_cleaned[lon_col] < 109.5) &
                                  (df_cleaned[lat_col] > 33.5) & (df_cleaned[lat_col] < 35)]
            print(f"移除了 {initial_len - len(df_cleaned)} 条经纬度异常的记录")

        # 保存清洗后的数据
        output_file = os.path.join(self.output_dir, f"附件5_西安市{data_type}数据_cleaned.csv")
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"西安市{data_type}数据清洗完成，已保存到 {output_file}")

        return df_cleaned

    def clean_xian_route_data(self):
        """
        清洗西安市赛跑线路相关数据
        """
        print("\n开始清洗西安市赛跑线路相关数据...")
        cleaned_data = {}

        # 1. 加载地铁站点数据
        subway_df = self.load_data("附件9_xian_subway_stations.csv")
        if subway_df is not None:
            # 确保经纬度在西安范围内（经度108-109.5，纬度33.5-35）
            subway_df = subway_df[
                (subway_df['longitude'] > 108) &
                (subway_df['longitude'] < 109.5) &
                (subway_df['latitude'] > 33.5) &
                (subway_df['latitude'] < 35)
            ]
            print(f"地铁站点数据清洗完成，共 {len(subway_df)} 条记录")
            # 保存清洗后的数据
            output_file = os.path.join(self.output_dir, "附件9_xian_subway_stations_cleaned.csv")
            subway_df.to_csv(output_file, index=False)
            cleaned_data['subway'] = subway_df

        # 2. 加载住宿设施数据
        hotel_df = self.load_data("附件5_xian_hotels.csv")
        if hotel_df is not None:
            # 假设住宿容量列名为'capacity'
            if 'capacity' in hotel_df.columns:
                # 处理缺失值
                hotel_df['capacity'] = hotel_df['capacity'].fillna(100)
                # 处理异常值
                q1 = hotel_df['capacity'].quantile(0.25)
                q3 = hotel_df['capacity'].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                hotel_df = hotel_df[(hotel_df['capacity'] >= lower_bound) & (hotel_df['capacity'] <= upper_bound)]
            print(f"住宿设施数据清洗完成，共 {len(hotel_df)} 条记录")
            # 保存清洗后的数据
            output_file = os.path.join(self.output_dir, "附件5_xian_hotels_cleaned.csv")
            hotel_df.to_csv(output_file, index=False)
            cleaned_data['hotels'] = hotel_df

        # 3. 加载餐饮设施数据
        restaurant_df = self.load_data("附件5_xian_restaurants.csv")
        if restaurant_df is not None:
            print(f"餐饮设施数据清洗完成，共 {len(restaurant_df)} 条记录")
            # 保存清洗后的数据
            output_file = os.path.join(self.output_dir, "附件5_xian_restaurants_cleaned.csv")
            restaurant_df.to_csv(output_file, index=False)
            cleaned_data['restaurants'] = restaurant_df

        # 4. 加载景点数据
        attraction_df = self.load_data("附件5_xian_attractions.csv")
        if attraction_df is not None:
            print(f"景点数据清洗完成，共 {len(attraction_df)} 条记录")
            # 保存清洗后的数据
            output_file = os.path.join(self.output_dir, "附件5_xian_attractions_cleaned.csv")
            attraction_df.to_csv(output_file, index=False)
            cleaned_data['attractions'] = attraction_df

        # 5. 加载道路连接信息
        road_df = self.load_data("附件7_xian_road_connections.csv")
        if road_df is not None:
            print(f"道路连接信息清洗完成，共 {len(road_df)} 条记录")
            # 保存清洗后的数据
            output_file = os.path.join(self.output_dir, "附件7_xian_road_connections_cleaned.csv")
            road_df.to_csv(output_file, index=False)
            cleaned_data['roads'] = road_df

        # 6. 加载公交站点数据
        bus_df = self.load_data("附件8_xian_bus_stations.csv")
        if bus_df is not None:
            # 确保经纬度在西安范围内（经度108-109.5，纬度33.5-35）
            bus_df = bus_df[
                (bus_df['longitude'] > 108) &
                (bus_df['longitude'] < 109.5) &
                (bus_df['latitude'] > 33.5) &
                (bus_df['latitude'] < 35)
            ]
            print(f"公交站点数据清洗完成，共 {len(bus_df)} 条记录")
            # 保存清洗后的数据
            output_file = os.path.join(self.output_dir, "附件8_xian_bus_stations_cleaned.csv")
            bus_df.to_csv(output_file, index=False)
            cleaned_data['bus'] = bus_df

        print("西安市赛跑线路相关数据清洗完成！")

        return cleaned_data

    def clean_all_data(self):
        """
        清洗所有数据

        返回:
        清洗后的数据字典
        """
        print("开始清洗所有数据...")
        cleaned_data = {}

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 清洗气象数据
        cleaned_data['meteorological'] = self.clean_meteorological_data()

        # 清洗轨道交通客运量数据
        cleaned_data['subway_traffic'] = self.clean_subway_traffic_data()

        # 清洗人口普查数据
        province_cleaned, city_cleaned = self.clean_population_data()
        cleaned_data['population_province'] = province_cleaned
        cleaned_data['population_city'] = city_cleaned

        # 清洗马拉松赛历数据
        cleaned_data['marathon_history'] = self.clean_marathon_history_data()

        # 清洗西安市基础数据
        cleaned_data['xian_accommodation'] = self.clean_xian_basic_data('住宿')
        cleaned_data['xian_restaurant'] = self.clean_xian_basic_data('餐饮')
        cleaned_data['xian_attraction'] = self.clean_xian_basic_data('景点')
        cleaned_data['xian_road_facility'] = self.clean_xian_basic_data('道路')

        print("所有数据清洗完成！")

        return cleaned_data

if __name__ == "__main__":
    # 创建数据清洗器
    cleaner = DataCleaner()

    # 清洗所有数据
    cleaned_data = cleaner.clean_all_data()