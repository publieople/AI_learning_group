#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from idna import encode
import pandas as pd
import numpy as np
import glob
import re
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.plot import show
import warnings
import argparse
import chardet
from data_cleaning import DataCleaner
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, base_path="附件"):
        """
        数据预处理类

        参数:
        base_path: 附件数据的基础路径
        """
        self.base_path = base_path
        # 创建输出目录
        self.output_dir = "processed_data"
        os.makedirs(self.output_dir, exist_ok=True)

    def process_meteorological_data(self, years=None, cities=None):
        """
        处理气象数据（附件1）

        参数:
        years: 需要处理的年份列表，如果为None则处理所有年份
        cities: 需要处理的城市站点列表，如果为None则处理所有站点

        返回:
        处理后的气象数据DataFrame
        """
        print("开始处理气象数据...")

        # 气象数据路径
        meteo_base_path = os.path.join(self.base_path, "附件1：中国气象数据")
        print(f"气象数据基础路径: {meteo_base_path}")

        # 如果未指定年份，获取所有年份文件夹
        if years is None:
            years = [d for d in os.listdir(meteo_base_path)
                     if os.path.isdir(os.path.join(meteo_base_path, d)) and 'china_isd_lite' in d]
            years = [y.split('_')[-1] for y in years]

        print(f"待处理年份: {years}")

        all_data = []
        processed_stations = set()  # 用于记录已处理的站点
        error_files = []  # 记录处理失败的文件
        precip_stats = {'total': 0, 'non_zero': 0, 'zero': 0}  # 统计降水量数据

        for year in years:
            print(f"\n处理{year}年的气象数据...")
            year_path = os.path.join(meteo_base_path, f"china_isd_lite_{year}")

            if not os.path.exists(year_path):
                print(f"警告: {year_path} 路径不存在")
                continue

            # 获取该年份下所有站点文件
            station_files = glob.glob(os.path.join(year_path, "*"))
            print(f"找到{len(station_files)}个站点文件")

            for station_file in station_files:
                try:
                    # 提取站点ID
                    station_id = os.path.basename(station_file).split('-')[0]

                    # 如果已经处理过该站点，跳过
                    if station_id in processed_stations:
                        continue

                    processed_stations.add(station_id)

                    # 读取数据
                    try:
                        with open(station_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                    except UnicodeDecodeError:
                        # 如果UTF-8解码失败，尝试其他编码
                        with open(station_file, 'r', encoding='gbk') as f:
                            lines = f.readlines()

                    valid_lines = 0  # 记录有效数据行数
                    # 解析数据
                    for line_num, line in enumerate(lines, 1):
                        try:
                            parts = line.strip().split()
                            if len(parts) >= 9:  # 确保有足够的数据列
                                year, month, day, hour = parts[0], parts[1], parts[2], parts[3]

                                # 数据验证
                                if not all(x.isdigit() for x in [year, month, day, hour]):
                                    continue

                                # 温度、露点、气压、风向、风速、云量、降水量
                                temp = float(parts[4]) / 10.0 if parts[4] != '-9999' else np.nan
                                dewp = float(parts[5]) / 10.0 if parts[5] != '-9999' else np.nan
                                pressure = float(parts[6]) / 10.0 if parts[6] != '-9999' else np.nan
                                wind_dir = float(parts[7]) if parts[7] != '-9999' else np.nan
                                wind_speed = float(parts[8]) / 10.0 if parts[8] != '-9999' else np.nan

                                # 云量和降水量处理
                                cloud = np.nan
                                precip = np.nan

                                # 检查是否有云量数据
                                if len(parts) > 9:
                                    try:
                                        cloud = float(parts[9]) if parts[9] != '-9999' else np.nan
                                    except ValueError:
                                        cloud = np.nan

                                # 检查是否有降水量数据（第12列，索引11）
                                if len(parts) > 11:
                                    try:
                                        precip_str = parts[11].strip()
                                        if precip_str != '-9999':
                                            # 降水量数据需要除以10转换为毫米
                                            precip = float(precip_str) / 10.0
                                            precip_stats['total'] += 1
                                            if precip > 0:
                                                precip_stats['non_zero'] += 1
                                            else:
                                                precip_stats['zero'] += 1
                                        else:
                                            precip = np.nan
                                    except (ValueError, IndexError) as e:
                                        precip = np.nan

                                # 数据合理性检查 - 放宽条件
                                # 1. 只检查非空值
                                # 2. 使用更合理的范围
                                if ((np.isnan(temp) or -50 <= temp <= 50) and  # 温度范围：-50℃到50℃
                                    (np.isnan(dewp) or -50 <= dewp <= 50) and  # 露点范围：-50℃到50℃
                                    (np.isnan(pressure) or 800 <= pressure <= 1100) and  # 气压范围：800-1100百帕
                                    (np.isnan(wind_dir) or 0 <= wind_dir <= 360) and  # 风向范围：0-360度
                                    (np.isnan(wind_speed) or 0 <= wind_speed <= 100) and  # 风速范围：0-100米/秒
                                    (np.isnan(cloud) or 0 <= cloud <= 8) and  # 云量范围：0-8
                                    (np.isnan(precip) or 0 <= precip <= 1000)):  # 降水量范围：0-1000毫米

                                    # 构建数据行
                                    data_row = {
                                        'station_id': station_id,
                                        'datetime': f"{year}-{month.zfill(2)}-{day.zfill(2)} {hour.zfill(2)}:00:00",
                                        'temperature': temp,
                                        'dew_point': dewp,
                                        'pressure': pressure,
                                        'wind_direction': wind_dir,
                                        'wind_speed': wind_speed,
                                        'cloud_cover': cloud,
                                        'precipitation': precip
                                    }
                                    all_data.append(data_row)
                                    valid_lines += 1

                        except (ValueError, IndexError) as e:
                            continue

                except Exception as e:
                    print(f"处理文件{station_file}时出错: {e}")
                    error_files.append(station_file)
                    continue

        # 检查是否有数据被处理
        if not all_data:
            print("警告: 没有成功处理任何数据！")
            return pd.DataFrame()  # 返回空DataFrame

        # 转换为DataFrame
        df = pd.DataFrame(all_data)
        print(f"\n成功处理的数据行数: {len(df)}")

        # 转换日期时间列
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 添加时间相关特征
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour

        # 数据统计信息
        print("\n数据统计信息:")
        print(f"总记录数: {len(df)}")
        print(f"站点数量: {df['station_id'].nunique()}")
        print(f"时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
        print("\n降水量统计:")
        print(f"总降水量记录数: {precip_stats['total']}")
        print(f"非零降水量记录数: {precip_stats['non_zero']}")
        print(f"零降水量记录数: {precip_stats['zero']}")
        print("\n各气象要素的统计信息:")
        print(df[['temperature', 'dew_point', 'pressure', 'wind_speed', 'cloud_cover', 'precipitation']].describe())

        # 保存处理后的数据
        output_file = os.path.join(self.output_dir, "附件1_meteorological_data.csv")
        df.to_csv(output_file, index=False)
        print(f"\n气象数据处理完成，已保存到{output_file}")

        # 如果有处理失败的文件，输出信息
        if error_files:
            print("\n处理失败的文件:")
            for file in error_files:
                print(f"- {file}")

        return df

    def process_marathon_history(self):
        """
        处理马拉松赛历数据（附件12）

        返回:
        处理后的马拉松赛历数据DataFrame
        """
        print("开始处理马拉松赛历数据...")

        # 马拉松赛历数据路径
        marathon_path = os.path.join(self.base_path, "附件12：马拉松赛历数据", "马拉松赛历数据.xlsx")

        try:
            # 读取Excel文件
            df = pd.read_excel(marathon_path)

            # 数据清洗和转换
            # 1. 处理日期格式
            if 'raceTime' in df.columns:
                df['raceTime'] = pd.to_datetime(df['raceTime'], errors='coerce')

            # 2. 处理数值型数据
            numeric_columns = ['raceScale', 'lon', 'lat']
            for col in numeric_columns:
                if col in df.columns:
                    if col == 'raceScale':
                        # 处理raceScale列中的'人'字符
                        df[col] = df[col].astype(str).str.replace('人', '', regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            # 3. 提取年份和月份
            if 'raceTime' in df.columns:
                df['年份'] = df['raceTime'].dt.year.astype('Int64')
                df['月份'] = df['raceTime'].dt.month.astype('Int64')

            # 保存处理后的数据
            output_file = os.path.join(self.output_dir, "附件12_marathon_history.csv")
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"马拉松赛历数据处理完成，已保存到{output_file}")

            return df

        except Exception as e:
            print(f"处理马拉松赛历数据时出错: {e}")
            return None

    def process_subway_data(self):
        """
        兼容处理2021-2024年轨道交通客运量数据（附件2）
        只保留：城市、运营线路条数、运营里程（公里）、客运量（万人次）、进站量（万人次）、客运强度（万人次每公里日）
        """
        print("开始处理轨道交通客运量数据...")
        import re
        all_data = []
        years = ['2021', '2022', '2023', '2024']
        base_path = os.path.join(self.base_path, "附件2：2021-2024年我国主要城市逐月轨道交通客运量数据")

        # 只保留以下字段
        keep_fields = {
            '城市': ['城市'],
            '运营线路条数': ['运营线路', '运营线', '线路数', '线路条'],
            '运营里程（公里）': ['运营里', '里程'],
            '客运量（万人次）': ['客运量'],
            '进站量（万人次）': ['进站量'],
            '客运强度（万人次每公里日）': ['客运强']
        }

        for year in years:
            file_path = os.path.join(base_path, f"{year}.xlsx")
            if not os.path.exists(file_path):
                print(f"未找到{file_path}")
                continue

            print(f"正在处理{year}年数据...")
            xls = pd.ExcelFile(file_path)

            for sheet_name in xls.sheet_names:
                print(f"处理工作表: {sheet_name}")
                try:
                    # 读取整个sheet
                    df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str, header=None)

                    # 查找标题行
                    title_row = None
                    for idx, row in df.iterrows():
                        row_str = ' '.join(str(val) for val in row if pd.notna(val))
                        if '序号' in row_str:
                            title_row = idx
                            break

                    if title_row is None:
                        print(f"警告：在{sheet_name}中未找到标题行")
                        continue

                    # 获取标题行
                    headers = df.iloc[title_row]

                    # 创建列映射（用前三个汉字匹配）
                    col_map = {}
                    for idx, header in enumerate(headers):
                        header_str = str(header).replace(' ', '')
                        prefix = header_str[:3]
                        for field, patterns in keep_fields.items():
                            for pat in patterns:
                                if prefix == pat[:3]:
                                    col_map[field] = idx
                                    break

                    # 必须有城市和客运量
                    if '城市' not in col_map or '客运量（万人次）' not in col_map:
                        print(f"警告：{sheet_name}中缺少必要的列（城市或客运量）")
                        continue

                    # 从标题行之后开始处理数据，直到遇到总计行
                    for idx in range(title_row + 1, len(df)):
                        row = df.iloc[idx]
                        city = str(row[col_map['城市']]).strip() if '城市' in col_map else None
                        if '总' in city or city == '' or city == 'nan':
                            break

                        # 解析月份
                        m = re.search(r'(\d+)', sheet_name)
                        month = int(m.group(1)) if m else None

                        # 构建数据行
                        data_row = {
                            '城市': city,
                            '年份': int(year),
                            '月份': month
                        }
                        for field in keep_fields:
                            if field == '城市':
                                continue
                            idx_col = col_map.get(field, None)
                            value = None
                            if idx_col is not None:
                                try:
                                    value_str = str(row[idx_col]).replace(',', '').replace(' ', '')
                                    value = float(value_str) if value_str and value_str != 'nan' else None
                                except Exception as e:
                                    print(f"警告：无法转换{field}数据 '{row[idx_col]}' - {str(e)}")
                            data_row[field] = value
                        # 只要城市和客运量有值就保留
                        if city and data_row['客运量（万人次）'] is not None:
                            all_data.append(data_row)
                except Exception as e:
                    print(f"处理{year}年{sheet_name}时出错: {str(e)}")
                    continue

        # 合并所有年份
        df_all = pd.DataFrame(all_data)
        if df_all.empty:
            print("未能提取到任何有效的城市客运量数据，请检查原始表格格式或字段匹配规则。")
            return df_all

        print(f"成功提取到{len(df_all)}条数据记录")
        print("提取到的字段：", df_all.columns.tolist())
        print("数据预览：")
        print(df_all.head())

        # 数值列转为float
        for col in df_all.columns:
            if col not in ['城市', '年份', '月份']:
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

        # 保存
        output_file = os.path.join(self.output_dir, "附件2_subway_traffic.csv")
        df_all.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"轨道交通客运量数据处理完成，已保存到{output_file}")

        return df_all

    def detect_encoding(self, file_path):
        """
        使用chardet检测文件编码

        参数:
        file_path: 文件路径

        返回:
        检测到的编码
        """
        # 读取文件的前10000个字节来检测编码
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            # print(f"检测到文件 {file_path} 的编码为 {encoding}，置信度：{confidence:.2f}")
            return encoding

    def process_population_data(self):
        """
        处理人口普查数据（附件3）

        返回:
        处理后的人口数据DataFrame
        """
        print("开始处理人口普查数据...")

        # 人口普查数据路径
        pop_base_path = os.path.join(self.base_path, "附件3：我国省市两级第五、六、七次人口普查数据（包括年龄和性别）", "excel")

        # 处理省级数据
        province_dfs = []
        city_dfs = []

        # 处理五普、六普、七普数据
        census_names = ['五普', '六普', '七普']
        census_years = [2000, 2010, 2020]  # 对应的普查年份

        for census, year in zip(census_names, census_years):
            # 省级数据
            province_file = os.path.join(pop_base_path, f"【{census}】分年龄、性别的人口_省.xls")
            # 市级数据
            city_file = os.path.join(pop_base_path, f"【{census}】分年龄、性别的人口_地级市.xls")

            try:
                # 检测并读取省级数据
                try:
                    prov_encoding = self.detect_encoding(province_file)
                    # 读取前两行作为表头
                    prov_df = pd.read_excel(province_file, encoding=prov_encoding, header=[0, 1])
                except Exception as e:
                    prov_df = pd.read_excel(province_file, header=[0, 1])

                # 处理多级表头
                prov_df.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] for col in prov_df.columns]
                prov_df['普查年份'] = year
                prov_df['普查名称'] = census
                province_dfs.append(prov_df)

                # 检测并读取市级数据
                try:
                    city_encoding = self.detect_encoding(city_file)
                    # 读取前两行作为表头
                    city_df = pd.read_excel(city_file, encoding=city_encoding, header=[0, 1])
                except Exception as e:
                    city_df = pd.read_excel(city_file, header=[0, 1])

                # 处理多级表头
                city_df.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] for col in city_df.columns]
                city_df['普查年份'] = year
                city_df['普查名称'] = census
                city_dfs.append(city_df)

            except Exception as e:
                print(f"处理{census}人口数据时出错: {e}")

        # 合并所有普查的数据
        if province_dfs:
            province_combined = pd.concat(province_dfs, ignore_index=True)
            output_file = os.path.join(self.output_dir, "附件3_population_province.csv")
            province_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"省级人口数据处理完成，已保存到{output_file}")

        if city_dfs:
            city_combined = pd.concat(city_dfs, ignore_index=True)
            output_file = os.path.join(self.output_dir, "附件3_population_city.csv")
            city_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"市级人口数据处理完成，已保存到{output_file}")

        # 返回市级数据，因为主要分析城市
        return city_combined if city_dfs else None

    def process_population_density(self, year=2020):
        """
        处理人口密度数据（附件4）

        参数:
        year: 需要处理的年份，默认为2020年

        返回:
        处理后的人口密度数据
        """
        print(f"开始处理{year}年人口密度数据...")

        # 人口密度数据路径
        density_base_path = os.path.join(self.base_path, "附件4：全国人口密度分布",
                                        "中国人口密度公里格网栅格数据", f"china{year}")

        try:
            # 读取栅格数据
            # 注意：这里需要使用rasterio库读取.adf格式的栅格数据
            raster_file = os.path.join(density_base_path, "w001001.adf")

            with rasterio.open(raster_file) as src:
                # 读取栅格数据
                population_density = src.read(1)

                # 获取元数据
                meta = src.meta

                print(f"{year}年人口密度数据读取成功，形状为{population_density.shape}")

                # 这里可以进行进一步的处理，如裁剪感兴趣区域、重采样等

                # 保存处理后的数据
                # 由于栅格数据较大，这里只保存一些统计信息
                stats = {
                    'min': np.nanmin(population_density),
                    'max': np.nanmax(population_density),
                    'mean': np.nanmean(population_density),
                    'median': np.nanmedian(population_density),
                    'std': np.nanstd(population_density)
                }

                stats_df = pd.DataFrame([stats])
                stats_df['year'] = year

                output_file = os.path.join(self.output_dir, f"附件4_population_density_stats_{year}.csv")
                stats_df.to_csv(output_file, index=False)
                print(f"{year}年人口密度统计数据已保存到{output_file}")

                return population_density, meta

        except Exception as e:
            print(f"处理{year}年人口密度数据时出错: {e}")
            return None, None

    def process_ultra_marathon_data(self):
        """
        处理超级马拉松数据（附件11）

        返回:
        处理后的超级马拉松数据DataFrame
        """
        print("开始处理超级马拉松数据...")

        # 超级马拉松数据路径
        ultra_path = os.path.join(self.base_path, "附件11：超级马拉松跑的大数据集", "TWO_CENTURIES_OF_UM_RACES.csv")

        try:
            # 读取CSV文件（可能很大，使用分块读取）
            chunks = pd.read_csv(ultra_path, chunksize=100000)

            # 处理第一个块以获取列名
            first_chunk = next(chunks)
            processed_chunks = [first_chunk]

            # 处理剩余的块
            for chunk in chunks:
                processed_chunks.append(chunk)

            # 合并所有块
            df = pd.concat(processed_chunks, ignore_index=True)

            # 数据清洗和转换
            # 1. 处理日期格式
            date_columns = ['date', 'end_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            # 2. 提取年份和月份
            if 'date' in df.columns:
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month

            # 3. 计算比赛持续时间
            if 'date' in df.columns and 'end_date' in df.columns:
                df['duration_days'] = (df['end_date'] - df['date']).dt.days + 1

            # 保存处理后的数据（由于数据可能很大，这里只保存一个样本）
            sample_df = df.sample(n=min(10000, len(df)), random_state=42)
            output_file = os.path.join(self.output_dir, "附件11_ultra_marathon_sample.csv")
            sample_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"超级马拉松数据处理完成，已保存样本到{output_file}")

            # 保存完整数据
            output_file_full = os.path.join(self.output_dir, "附件11_ultra_marathon_full.csv")
            df.to_csv(output_file_full, index=False, encoding='utf-8-sig')
            print(f"超级马拉松完整数据已保存到{output_file_full}")

            return df

        except Exception as e:
            print(f"处理超级马拉松数据时出错: {e}")
            return None

    def process_xian_basic_data(self):
        """
        处理西安市基础数据（附件5中的住宿、餐饮、景点）
        """
        print("开始处理西安市基础数据...")

        # 1. 加载住宿设施数据
        hotel_file = os.path.join(self.base_path, "附件5：西安市基础数据", "西安市住宿服务数据.csv")
        try:
            hotel_df = pd.read_csv(hotel_file, encoding='utf-8')
            print(f"住宿设施数据加载成功，共 {len(hotel_df)} 条记录")
            # 假设住宿容量列名为'capacity'
            if 'capacity' not in hotel_df.columns:
                hotel_df['capacity'] = 100  # 默认值
            # 保存为CSV便于后续使用
            hotel_output = os.path.join(self.output_dir, "附件5_xian_hotels.csv")
            hotel_df.to_csv(hotel_output, index=False)
        except Exception as e:
            print(f"加载住宿设施数据时出错: {e}")
            hotel_df = None

        # 2. 加载餐饮设施数据
        restaurant_file = os.path.join(self.base_path, "附件5：西安市基础数据", "西安市餐饮数据.csv")
        try:
            restaurant_df = pd.read_csv(restaurant_file, encoding='utf-8')
            print(f"餐饮设施数据加载成功，共 {len(restaurant_df)} 条记录")
            # 保存为CSV便于后续使用
            restaurant_output = os.path.join(self.output_dir, "附件5_xian_restaurants.csv")
            restaurant_df.to_csv(restaurant_output, index=False)
        except Exception as e:
            print(f"加载餐饮设施数据时出错: {e}")
            restaurant_df = None

        # 3. 加载景点数据
        attraction_file = os.path.join(self.base_path, "附件5：西安市基础数据", "西安市风景名胜数据.csv")
        try:
            attraction_df = pd.read_csv(attraction_file, encoding='utf-8')
            print(f"景点数据加载成功，共 {len(attraction_df)} 条记录")
            # 保存为CSV便于后续使用
            attraction_output = os.path.join(self.output_dir, "附件5_xian_attractions.csv")
            attraction_df.to_csv(attraction_output, index=False)
        except Exception as e:
            print(f"加载景点数据时出错: {e}")
            attraction_df = None

        print("西安市基础数据处理完成！")

        return {
            'hotels': hotel_df,
            'restaurants': restaurant_df,
            'attractions': attraction_df
        }

    def process_shaanxi_terrain(self):
        """
        处理陕西省地形数据（附件6）
        """
        print("开始处理陕西省地形数据...")

        # 4. 加载地形数据（附件6）
        terrain_file = os.path.join(self.base_path, "附件6：陕西省12.5分辨率地形数据", "陕西WGS84.tif")
        try:
            with rasterio.open(terrain_file) as src:
                terrain_data = src.read(1)
                meta = src.meta
                print(f"地形数据加载成功，形状: {terrain_data.shape}")
        except Exception as e:
            print(f"加载地形数据时出错: {e}")
            terrain_data = None
            meta = None

        print("陕西省地形数据处理完成！")

        return {
            'terrain_data': terrain_data,
            'meta': meta
        }

    def process_xian_road_connections(self):
        """
        处理西安市道路连接信息（附件7）
        """
        print("开始处理西安市道路连接信息...")

        # 5. 加载道路数据（附件7）
        road_file = os.path.join(self.base_path, "附件7：2025年西安市道路数据", "路线连接信息.csv")
        try:
            road_df = pd.read_csv(road_file, encoding='gbk')
            print(f"道路连接信息加载成功，共 {len(road_df)} 条记录")
            # 保存为CSV便于后续使用
            road_output = os.path.join(self.output_dir, "附件7_xian_road_connections.csv")
            road_df.to_csv(road_output, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"加载道路连接信息时出错: {e}")
            road_df = None

        print("西安市道路连接信息处理完成！")

        return road_df

    def process_xian_bus_stations(self):
        """
        处理西安市公交站点数据（附件8）
        """
        print("开始处理西安市公交站点数据...")

        # 6. 加载公交站点数据（附件8）
        bus_file = os.path.join(self.base_path, "附件8：西安_2024年公交站点和线路数据", "公交站点（含经纬度）.xlsx")
        try:
            bus_df = pd.read_excel(bus_file)
            print(f"公交站点数据加载成功，共 {len(bus_df)} 条记录")
            # 重命名列以统一格式
            bus_df.columns = ['FID', 'name', 'line', 'name_st', '经度', '纬度']
            # 保存为CSV便于后续使用
            bus_output = os.path.join(self.output_dir, "附件8_xian_bus_stations.csv")
            bus_df.to_csv(bus_output, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"加载公交站点数据时出错: {e}")
            bus_df = None

        print("西安市公交站点数据处理完成！")

        return bus_df

    def process_xian_subway_stations(self):
        """
        处理西安市地铁站点数据（附件9）
        """
        print("开始处理西安市地铁站点数据...")

        # 7. 加载地铁站点数据（附件9）
        subway_file = os.path.join(self.base_path, "附件9：西安_2024年地铁数据", "地铁站点（含经纬度）.xlsx")
        try:
            subway_df = pd.read_excel(subway_file)
            print(f"地铁站点数据加载成功，共 {len(subway_df)} 条记录")
            # 重命名列以统一格式
            subway_df.columns = ['FID', 'name', '经度', '纬度']
            # 保存为CSV便于后续使用
            subway_output = os.path.join(self.output_dir, "附件9_xian_subway_stations.csv")
            subway_df.to_csv(subway_output, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"加载地铁站点数据时出错: {e}")
            subway_df = None

        print("西安市地铁站点数据处理完成！")

        return subway_df

    def run_single_attachment(self, attachment_num, **kwargs):
        """
        根据附件编号运行单个附件的预处理

        参数:
        attachment_num: 要处理的附件编号
        **kwargs: 附加参数

        返回:
        处理结果
        """
        try:
            if attachment_num == 1:
                years = kwargs.get('years', None)
                return self.run_attachment_1(years=years)
            elif attachment_num == 2:
                return self.run_attachment_2()
            elif attachment_num == 3:
                return self.run_attachment_3()
            elif attachment_num == 4:
                year = kwargs.get('year', 2020)
                return self.run_attachment_4(year=year)
            elif attachment_num == 5:
                return self.run_attachment_5()
            elif attachment_num == 6:
                return self.run_attachment_6()
            elif attachment_num == 7:
                return self.run_attachment_7()
            elif attachment_num == 8:
                return self.run_attachment_8()
            elif attachment_num == 9:
                return self.run_attachment_9()
            elif attachment_num == 11:
                return self.run_attachment_11()
            elif attachment_num == 12:
                return self.run_attachment_12()
            else:
                raise ValueError(f"不支持的附件编号: {attachment_num}")
        except Exception as e:
            print(f"处理附件{attachment_num}时发生错误: {str(e)}")
            return None

    def run_attachment_1(self, years=None):
        """
        单独处理附件1：中国气象数据

        参数:
        years: 需要处理的年份列表，如果为None则处理默认年份

        返回:
        处理后的气象数据DataFrame
        """
        print("\n--- 开始处理附件1：中国气象数据 ---")
        if years is None:
            years = ['2020', '2021', '2022', '2023']  # 默认年份
        return self.process_meteorological_data(years=years)

    def run_attachment_2(self):
        """
        单独处理附件2：轨道交通客运量数据

        返回:
        处理后的轨道交通数据DataFrame
        """
        print("\n--- 开始处理附件2：轨道交通客运量数据 ---")
        return self.process_subway_data()

    def run_attachment_3(self):
        """
        单独处理附件3：人口普查数据

        返回:
        处理后的人口普查数据DataFrame
        """
        print("\n--- 开始处理附件3：人口普查数据 ---")
        return self.process_population_data()

    def run_attachment_4(self, year=2020):
        """
        单独处理附件4：人口密度数据

        参数:
        year: 要处理的年份，默认为2020

        返回:
        处理后的人口密度数据
        """
        print(f"\n--- 开始处理附件4：人口密度数据 ({year}年) ---")
        return self.process_population_density(year=year)
    def run_attachment_5(self):
        """
        单独处理附件5：西安市住宿、餐饮、景点数据

        返回:
        处理后的西安基础数据字典
        """
        print("\n--- 开始处理附件5：西安市基础数据 ---")
        return self.process_xian_basic_data()

    def run_attachment_6(self):
        """
        单独处理附件6：陕西省地形数据

        返回:
        处理后的地形数据
        """
        print("\n--- 开始处理附件6：陕西省地形数据 ---")
        return self.process_shaanxi_terrain()

    def run_attachment_7(self):
        """
        单独处理附件7：西安市道路连接信息

        返回:
        处理后的道路连接数据DataFrame
        """
        print("\n--- 开始处理附件7：西安市道路连接信息 ---")
        return self.process_xian_road_connections()

    def run_attachment_8(self):
        """
        单独处理附件8：西安市公交站点数据

        返回:
        处理后的公交站点数据DataFrame
        """
        print("\n--- 开始处理附件8：西安市公交站点数据 ---")
        return self.process_xian_bus_stations()

    def run_attachment_9(self):
        """
        单独处理附件9：西安市地铁站点数据

        返回:
        处理后的地铁站点数据DataFrame
        """
        print("\n--- 开始处理附件9：西安市地铁站点数据 ---")
        return self.process_xian_subway_stations()

    def run_attachment_11(self):
        """
        单独处理附件11：超级马拉松数据

        返回:
        处理后的超级马拉松数据DataFrame
        """
        print("\n--- 开始处理附件11：超级马拉松数据 ---")
        return self.process_ultra_marathon_data()

    def run_attachment_12(self):
        """
        单独处理附件12：马拉松赛历数据

        返回:
        处理后的马拉松赛历数据DataFrame
        """
        print("\n--- 开始处理附件12：马拉松赛历数据 ---")
        return self.process_marathon_history()

    def run_all_preprocessing(self):
        """
        运行所有预处理步骤，并添加数据清洗环节
        """
        # 创建处理后的数据目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 分别处理每个附件
        print("\n=== 开始批量处理所有附件 ===")

        # 处理附件1：气象数据
        print("\n--- 处理附件1：中国气象数据 ---")
        recent_years = ['2020', '2021', '2022', '2023']
        meteo_data = self.run_attachment_1(years=recent_years)

        # 处理附件2：轨道交通客运量数据
        print("\n--- 处理附件2：轨道交通客运量数据 ---")
        subway_data = self.run_attachment_2()

        # 处理附件3：人口普查数据
        print("\n--- 处理附件3：人口普查数据 ---")
        population_data = self.run_attachment_3()

        # 处理附件4：人口密度数据（2020年）
        print("\n--- 处理附件4：人口密度数据 (2020年) ---")
        density_data, meta = self.run_attachment_4(year=2020)

        # 处理附件5：西安市基础数据
        print("\n--- 处理附件5：西安市基础数据 ---")
        xian_basic_data = self.run_attachment_5()

        # 处理附件6：陕西省地形数据
        print("\n--- 处理附件6：陕西省地形数据 ---")
        terrain_data = self.run_attachment_6()

        # 处理附件7：西安市道路连接信息
        print("\n--- 处理附件7：西安市道路连接信息 ---")
        road_data = self.run_attachment_7()

        # 处理附件8：西安市公交站点数据
        print("\n--- 处理附件8：西安市公交站点数据 ---")
        bus_data = self.run_attachment_8()

        # 处理附件9：西安市地铁站点数据
        print("\n--- 处理附件9：西安市地铁站点数据 ---")
        subway_station_data = self.run_attachment_9()

        # 处理附件11：超级马拉松数据
        print("\n--- 处理附件11：超级马拉松数据 ---")
        ultra_data = self.run_attachment_11()

        # 处理附件12：马拉松赛历数据
        print("\n--- 处理附件12：马拉松赛历数据 ---")
        marathon_data = self.run_attachment_12()

        print("\n=== 所有数据预处理完成 ===")

        # 创建数据清洗器并进行数据清洗
        print("\n开始进行数据清洗...")
        cleaner = DataCleaner(data_dir=self.output_dir)
        cleaned_data = cleaner.clean_all_data()
        print("数据清洗完成！")

        # 返回所有处理后的数据
        return {
            'meteorological': meteo_data,
            'marathon_history': marathon_data,
            'subway_traffic': subway_data,
            'population': population_data,
            'population_density': density_data,
            'xian_basic': xian_basic_data,
            'terrain': terrain_data,
            'road_connections': road_data,
            'bus_stations': bus_data,
            'subway_stations': subway_station_data,
            'ultra_marathon': ultra_data,
            'cleaned_data': cleaned_data
        }


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='数据预处理工具')
    parser.add_argument('--base_path', type=str, default="附件",
                      help='附件数据的基础路径')
    parser.add_argument('--attachment', type=int, default=None,
                      help='要处理的附件编号（1-4, 5, 6, 7, 8, 9, 11, 12），不指定则处理所有附件')
    parser.add_argument('--years', type=str, nargs='+',
                      help='要处理的年份列表（仅用于附件1），例如：2020 2021 2022')
    parser.add_argument('--year', type=int, default=2020,
                      help='要处理的年份（仅用于附件4），例如：2020')

    args = parser.parse_args()

    # 创建数据预处理器
    preprocessor = DataPreprocessor(base_path=args.base_path)

    # 如果没有指定附件，则处理所有附件
    if args.attachment is None:
        print("未指定附件编号，将处理所有附件...")
        processed_data = preprocessor.run_all_preprocessing()
    else:
        print(f"\n开始处理附件{args.attachment}...")
        processed_data = preprocessor.run_single_attachment(
            attachment_num=args.attachment,
            years=args.years,
            year=args.year
        )

    print("\n数据预处理完成！")
