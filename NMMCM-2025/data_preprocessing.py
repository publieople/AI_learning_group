#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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

        # 如果未指定年份，获取所有年份文件夹
        if years is None:
            years = [d for d in os.listdir(meteo_base_path)
                     if os.path.isdir(os.path.join(meteo_base_path, d)) and 'china_isd_lite' in d]
            years = [y.split('_')[-1] for y in years]

        all_data = []

        for year in years:
            print(f"处理{year}年的气象数据...")
            year_path = os.path.join(meteo_base_path, f"china_isd_lite_{year}")

            # 获取该年份下所有站点文件
            station_files = glob.glob(os.path.join(year_path, "*"))

            # 如果指定了城市，只处理这些城市的站点
            if cities is not None:
                # 这里需要站点ID与城市的映射关系，暂时跳过筛选
                pass

            for station_file in station_files:
                try:
                    # 提取站点ID
                    station_id = os.path.basename(station_file).split('-')[0]

                    # 读取数据
                    with open(station_file, 'r') as f:
                        lines = f.readlines()

                    # 解析数据
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 9:  # 确保有足够的数据列
                            year, month, day, hour = parts[0], parts[1], parts[2], parts[3]

                            # 温度、露点、气压、风向、风速、云量、降水量
                            temp = float(parts[4]) / 10.0 if parts[4] != '-9999' else np.nan  # 温度，除以10转为摄氏度
                            dewp = float(parts[5]) / 10.0 if parts[5] != '-9999' else np.nan  # 露点，除以10转为摄氏度
                            pressure = float(parts[6]) / 10.0 if parts[6] != '-9999' else np.nan  # 气压，除以10转为百帕
                            wind_dir = float(parts[7]) if parts[7] != '-9999' else np.nan  # 风向
                            wind_speed = float(parts[8]) / 10.0 if parts[8] != '-9999' else np.nan  # 风速，除以10转为米/秒

                            # 云量和降水量可能不存在
                            cloud = float(parts[9]) if len(parts) > 9 and parts[9] != '-9999' else np.nan
                            precip = float(parts[10]) / 10.0 if len(parts) > 10 and parts[10] != '-9999' else np.nan  # 降水量，除以10转为毫米

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
                except Exception as e:
                    print(f"处理文件{station_file}时出错: {e}")

        # 转换为DataFrame
        df = pd.DataFrame(all_data)
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 保存处理后的数据
        output_file = os.path.join(self.output_dir, "附件1_meteorological_data.csv")
        df.to_csv(output_file, index=False)
        print(f"气象数据处理完成，已保存到{output_file}")

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
            if '比赛日期' in df.columns:
                df['比赛日期'] = pd.to_datetime(df['比赛日期'], errors='coerce')

            # 2. 处理数值型数据
            numeric_columns = ['报名人数', '完赛人数', '报名费']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 3. 计算完赛率
            if '报名人数' in df.columns and '完赛人数' in df.columns:
                df['完赛率'] = df['完赛人数'] / df['报名人数']

            # 4. 提取年份和月份
            if '比赛日期' in df.columns:
                df['年份'] = df['比赛日期'].dt.year
                df['月份'] = df['比赛日期'].dt.month

            # 保存处理后的数据
            output_file = os.path.join(self.output_dir, "附件12_marathon_history.csv")
            df.to_csv(output_file, index=False)
            print(f"马拉松赛历数据处理完成，已保存到{output_file}")

            return df

        except Exception as e:
            print(f"处理马拉松赛历数据时出错: {e}")
            return None

    def process_subway_data(self):
        """
        兼容处理2021-2024年轨道交通客运量数据（附件2）
        返回：标准化后的DataFrame，字段包括：城市、年份、月份、客运量（万人次）
        """
        print("开始处理轨道交通客运量数据...")
        import re
        all_data = []
        years = ['2021', '2022', '2023', '2024']
        base_path = os.path.join(self.base_path, "附件2：2021-2024年我国主要城市逐月轨道交通客运量数据")
        for year in years:
            file_path = os.path.join(base_path, f"{year}.xlsx")
            if not os.path.exists(file_path):
                print(f"未找到{file_path}")
                continue
            xls = pd.ExcelFile(file_path)
            # 2021/2022年：每月一个sheet，2023/2024年：每月一个sheet，2023年3月起多一列
            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
                except Exception as e:
                    print(f"读取{year}年{sheet_name}失败: {e}")
                    continue
                # 标题行模糊匹配
                col_map = {}
                for col in df.columns:
                    if re.search("城市", col):
                        col_map['城市'] = col
                    elif re.search("客运量", col):
                        col_map['客运量'] = col
                    elif re.search("序", col):
                        col_map['序号'] = col
                # 跳过无效sheet
                if not col_map or '城市' not in col_map or '客运量' not in col_map:
                    continue
                # 只保留有效数据行
                for _, row in df.iterrows():
                    city = str(row[col_map['城市']]).strip()
                    if '总' in city or city == '' or city == 'nan':
                        continue
                    try:
                        value = float(str(row[col_map['客运量']]).replace(',', '').replace(' ', ''))
                    except:
                        value = None
                    # 解析月份
                    month = None
                    # sheet名可能为"6月""2023年6月"等
                    m = re.search(r'(\d+)', sheet_name)
                    if m:
                        month = int(m.group(1))
                    all_data.append({
                        '城市': city,
                        '年份': int(year),
                        '月份': month,
                        '客运量_万人次': value
                    })
        # 合并所有年份
        df_all = pd.DataFrame(all_data)
        # 去除无效行
        df_all = df_all.dropna(subset=['城市', '年份', '月份', '客运量_万人次'])
        # 保存
        output_file = os.path.join(self.output_dir, "附件2_subway_traffic.csv")
        df_all.to_csv(output_file, index=False)
        print(f"轨道交通客运量数据处理完成，已保存到{output_file}")
        return df_all

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
                # 读取省级数据
                prov_df = pd.read_excel(province_file)
                prov_df['普查年份'] = year
                prov_df['普查名称'] = census
                province_dfs.append(prov_df)

                # 读取市级数据
                city_df = pd.read_excel(city_file)
                city_df['普查年份'] = year
                city_df['普查名称'] = census
                city_dfs.append(city_df)

            except Exception as e:
                print(f"处理{census}人口数据时出错: {e}")

        # 合并所有普查的数据
        if province_dfs:
            province_combined = pd.concat(province_dfs, ignore_index=True)
            output_file = os.path.join(self.output_dir, "附件3_population_province.csv")
            province_combined.to_csv(output_file, index=False)
            print(f"省级人口数据处理完成，已保存到{output_file}")

        if city_dfs:
            city_combined = pd.concat(city_dfs, ignore_index=True)
            output_file = os.path.join(self.output_dir, "附件3_population_city.csv")
            city_combined.to_csv(output_file, index=False)
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
            sample_df.to_csv(output_file, index=False)
            print(f"超级马拉松数据处理完成，已保存样本到{output_file}")

            # 保存完整数据
            output_file_full = os.path.join(self.output_dir, "附件11_ultra_marathon_full.csv")
            df.to_csv(output_file_full, index=False)
            print(f"超级马拉松完整数据已保存到{output_file_full}")

            return df

        except Exception as e:
            print(f"处理超级马拉松数据时出错: {e}")
            return None

    def run_all_preprocessing(self):
        """
        运行所有预处理步骤
        """
        # 创建处理后的数据目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 处理气象数据（选择近几年的数据）
        recent_years = ['2020', '2021', '2022', '2023']
        meteo_data = self.process_meteorological_data(years=recent_years)

        # 处理轨道交通客运量数据
        subway_data = self.process_subway_data()

        # 处理人口普查数据
        population_data = self.process_population_data()

        # 处理人口密度数据（2020年）
        density_data, meta = self.process_population_density(year=2020)

        # 处理超级马拉松数据
        ultra_data = self.process_ultra_marathon_data()

        # 处理马拉松赛历数据
        marathon_data = self.process_marathon_history()

        print("所有数据预处理完成！")

        # 返回所有处理后的数据
        return {
            'meteorological': meteo_data,
            'marathon_history': marathon_data,
            'subway_traffic': subway_data,
            'population': population_data,
            'population_density': density_data,
            'ultra_marathon': ultra_data
        }


if __name__ == "__main__":
    # 设置附件基础路径
    base_path = "附件"

    # 创建数据预处理器
    preprocessor = DataPreprocessor(base_path=base_path)

    # 运行所有预处理步骤
    processed_data = preprocessor.run_all_preprocessing()
