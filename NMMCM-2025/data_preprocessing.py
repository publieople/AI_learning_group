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
import argparse
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
        返回：标准化后的DataFrame，字段包括：城市、年份、月份、客运量（万人次）等所有可用数据
        """
        print("开始处理轨道交通客运量数据...")
        import re
        all_data = []
        years = ['2021', '2022', '2023', '2024']
        base_path = os.path.join(self.base_path, "附件2：2021-2024年我国主要城市逐月轨道交通客运量数据")

        # 定义所有可能的列名模式
        column_patterns = {
            '城市': ['城市', '地区', '城市名称', '城市名'],
            '客运量': ['客运量', '客运量（万人次）', '客运量(万人次)', '客运量（万人）', '客运量(万人)'],
            '序号': ['序号', '编号', 'No.', 'NO.'],
            '线路数': ['线路数', '线路数量', '线路条数'],
            '运营里程': ['运营里程', '运营里程（公里）', '运营里程(公里)'],
            '车站数': ['车站数', '车站数量', '站点数', '站点数量'],
            '日均客运量': ['日均客运量', '日均客运量（万人次）', '日均客运量(万人次)'],
            '最高日客运量': ['最高日客运量', '最高日客运量（万人次）', '最高日客运量(万人次)'],
            '最低日客运量': ['最低日客运量', '最低日客运量（万人次）', '最低日客运量(万人次)']
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
                        row_str = ' '.join(str(val).lower() for val in row if pd.notna(val))
                        if '序号' in row_str:
                            title_row = idx
                            break

                    if title_row is None:
                        print(f"警告：在{sheet_name}中未找到标题行")
                        continue

                    # 获取标题行
                    headers = df.iloc[title_row]

                    # 创建列映射
                    col_map = {}
                    for idx, header in enumerate(headers):
                        header_str = str(header).lower()
                        for col_name, patterns in column_patterns.items():
                            if any(pattern in header_str for pattern in patterns):
                                col_map[col_name] = idx
                                break

                    # 验证必要的列是否存在
                    if '城市' not in col_map or '客运量' not in col_map:
                        print(f"警告：{sheet_name}中缺少必要的列（城市或客运量）")
                        continue

                    # 从标题行之后开始处理数据，直到遇到总计行
                    for idx in range(title_row + 1, len(df)):
                        row = df.iloc[idx]
                        city = str(row[col_map['城市']]).strip()

                        # 如果遇到总计行，结束当前sheet的处理
                        if '总' in city or city == '' or city == 'nan':
                            break

                        # 创建数据字典，包含所有可能的字段
                        data_row = {
                            '城市': city,
                            '年份': int(year),
                            '月份': None,
                            '客运量_万人次': None,
                            '线路数': None,
                            '运营里程_公里': None,
                            '车站数': None,
                            '日均客运量_万人次': None,
                            '最高日客运量_万人次': None,
                            '最低日客运量_万人次': None
                        }

                        # 解析月份
                        m = re.search(r'(\d+)', sheet_name)
                        if m:
                            data_row['月份'] = int(m.group(1))

                        # 处理所有可用的数值列
                        numeric_columns = {
                            '客运量': '客运量_万人次',
                            '线路数': '线路数',
                            '运营里程': '运营里程_公里',
                            '车站数': '车站数',
                            '日均客运量': '日均客运量_万人次',
                            '最高日客运量': '最高日客运量_万人次',
                            '最低日客运量': '最低日客运量_万人次'
                        }

                        for col_name, data_col in numeric_columns.items():
                            if col_name in col_map:
                                try:
                                    value_str = str(row[col_map[col_name]]).strip()
                                    value_str = value_str.replace(',', '').replace(' ', '')
                                    if value_str and value_str != 'nan':
                                        data_row[data_col] = float(value_str)
                                except Exception as e:
                                    print(f"警告：无法转换{col_name}数据 '{row[col_map[col_name]]}' - {str(e)}")

                        # 只添加有效的数据行（至少包含城市和客运量）
                        if city and data_row['客运量_万人次'] is not None:
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

        # 数据清洗
        # 1. 确保所有数值列都是浮点数类型
        numeric_columns = [col for col in df_all.columns if col not in ['城市', '年份', '月份']]
        for col in numeric_columns:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

        # 2. 删除完全为空的行
        df_all = df_all.dropna(how='all', subset=numeric_columns)

        # 保存
        output_file = os.path.join(self.output_dir, "附件2_subway_traffic.csv")
        df_all.to_csv(output_file, index=False, encoding='utf-8-sig')
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
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='数据预处理工具')
    parser.add_argument('--base_path', type=str, default="附件",
                      help='附件数据的基础路径')
    parser.add_argument('--attachments', type=str, nargs='+',
                      help='要处理的附件编号列表，例如：1 2 3')
    parser.add_argument('--years', type=str, nargs='+',
                      help='要处理的年份列表（仅用于附件1），例如：2020 2021 2022')

    args = parser.parse_args()

    # 创建数据预处理器
    preprocessor = DataPreprocessor(base_path=args.base_path)

    # 如果没有指定附件，则处理所有附件
    if not args.attachments:
        print("未指定附件编号，将处理所有附件...")
        processed_data = preprocessor.run_all_preprocessing()
    else:
        processed_data = {}
        for attachment in args.attachments:
            print(f"\n开始处理附件{attachment}...")
            if attachment == "1":
                processed_data['meteorological'] = preprocessor.process_meteorological_data(years=args.years)
            elif attachment == "2":
                processed_data['subway_traffic'] = preprocessor.process_subway_data()
            elif attachment == "3":
                processed_data['population'] = preprocessor.process_population_data()
            elif attachment == "4":
                processed_data['population_density'], _ = preprocessor.process_population_density()
            elif attachment == "11":
                processed_data['ultra_marathon'] = preprocessor.process_ultra_marathon_data()
            elif attachment == "12":
                processed_data['marathon_history'] = preprocessor.process_marathon_history()
            else:
                print(f"警告：未知的附件编号 {attachment}")

    print("\n数据预处理完成！")
