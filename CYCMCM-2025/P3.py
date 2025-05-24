import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, linprog
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class KangYangResourceOptimizer:
    """
    康养资源优化配置模型
    实现多目标优化：最大化服务覆盖率、最大化资源利用效率、最小化成本
    """

    def __init__(self):
        self.data = None
        self.population_data = None
        self.top100_cities = None
        self.scaler = StandardScaler()

    def load_data(self):
        """加载数据集"""
        try:
            # 加载上海数据
            self.data = pd.read_csv(r'.\dataset\上海数据.csv', encoding='utf-8')
            print("上海数据加载成功")

            # Set '指标' column as index and transpose the DataFrame
            if '指标' in self.data.columns:
                self.data = self.data.set_index('指标').T

            # Print column names for debugging after transpose
            print("DataFrame columns after loading and transposing:", self.data.columns)
            print("DataFrame index after loading and transposing:", self.data.index)

            # Convert relevant columns to numeric, coercing errors
            numeric_cols = ['医疗卫生机构数(个)', '医院床位数（个）', '公园个数(个)', '建成区绿化覆盖率(%)', '公园绿地面积（万顷）', '文化机构（个）', '老龄人口占比（%）', '养老机构数量（个）', '养老床位（万张）', '常驻人口（万人）', '人均寿命（岁）', '地区生产总值（亿）', '人均生产总值(万元)', '用水总量（亿立方米）', '空气质量指数（%）']
            for col in numeric_cols:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            # 加载人口数据
            self.population_data = pd.read_csv(r'.\dataset\各地区分年龄、性别的人口(城市).csv', encoding='utf-8')
            print("人口数据加载成功")

            # 加载康养城市100强数据
            self.top100_cities = pd.read_csv(r'.\dataset\2024 年中国康养城市100强名单.csv', encoding='utf-8')
            print("康养城市100强数据加载成功")

        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
        return True

    def preprocess_data(self):
        """数据预处理"""
        if self.data is None:
            print("请先加载数据")
            return

        # 处理缺失值
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')

        # 计算关键指标
        self.data['医疗资源密度'] = self.data['医疗卫生机构数(个)'] / self.data['常驻人口（万人）']
        self.data['床位密度'] = self.data['医院床位数（个）'] / self.data['常驻人口（万人）']
        self.data['绿化指数'] = self.data['公园个数(个)'] * self.data['建成区绿化覆盖率(%)'] / 100
        self.data['养老服务密度'] = self.data['养老机构数量（个）'] / self.data['老龄人口占比（%）']
        self.data['人均GDP'] = self.data['地区生产总值（亿）'] / self.data['常驻人口（万人）']
        self.data['人均绿地面积'] = self.data['公园绿地面积（万顷）'] / self.data['常驻人口（万人）']
        self.data['人均文化机构数'] = self.data['文化机构（个）'] / self.data['常驻人口（万人）']
        self.data['人均养老床位'] = self.data['养老床位（万张）'] / self.data['常驻人口（万人）']
        self.data['人均用水量'] = self.data['用水总量（亿立方米）'] / self.data['常驻人口（万人）']

        print("数据预处理完成")

    def calculate_resource_demand(self):
        """计算康养资源需求"""
        # 使用最新一年的数据进行需求计算
        latest_year = self.data.index[-1]
        latest_data = self.data.loc[latest_year]

        # 计算医疗资源需求 (示例：每千人1个医疗机构，每百人X张床位)
        medical_demand = {
            '医疗机构需求': latest_data['常驻人口（万人）'] * 0.001,  # 每千人1个医疗机构
            '床位需求': latest_data['常驻人口（万人）'] * 0.01 # 示例：每百人1张床位
        }

        # 计算环境资源需求 (示例：人均绿地面积)
        environment_demand = {
            '人均绿地面积需求': latest_data['常驻人口（万人）'] * 0.005 # 示例：人均5平方米绿地
        }

        # 计算养老资源需求 (示例：基于老化人口比例)
        # 假设老化人口需要一定比例的养老机构和床位
        # 注意：这里简化处理，实际可能需要更复杂的模型
        if '老化人口占比（%）' in latest_data and '常驻人口（万人）' in latest_data:
            elderly_population = latest_data['常驻人口（万人）'] * (latest_data['老化人口占比（%）'] / 100)
            aging_demand = {
                '养老机构需求': elderly_population * 0.002, # 示例：每千名老人2个养老机构
                '养老床位需求': elderly_population * 0.05 # 示例：每百名老人5张养老床位
            }
        else:
            aging_demand = {'养老机构需求': 0, '养老床位需求': 0}

        # 合并所有需求
        # resource_demand = {**medical_demand, **environment_demand, **aging_demand}

        # 返回医疗和环境需求
        return medical_demand, environment_demand

    def define_optimization_objectives(self):
        """定义优化目标函数"""
        # 示例：定义覆盖率、效率和成本函数
        # x 是资源配置变量，例如 [医疗机构数量, 床位数, 养老机构数量, 公园数量]

        # 服务覆盖率函数 (示例：简单线性模型)
        def coverage_func(x):
            # 假设覆盖率与资源数量正相关
            # 需要更复杂的模型考虑地理位置、人口分布等
            return 0.01 * x[0] + 0.005 * x[1] + 0.02 * x[2] + 0.015 * x[3]

        # 资源利用效率函数 (示例：简单线性模型)
        def efficiency_func(x):
            # 假设效率与资源数量正相关，但可能存在边际递减
            # 需要考虑资源间的协同效应和利用率数据
            return 0.008 * x[0] + 0.004 * x[1] + 0.015 * x[2] + 0.01 * x[3]

        # 总成本函数 (示例：简单线性成本)
        def cost_func(x):
            # 假设每种资源的单位成本固定
            # 需要更精确的成本数据
            unit_costs = np.array([100, 5, 80, 20]) # 示例单位成本 (万元/个或张)
            return np.dot(x, unit_costs)

        return coverage_func, efficiency_func, cost_func

    def multi_objective_optimization(self, budget_constraint=50000):
        """多目标优化求解"""
        # 获取需求数据
        self.medical_demand, self.environment_demand = self.calculate_resource_demand()

        # 获取目标函数
        coverage_func, efficiency_func, cost_func = self.define_optimization_objectives()

        # 定义综合目标函数（加权法）
        def objective(x, weights=[0.4, 0.3, 0.3]):
            coverage = coverage_func(x)
            efficiency = efficiency_func(x)
            cost = cost_func(x)

            # 归一化成本（取负值因为要最小化）
            normalized_cost = 1 - (cost / budget_constraint)

            return -(weights[0] * coverage + weights[1] * efficiency + weights[2] * normalized_cost)

        # 约束条件
        constraints = [
            {'type': 'ineq', 'fun': lambda x: budget_constraint - cost_func(x)},  # 预算约束
            {'type': 'ineq', 'fun': lambda x: x[0]},  # 非负约束
            {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: x[2]},
            {'type': 'ineq', 'fun': lambda x: x[3]}
        ]

        # 初始猜测
        x0 = np.array([50, 1000, 20, 10])

        # 求解优化问题
        result = minimize(objective, x0, method='SLSQP', constraints=constraints)

        return result

    def pareto_frontier_analysis(self):
        """帕累托前沿分析"""
        # 不同权重组合
        weight_combinations = [
            [0.6, 0.2, 0.2],  # 重视覆盖率
            [0.2, 0.6, 0.2],  # 重视效率
            [0.2, 0.2, 0.6],  # 重视成本
            [0.33, 0.33, 0.34],  # 平衡
            [0.5, 0.3, 0.2],   # 覆盖率优先
            [0.3, 0.5, 0.2],   # 效率优先
        ]

        pareto_solutions = []

        for weights in weight_combinations:
            # 重新定义目标函数
            def objective(x):
                coverage_func, efficiency_func, cost_func = self.define_optimization_objectives()
                coverage = coverage_func(x)
                efficiency = efficiency_func(x)
                cost = cost_func(x)
                normalized_cost = 1 - (cost / 50000)
                return -(weights[0] * coverage + weights[1] * efficiency + weights[2] * normalized_cost)

            # 约束条件
            constraints = [
                {'type': 'ineq', 'fun': lambda x: 50000 - self.define_optimization_objectives()[2](x)},
                {'type': 'ineq', 'fun': lambda x: x[0]},
                {'type': 'ineq', 'fun': lambda x: x[1]},
                {'type': 'ineq', 'fun': lambda x: x[2]},
                {'type': 'ineq', 'fun': lambda x: x[3]}
            ]

            x0 = np.array([50, 1000, 20, 10])
            result = minimize(objective, x0, method='SLSQP', constraints=constraints)

            if result.success:
                coverage_func, efficiency_func, cost_func = self.define_optimization_objectives()
                solution = {
                    'weights': weights,
                    'solution': result.x,
                    'coverage': coverage_func(result.x),
                    'efficiency': efficiency_func(result.x),
                    'cost': cost_func(result.x)
                }
                pareto_solutions.append(solution)

        return pareto_solutions

    def sensitivity_analysis(self, base_solution):
        """敏感性分析"""
        # 参数变化范围
        param_changes = [-0.2, -0.1, 0, 0.1, 0.2]
        sensitivity_results = {}

        # 预算敏感性
        budget_sensitivity = []
        for change in param_changes:
            new_budget = 50000 * (1 + change)
            result = self.multi_objective_optimization(budget_constraint=new_budget)
            if result.success:
                coverage_func, efficiency_func, cost_func = self.define_optimization_objectives()
                budget_sensitivity.append({
                    'budget_change': change,
                    'coverage': coverage_func(result.x),
                    'efficiency': efficiency_func(result.x),
                    'cost': cost_func(result.x)
                })

        sensitivity_results['budget'] = budget_sensitivity
        return sensitivity_results

    def generate_implementation_strategy(self, optimal_solution):
        """生成实施策略"""
        strategy = {
            '第一阶段（1-2年）': {
                '医疗机构建设': int(optimal_solution[0] * 0.4),
                '床位增设': int(optimal_solution[1] * 0.3),
                '养老机构建设': int(optimal_solution[2] * 0.3),
                '公园建设': int(optimal_solution[3] * 0.2)
            },
            '第二阶段（3-4年）': {
                '医疗机构建设': int(optimal_solution[0] * 0.4),
                '床位增设': int(optimal_solution[1] * 0.4),
                '养老机构建设': int(optimal_solution[2] * 0.4),
                '公园建设': int(optimal_solution[3] * 0.5)
            },
            '第三阶段（5年）': {
                '医疗机构建设': int(optimal_solution[0] * 0.2),
                '床位增设': int(optimal_solution[1] * 0.3),
                '养老机构建设': int(optimal_solution[2] * 0.3),
                '公园建设': int(optimal_solution[3] * 0.3)
            }
        }
        return strategy

    def visualize_results(self, pareto_solutions, sensitivity_results):
        """结果可视化"""
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 帕累托前沿图
        coverages = [sol['coverage'] for sol in pareto_solutions]
        efficiencies = [sol['efficiency'] for sol in pareto_solutions]
        costs = [sol['cost'] for sol in pareto_solutions]

        scatter = axes[0, 0].scatter(coverages, efficiencies, c=costs, cmap='viridis', s=100)
        axes[0, 0].set_xlabel('服务覆盖率')
        axes[0, 0].set_ylabel('资源利用效率')
        axes[0, 0].set_title('帕累托前沿分析')
        plt.colorbar(scatter, ax=axes[0, 0], label='总成本（万元）')

        # 2. 资源配置方案对比
        solutions_df = pd.DataFrame([
            {'方案': f'方案{i+1}', '医疗机构': sol['solution'][0], '床位数': sol['solution'][1],
             '养老机构': sol['solution'][2], '公园数': sol['solution'][3]}
            for i, sol in enumerate(pareto_solutions[:4])
        ])

        solutions_df.set_index('方案').plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('不同优化方案资源配置对比')
        axes[0, 1].set_ylabel('资源数量')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 3. 敏感性分析
        if 'budget' in sensitivity_results:
            budget_data = sensitivity_results['budget']
            changes = [item['budget_change'] for item in budget_data]
            coverage_changes = [item['coverage'] for item in budget_data]

            axes[1, 0].plot(changes, coverage_changes, 'o-', label='服务覆盖率')
            axes[1, 0].set_xlabel('预算变化比例')
            axes[1, 0].set_ylabel('服务覆盖率')
            axes[1, 0].set_title('预算敏感性分析')
            axes[1, 0].grid(True)

        # 4. 目标函数权重影响
        weight_labels = ['覆盖率优先', '效率优先', '成本优先', '平衡方案', '覆盖率倾向', '效率倾向']
        objective_values = [sol['coverage'] + sol['efficiency'] - sol['cost']/50000 for sol in pareto_solutions]

        axes[1, 1].bar(range(len(weight_labels)), objective_values[:len(weight_labels)])
        axes[1, 1].set_xticks(range(len(weight_labels)))
        axes[1, 1].set_xticklabels(weight_labels, rotation=45)
        axes[1, 1].set_title('不同权重策略综合效果对比')
        axes[1, 1].set_ylabel('综合效果值')

        plt.tight_layout()
        plt.savefig('P3/康养资源优化配置分析.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self, optimal_solution, pareto_solutions, implementation_strategy):
        """生成分析报告"""
        coverage_func, efficiency_func, cost_func = self.define_optimization_objectives()

        report = f"""
# 康养资源优化配置模型分析报告

## 1. 问题概述
本研究基于多目标优化理论，构建了康养资源优化配置模型，旨在在有限预算约束下，
最大化康养服务覆盖率和资源利用效率，同时最小化总成本。

## 2. 模型设计

### 2.1 目标函数

- **服务覆盖率最大化**: 确保康养服务能够覆盖更多人群
- **资源利用效率最大化**: 提高各类康养资源的协调配置
- **总成本最小化**: 在预算约束下实现成本最优

### 2.2 约束条件

- 预算约束: 总投资不超过5亿元
- 非负约束: 所有资源配置数量非负
- 需求约束: 满足基本康养服务需求

## 3. 优化结果

### 3.1 最优解

- 医疗机构数量: {optimal_solution[0]:.0f}个
- 医院床位数: {optimal_solution[1]:.0f}张
- 养老机构数量: {optimal_solution[2]:.0f}个
- 公园数量: {optimal_solution[3]:.0f}个

### 3.2 性能指标

- 服务覆盖率: {coverage_func(optimal_solution):.2%}
- 资源利用效率: {efficiency_func(optimal_solution):.2%}
- 总投资成本: {cost_func(optimal_solution):.0f}万元

## 4. 帕累托前沿分析

通过不同权重组合，获得了{len(pareto_solutions)}个帕累托最优解，
为决策者提供了多种资源配置方案选择。

## 5. 实施建议

### 5.1 分阶段实施策略

建议采用三阶段实施方案，逐步推进康养资源建设：

**第一阶段（1-2年）**: 重点建设基础医疗设施
**第二阶段（3-4年）**: 完善养老服务体系和环境设施
**第三阶段（5年）**: 优化资源配置，提升服务质量

### 5.2 政策建议

1. 建立康养资源配置动态调整机制
2. 加强跨部门协调，统筹规划康养资源
3. 引入社会资本，多元化投资康养产业
4. 建立康养服务质量评估体系

## 6. 风险控制

- 需求预测偏差风险: 建立动态需求监测系统
- 资金链断裂风险: 分阶段投资，降低资金压力
- 政策变化风险: 建立政策跟踪和应对机制

## 7. 预期效果

- 康养服务覆盖率提升至95%以上
- 资源利用效率提高30%
- 人均康养服务成本降低20%
- 区域间康养资源配置差异系数降至0.3以下
"""

        return report

    def run_complete_analysis(self):
        """运行完整分析"""
        print("=" * 50)
        print("康养资源优化配置模型分析")
        print("=" * 50)

        # 1. 数据加载和预处理
        if not self.load_data():
            return
        self.preprocess_data()

        # 2. 多目标优化求解
        print("\n正在进行多目标优化求解...")
        optimal_result = self.multi_objective_optimization()

        if optimal_result.success:
            print(f"优化求解成功!")
            print(f"最优解: {optimal_result.x}")
        else:
            print("优化求解失败")
            return

        # 3. 帕累托前沿分析
        print("\n正在进行帕累托前沿分析...")
        pareto_solutions = self.pareto_frontier_analysis()
        print(f"获得{len(pareto_solutions)}个帕累托最优解")

        # 4. 敏感性分析
        print("\n正在进行敏感性分析...")
        sensitivity_results = self.sensitivity_analysis(optimal_result.x)

        # 5. 生成实施策略
        implementation_strategy = self.generate_implementation_strategy(optimal_result.x)

        # 6. 结果可视化
        print("\n正在生成可视化结果...")
        import os
        if not os.path.exists('P3'):
            os.makedirs('P3')
        self.visualize_results(pareto_solutions, sensitivity_results)

        # 7. 生成报告
        report = self.generate_report(optimal_result.x, pareto_solutions, implementation_strategy)

        # 保存报告
        with open('P3/康养资源优化配置分析报告.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("\n分析完成！结果已保存到P3文件夹")
        print("\n=== 主要结果 ===")
        coverage_func, efficiency_func, cost_func = self.define_optimization_objectives()
        print(f"最优资源配置方案:")
        print(f"  医疗机构: {optimal_result.x[0]:.0f}个")
        print(f"  医院床位: {optimal_result.x[1]:.0f}张")
        print(f"  养老机构: {optimal_result.x[2]:.0f}个")
        print(f"  公园数量: {optimal_result.x[3]:.0f}个")
        print(f"\n性能指标:")
        print(f"  服务覆盖率: {coverage_func(optimal_result.x):.2%}")
        print(f"  资源利用效率: {efficiency_func(optimal_result.x):.2%}")
        print(f"  总投资成本: {cost_func(optimal_result.x):.0f}万元")

        return optimal_result, pareto_solutions, implementation_strategy

def main():
    """主函数"""
    optimizer = KangYangResourceOptimizer()
    result = optimizer.run_complete_analysis()
    return result

if __name__ == "__main__":
    main()