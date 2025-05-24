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
        # 默认系数，可以根据实际数据或专家经验进行调整
        self.coverage_coeffs = np.array([0.01, 0.005, 0.02, 0.015]) # 医疗机构, 床位, 养老机构, 公园
        self.efficiency_coeffs = np.array([0.008, 0.004, 0.015, 0.01]) # 医疗机构, 床位, 养老机构, 公园
        self.unit_costs = np.array([100, 5, 80, 20]) # 万元/个或张: 医疗机构, 床位, 养老机构, 公园

    def load_data(self):
        """加载数据集"""
        try:
            # 加载上海数据
            self.data = pd.read_csv(r'.\dataset\上海数据.csv', encoding='utf-8')
            print("上海数据加载成功")

            # 将'指标'列设置为索引并转置数据框
            if '指标' in self.data.columns:
                self.data = self.data.set_index('指标').T

            # 转置后打印列名用于调试
            print("DataFrame columns after loading and transposing:", self.data.columns)
            print("DataFrame index after loading and transposing:", self.data.index)

            # Convert relevant columns to numeric, coercing errors
            numeric_cols = ['医疗卫生机构数(个)', '医院床位数（个）', '公园个数(个)', '建成区绿化覆盖率(%)', '公园绿地面积（万顷）', '文化机构（个）', '老龄人口占比（%）', '养老机构数量（个）', '养老床位（万张）', '常驻人口（万人）', '人均寿命（岁）', '地区生产总值（亿）', '人均生产总值(万元)', '用水总量（亿立方米）', '空气质量指数（%）'] # Corrected back to 老龄人口占比（%）
            for col in numeric_cols:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            # # 加载人口数据 (当前模型未直接使用，若后续分析需要可取消注释)
            # self.population_data = pd.read_csv(r'.\dataset\各地区分年龄、性别的人口(城市).csv', encoding='utf-8')
            # print("人口数据加载成功")

            # # 加载康养城市100强数据 (当前模型未直接使用，若后续分析需要可取消注释)
            # self.top100_cities = pd.read_csv(r'.\dataset\2024 年中国康养城市100强名单.csv', encoding='utf-8')
            # print("康养城市100强数据加载成功")

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
        # 注意：这里的 '人均绿地面积需求' 实际上是总绿地面积需求，单位与公园数量不直接对应
        # 因此，在优化约束中，公园数量(x[3])的约束将保持非负，或设定一个最小固定值
        environment_demand = {
            '总绿地面积需求': latest_data['常驻人口（万人）'] * 0.005 # 示例：人均5平方米绿地，这里计算的是总需求
        }

        # 计算养老资源需求 (示例：基于老化人口比例)
        if '老龄人口占比（%）' in latest_data and '常驻人口（万人）' in latest_data and pd.notna(latest_data['老龄人口占比（%）']) and pd.notna(latest_data['常驻人口（万人）']):
            elderly_population = latest_data['常驻人口（万人）'] * (latest_data['老龄人口占比（%）'] / 100) # Corrected typo here
            aging_demand = {
                '养老机构需求': elderly_population * 0.002, # 示例：每千名老人2个养老机构
                '养老床位需求': elderly_population * 0.05  # 示例：每百名老人5张养老床位
            }
        else:
            # 如果数据缺失，设置默认需求为0或一个基于总人口的较小比例
            print("警告: 老化人口占比或常驻人口数据缺失，养老需求可能不准确。")
            aging_demand = {
                '养老机构需求': latest_data.get('常驻人口（万人）', 0) * 0.0001, # 假设一个非常小的基础需求
                '养老床位需求': latest_data.get('常驻人口（万人）', 0) * 0.0005  # 假设一个非常小的基础需求
            }

        # 返回所有需求
        return medical_demand, environment_demand, aging_demand

    def define_optimization_objectives(self):
        """定义优化目标函数"""
        # 示例：定义覆盖率、效率和成本函数
        # x 是资源配置变量，例如 [医疗机构数量, 床位数, 养老机构数量, 公园数量]

        # 服务覆盖率函数 (示例：简单线性模型)
        def coverage_func(x):
            # 覆盖率与资源数量正相关，系数可配置
            return np.dot(x, self.coverage_coeffs)

        # 资源利用效率函数 (示例：简单线性模型)
        def efficiency_func(x):
            # 效率与资源数量正相关，系数可配置
            return np.dot(x, self.efficiency_coeffs)

        # 总成本函数 (示例：简单线性成本)
        def cost_func(x):
            # 每种资源的单位成本固定，系数可配置
            return np.dot(x, self.unit_costs)

        return coverage_func, efficiency_func, cost_func

    def multi_objective_optimization(self, budget_constraint=50000, weights=None):
        """多目标优化求解"""
        if weights is None:
            weights = [0.4, 0.3, 0.3] # 默认权重

        # 获取需求数据
        self.medical_demand, self.environment_demand, self.aging_demand = self.calculate_resource_demand()

        # 获取目标函数
        coverage_func, efficiency_func, cost_func = self.define_optimization_objectives()

        # 定义综合目标函数（加权法）
        def objective(x, current_weights):
            coverage = coverage_func(x)
            efficiency = efficiency_func(x)
            cost = cost_func(x)

            # 归一化成本（取负值因为要最小化）
            # 避免除以零或预算过小的情况
            normalized_cost = 1 - (cost / budget_constraint) if budget_constraint > 0 else 0

            return -(current_weights[0] * coverage + current_weights[1] * efficiency + current_weights[2] * normalized_cost)

        # 约束条件
        constraints = [
            {'type': 'ineq', 'fun': lambda x: budget_constraint - cost_func(x)},  # 预算约束
            # 资源数量非负约束
            {'type': 'ineq', 'fun': lambda x: x[0]},  # 医疗机构数量 >= 0
            {'type': 'ineq', 'fun': lambda x: x[1]},  # 床位数 >= 0
            {'type': 'ineq', 'fun': lambda x: x[2]},  # 养老机构数量 >= 0
            {'type': 'ineq', 'fun': lambda x: x[3]},  # 公园数量 >= 0
            # 需求满足约束 (x = [医疗机构数量, 床位数, 养老机构数量, 公园数量])
            {'type': 'ineq', 'fun': lambda x: x[0] - self.medical_demand.get('医疗机构需求', 0)},
            {'type': 'ineq', 'fun': lambda x: x[1] - self.medical_demand.get('床位需求', 0)},
            {'type': 'ineq', 'fun': lambda x: x[2] - self.aging_demand.get('养老机构需求', 0)},
            # 养老床位需求由养老机构提供，这里x[1]是医院床位数，养老床位需求暂不直接约束x[1]
            # 如果需要更细致，可以将养老床位作为独立变量或与养老机构关联
            # {'type': 'ineq', 'fun': lambda x: x_养老床位 - self.aging_demand.get('养老床位需求', 0)}
            # 公园数量x[3]的约束：由于environment_demand是面积，这里仅设非负，或可设为x[3] >= 1
        ]

        # 初始猜测
        x0 = np.array([50, 1000, 20, 10])

        # 求解优化问题
        result = minimize(objective, x0, args=(weights,), method='SLSQP', constraints=constraints)

        return result

    def pareto_frontier_analysis(self):
        """帕累托前沿分析"""
        # 获取需求数据，确保帕累托分析时需求是最新的
        # 注意：如果calculate_resource_demand依赖于类实例状态，且该状态可能在不同分析间变化，
        # 则应确保每次都正确计算。当前实现中，它主要依赖load_data的结果，相对稳定。
        # 为确保独立性，可以再次调用，或者依赖于multi_objective_optimization中已设置的self.xxx_demand
        # 如果multi_objective_optimization未被调用，则self.xxx_demand可能未初始化
        # 因此，在此处也获取一次需求是更稳健的做法
        medical_demand, _, aging_demand = self.calculate_resource_demand()

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

        for current_weights in weight_combinations:
            # 调用多目标优化函数，传入当前权重组合
            # 注意：这里复用了 multi_objective_optimization，但其内部 objective 定义会使用传入的 weights
            # 为了确保 pareto_frontier_analysis 中每次迭代使用不同的权重，
            # multi_objective_optimization 需要能接受 weights 参数
            # 或者，像之前一样，在循环内部重新定义 objective 函数
            # 为了代码简洁和复用，我们让 multi_objective_optimization 接受 weights
            result = self.multi_objective_optimization(weights=current_weights)

            if result.success:
                coverage_func, efficiency_func, cost_func = self.define_optimization_objectives()
                solution = {
                    'weights': current_weights,
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
        """生成实施策略，考虑阶段性侧重"""
        optimal_solution_int = np.round(optimal_solution).astype(int)

        # 定义各阶段的侧重点 (示例)
        # 权重矩阵，行代表阶段，列代表资源 [医疗机构, 床位, 养老机构, 公园]
        # 这些权重可以根据实际规划目标调整
        phase_focus_weights = np.array([
            [0.5, 0.4, 0.2, 0.1],  # 阶段1：侧重医疗和床位
            [0.3, 0.3, 0.4, 0.3],  # 阶段2：均衡发展，略侧重养老和公园
            [0.2, 0.3, 0.4, 0.6]   # 阶段3：侧重养老和公园完善
        ])

        # 归一化权重，使得每个阶段的资源分配比例总和为1 (按资源类型)
        # 这里我们希望每个阶段分配的资源是总优化量的一部分，而不是按权重重新分配总资源
        # 因此，这里的权重更像是指导“优先序”或“分配比例”

        # 简单的按比例分配，然后根据侧重调整
        # 初始平均分配到三个阶段 (或根据时间长度分配)
        base_ratio_per_phase = 1/3

        phase1_alloc = np.zeros_like(optimal_solution_int, dtype=float)
        phase2_alloc = np.zeros_like(optimal_solution_int, dtype=float)
        phase3_alloc = np.zeros_like(optimal_solution_int, dtype=float)

        # 根据侧重点调整分配比例
        # 这是一个简化的示例，实际可能需要更复杂的分配逻辑
        # 例如，可以设置每个阶段对各类资源的需求比例，然后按此比例分配

        # 示例：阶段1优先满足医疗和床位的一部分，阶段2和3再补充
        # 这里我们采用一种简化的思路：将总资源按阶段的“重要性”或“紧迫性”分配
        # 以下为一种更直接的按预设比例分配方式，但可以结合“侧重”思想调整这些比例

        # 预设各阶段分配总量的比例（可以根据项目周期、资金到位情况等调整）
        total_phase_ratios = np.array([0.35, 0.35, 0.30]) # 阶段1占35%，阶段2占35%，阶段3占30%

        phase1_target = optimal_solution_int * total_phase_ratios[0]
        phase2_target = optimal_solution_int * total_phase_ratios[1]
        phase3_target = optimal_solution_int * total_phase_ratios[2]

        # 考虑阶段侧重，微调各阶段内部资源分配
        # phase_focus_weights 的行是阶段，列是资源
        # 这里的 phase_focus_weights 更像是在每个阶段内部，如何分配该阶段的总投资到不同资源上
        # 但我们已经有了 optimal_solution_int 作为总量，所以这里调整的是“分配速度”

        # 重新思考分配逻辑：
        # optimal_solution_int 是最终要达到的各类资源数量。
        # 我们需要分阶段达到这个目标。
        # phase_focus_weights 可以理解为每个阶段“优先完成”哪些资源的“比例”。

        # 简化处理：直接使用预设的阶段比例，然后整数化并确保总和正确
        phase1_alloc_raw = optimal_solution_int * phase_focus_weights[0]
        phase2_alloc_raw = optimal_solution_int * phase_focus_weights[1]
        phase3_alloc_raw = optimal_solution_int * phase_focus_weights[2]

        # 上述逻辑不合理，因为phase_focus_weights的和不为1，且目标是分配optimal_solution_int
        # 正确的思路应该是：每个阶段分配optimal_solution_int的一部分
        # 这里的phase_focus_weights可以用来指导“如果某个阶段资源有限，优先投入哪里”
        # 或者，更简单地，我们预设每个阶段完成总目标的一个比例

        ratios_p1 = np.array([0.4, 0.3, 0.2, 0.1]) # 阶段1完成各类资源的比例
        ratios_p2 = np.array([0.3, 0.4, 0.4, 0.4]) # 阶段2完成各类资源的比例
        # 阶段3完成剩余的

        phase1_alloc = np.round(optimal_solution_int * ratios_p1).astype(int)
        remaining_after_p1 = optimal_solution_int - phase1_alloc
        phase2_alloc = np.round(remaining_after_p1 * (ratios_p2 / (1 - ratios_p1 + 1e-9))).astype(int) # 调整比例基数
        phase2_alloc = np.minimum(phase2_alloc, remaining_after_p1) # 确保不超过剩余量
        phase2_alloc = np.maximum(phase2_alloc, 0)

        phase3_alloc = optimal_solution_int - phase1_alloc - phase2_alloc
        phase3_alloc = np.maximum(phase3_alloc, 0)

        # 确保总和正确，如果因为取整有偏差，调整最后一个阶段
        current_total = phase1_alloc + phase2_alloc + phase3_alloc
        diff = optimal_solution_int - current_total
        phase3_alloc += diff # 将差额加到第三阶段
        # 再次确保非负，如果第三阶段调整后为负，说明前面分配过多，需要从前一阶段扣除
        if np.any(phase3_alloc < 0):
            over_alloc_p3 = np.abs(phase3_alloc[phase3_alloc < 0])
            phase3_alloc = np.maximum(phase3_alloc, 0)
            # 从第二阶段扣除 (简化处理，实际可能需要更复杂的调整逻辑)
            phase2_alloc[phase3_alloc < 0] -= over_alloc_p3
            phase2_alloc = np.maximum(phase2_alloc, 0)
            # 重新计算第三阶段的剩余
            phase3_alloc_final_check = optimal_solution_int - phase1_alloc - phase2_alloc
            phase3_alloc = np.maximum(phase3_alloc_final_check, 0)


        strategy = {
            '第一阶段（1-2年）': {
                '医疗机构建设': phase1_alloc[0],
                '床位增设': phase1_alloc[1],
                '养老机构建设': phase1_alloc[2],
                '公园建设': phase1_alloc[3]
            },
            '第二阶段（3-4年）': {
                '医疗机构建设': phase2_alloc[0],
                '床位增设': phase2_alloc[1],
                '养老机构建设': phase2_alloc[2],
                '公园建设': phase2_alloc[3]
            },
            '第三阶段（5年）': {
                '医疗机构建设': phase3_alloc[0],
                '床位增设': phase3_alloc[1],
                '养老机构建设': phase3_alloc[2],
                '公园建设': phase3_alloc[3]
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

    def generate_report(self, optimal_solution, pareto_solutions, implementation_strategy):
        """生成综合分析报告"""
        coverage_func, efficiency_func, cost_func = self.define_optimization_objectives()
        optimal_coverage = coverage_func(optimal_solution)
        optimal_efficiency = efficiency_func(optimal_solution)
        optimal_cost = cost_func(optimal_solution)

        # 默认权重，与 multi_objective_optimization 中定义的一致
        default_weights = [0.4, 0.3, 0.3]

        report = f"""
# 康养资源优化配置分析报告

## 1. 项目背景与目标

本项目旨在构建一个康养资源优化配置模型，以应对日益增长的康养服务需求。模型的核心目标是通过多目标优化方法，在给定的预算约束下，实现服务覆盖率的最大化、资源利用效率的最大化以及建设和运营成本的最小化。优化配置的资源主要包括医疗机构、床位数、养老机构和公园绿地。

（注：当前模型主要使用 `上海数据.csv` 进行分析。`各地区分年龄、性别的人口(城市).csv` 和 `2024 年中国康养城市100强名单.csv` 已在代码中加载但未在当前核心优化逻辑中使用，相关加载代码已被注释。若后续需要更细致的人口结构分析或城市对比，可取消相关代码注释并整合进模型。）

## 2. 数据描述与预处理

- **数据来源**: 主要基于 `上海数据.csv`，包含了上海市多年的康养相关指标数据。
- **关键指标**: 包括医疗卫生机构数、医院床位数、公园个数、老龄人口占比、养老机构数量等。
- **预处理步骤**:
  - 数据转置以匹配分析格式。
  - 数值型数据转换。
  - 缺失值通过前向填充和后向填充处理。
  - 计算了医疗资源密度、床位密度、绿化指数、养老服务密度等衍生指标。

## 3. 优化模型设定

### 3.1 目标函数

模型包含三个主要优化目标：

1. **最大化服务覆盖率**:
    `Coverage = {self.coverage_coeffs[0]} * x_medical + {self.coverage_coeffs[1]} * x_beds + {self.coverage_coeffs[2]} * x_elderly_facilities + {self.coverage_coeffs[3]} * x_parks`
    其中 `x_medical, x_beds, x_elderly_facilities, x_parks` 分别代表医疗机构数量、床位数、养老机构数量和公园数量。
    系数 `[{self.coverage_coeffs[0]}, {self.coverage_coeffs[1]}, {self.coverage_coeffs[2]}, {self.coverage_coeffs[3]}]` 为可配置参数，反映不同资源对覆盖率的贡献度。

2. **最大化资源利用效率**:
    `Efficiency = {self.efficiency_coeffs[0]} * x_medical + {self.efficiency_coeffs[1]} * x_beds + {self.efficiency_coeffs[2]} * x_elderly_facilities + {self.efficiency_coeffs[3]} * x_parks`
    系数 `[{self.efficiency_coeffs[0]}, {self.efficiency_coeffs[1]}, {self.efficiency_coeffs[2]}, {self.efficiency_coeffs[3]}]` 为可配置参数，反映不同资源对效率的贡献度。

3. **最小化总成本**:
    `Cost = {self.unit_costs[0]} * x_medical + {self.unit_costs[1]} * x_beds + {self.unit_costs[2]} * x_elderly_facilities + {self.unit_costs[3]} * x_parks`
    单位成本系数 `[{self.unit_costs[0]}, {self.unit_costs[1]}, {self.unit_costs[2]}, {self.unit_costs[3]}]` (万元/个或张)为可配置参数。

多目标优化采用加权求和法，综合目标函数为：
`Objective = w_coverage * Coverage + w_efficiency * Efficiency - w_cost * Cost` (实际优化时成本项已归一化并取负)
默认权重为：覆盖率 `w_coverage = {default_weights[0]}`, 效率 `w_efficiency = {default_weights[1]}`, 成本 `w_cost = {default_weights[2]}`.

### 3.2 约束条件

- **预算约束**: 总成本不超过预设上限 (默认为50000万元)。
  `{self.unit_costs[0]}*x_medical + {self.unit_costs[1]}*x_beds + {self.unit_costs[2]}*x_elderly_facilities + {self.unit_costs[3]}*x_parks <= 50000`
- **资源数量非负约束**: 各类资源数量必须为非负值。
  `x_medical, x_beds, x_elderly_facilities, x_parks >= 0`
- **需求满足约束**:
  - 医疗机构数量 >= {self.medical_demand.get('医疗机构需求', 0):.2f}
  - 床位数 >= {self.medical_demand.get('床位需求', 0):.2f}
  - 养老机构数量 >= {self.aging_demand.get('养老机构需求', 0):.2f}

## 4. 优化结果 (基于默认权重: 覆盖率={default_weights[0]}, 效率={default_weights[1]}, 成本={default_weights[2]})

- **最优资源配置方案**:
  - 医疗机构数量: {optimal_solution[0]:.2f} 个
  - 床位数: {optimal_solution[1]:.2f} 张
  - 养老机构数量: {optimal_solution[2]:.2f} 个
  - 公园数量: {optimal_solution[3]:.2f} 个
- **预期性能指标**:
  - 服务覆盖率: {optimal_coverage:.4f}
  - 资源利用效率: {optimal_efficiency:.4f}
  - 总成本: {optimal_cost:.2f} 万元

## 5. 帕累托前沿分析

帕累托前沿分析展示了不同权重偏好下的多种优化方案，帮助决策者理解不同目标之间的权衡关系。下表列出了部分帕累托最优解及其对应的性能指标：

| 权重 (覆盖率,效率,成本) | 医疗机构 | 床位数 | 养老机构 | 公园数 | 覆盖率 | 效率 | 成本 (万元) |
|-----------------------|----------|--------|----------|--------|--------|------|-------------|
"""
        for sol in pareto_solutions[:5]: # 最多展示5个
            report += f"| ({sol['weights'][0]},{sol['weights'][1]},{sol['weights'][2]}) | {sol['solution'][0]:.2f} | {sol['solution'][1]:.2f} | {sol['solution'][2]:.2f} | {sol['solution'][3]:.2f} | {sol['coverage']:.4f} | {sol['efficiency']:.4f} | {sol['cost']:.2f} |\n"
        report += """
## 6. 实施策略

基于优化结果，建议分阶段实施资源配置计划。以下是一个三阶段的实施策略示例，具体数量为整数化后的优化结果，并考虑了阶段性侧重：

"""
        for phase, details in implementation_strategy.items():
            report += f"### {phase}\n"
            report += f"- 医疗机构建设: {details['医疗机构建设']} 个\n"
            report += f"- 床位增设: {details['床位增设']} 张\n"
            report += f"- 养老机构建设: {details['养老机构建设']} 个\n"
            report += f"- 公园建设: {details['公园建设']} 个\n\n"
        report += "分配方式说明：以上各阶段分配数量是基于优化结果，通过预设的阶段性侧重比例（例如，阶段1侧重基础医疗，后续阶段逐步完善养老和环境设施）进行整数化分配，并确保各阶段分配总和与优化目标一致。"

        report += """

## 7. 结论与建议

本模型提供了一个量化的康养资源优化配置框架。通过调整目标函数权重、预算约束和资源单位成本等参数，可以适应不同地区和发展阶段的需求。建议结合地方实际情况，进一步细化模型参数，并考虑动态调整策略。

未来的工作可以包括：

- 引入更复杂的非线性目标函数和约束。
- 考虑资源配置的空间布局优化。
- 结合更详细的人口结构数据进行需求预测。
- 将模型与问题二的综合评价结果联动，例如将评价短板作为优化的重点方向。

---
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
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

        # 2. 多目标优化求解 (使用默认权重)
        print("\n正在进行多目标优化求解 (使用默认权重)...")
        # 获取默认权重，如果 multi_objective_optimization 的默认值发生变化，这里也需要同步
        default_weights = self.multi_objective_optimization.__defaults__[1] if self.multi_objective_optimization.__defaults__ and len(self.multi_objective_optimization.__defaults__) > 1 else [0.4, 0.3, 0.3]
        optimal_result = self.multi_objective_optimization(weights=default_weights)

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
        print(f"  服务覆盖率: {coverage_func(optimal_result.x):.2f} (数值)")
        print(f"  资源利用效率: {efficiency_func(optimal_result.x):.2f} (数值)")
        print(f"  总投资成本: {cost_func(optimal_result.x):.0f}万元")

        return optimal_result, pareto_solutions, implementation_strategy

def main():
    """主函数"""
    optimizer = KangYangResourceOptimizer()
    result = optimizer.run_complete_analysis()
    return result

if __name__ == "__main__":
    main()