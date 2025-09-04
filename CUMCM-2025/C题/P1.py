import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# 图表中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据（假设为CSV）
df = pd.read_csv('附件_男胎数据.csv')

# 2. 数据预处理
# 只保留男胎（Y染色体浓度非空）
df_male = df[df['Y染色体浓度'].notna()].copy()

# 将“检测孕周”转换为数值型
def parse_gw(gw_str):
    """
    解析孕周字符串，支持
      11w+6   -> 11.857
      12W     -> 12.0
      13      -> 13.0
      20w+ 1  -> 20.143   （允许空格）
    """
    gw_str = str(gw_str).strip().lower().replace(' ', '')  # 去空格、转小写
    # 正则：周数必须，天数可选
    m = re.match(r'^(\d+)(?:w(?:(?:\+)(\d+))?)?$', gw_str)
    if not m:
        raise ValueError(f"无法解析孕周: {gw_str}")
    weeks = int(m.group(1))
    days = int(m.group(2)) if m.group(2) else 0
    return weeks + days / 7

df_male['GW'] = df_male['检测孕周'].apply(parse_gw)

# 提取所需列
data = df_male[['GW', '孕妇BMI', 'Y染色体浓度']].dropna()
X = data[['GW', '孕妇BMI']]
y = data['Y染色体浓度']

# 3. 探索性分析
g = sns.pairplot(data)
# 设置坐标轴标签为中文
g.fig.suptitle('孕周、孕妇BMI、Y染色体浓度散点图', y=1.02)
for i, var in enumerate(['GW', '孕妇BMI', 'Y染色体浓度']):
    g.fig.axes[i].set_xlabel(var)
    g.fig.axes[i].set_ylabel(var)
plt.savefig('./P1/pairplot.png')

# 4. 建立线性回归模型
X_const = sm.add_constant(X)  # 添加截距项
model = sm.OLS(y, X_const).fit()

# 输出回归结果
print(model.summary())

# 5. 残差分析
residuals = model.resid
fitted = model.fittedvalues

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(fitted, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('拟合值')
plt.ylabel('残差')
plt.title('残差与拟合值')

plt.subplot(1, 2, 2)
sm.qqplot(residuals, line='45', fit=True)
plt.title('残差Q-Q图')
plt.tight_layout()
plt.savefig('./P1/residuals.png')

# 6. 异方差检验（Breusch-Pagan）
_, p_val, _, _ = het_breuschpagan(residuals, X_const)
print(f"Breusch-Pagan test p-value: {p_val:.4f}")
if p_val < 0.05:
    print("存在异方差，建议使用稳健标准误")
else:
    print("无异方差")

# 7. 若存在异方差，使用稳健标准误重新估计
robust_model = model.get_robustcov_results(cov_type='HC3')
print(robust_model.summary())