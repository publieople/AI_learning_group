import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan

# 图表中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据（假设为CSV）
df = pd.read_csv('附件_男胎数据.csv')

# ---------- 数据清洗 ----------
# 需要清洗的数值列
num_cols = ['年龄', '身高', '体重', '孕妇BMI', '怀孕次数', '生产次数',
            'GC含量', '在参考基因组上比对的比例', '重复读段的比例',
            '唯一比对的读段数', '13号染色体的Z值', '18号染色体的Z值',
            '21号染色体的Z值', 'X染色体的Z值', 'Y染色体浓度']

for col in num_cols:
    # 去掉首尾空格、替换常见脏字符
    df[col] = (
        df[col].astype(str)
               .str.strip()
               .replace({'': np.nan, '-': np.nan, 'NA': np.nan, 'null': np.nan})
    )
    # 强制转 float，转不下来的变成 NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.columns = df.columns.str.strip()
print('列名：', df.columns.tolist())

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


# 唯一比对读段数取对数，压缩尺度
df_male['log_unique'] = np.log1p(df_male['唯一比对的读段数'])

candidate_cols = [
    'GW',                # 孕周
    '孕妇BMI',           # BMI
    '年龄',
    '身高',
    '体重',
    '怀孕次数',
    '生产次数',
    'IVF妊娠',           # 类别变量
    'GC含量',
    '在参考基因组上比对的比例',
    '重复读段的比例',
    'log_unique',
    '13号染色体的Z值',
    '18号染色体的Z值',
    '21号染色体的Z值',
    'X染色体的Z值',
    'Y染色体浓度'        # 因变量
]

# ---------- 构造建模用的 DataFrame ----------
cont_vars = ['GW', '孕妇BMI', '年龄', '身高', '体重',
             '怀孕次数', '生产次数', 'GC含量',
             '在参考基因组上比对的比例', '重复读段的比例', 'log_unique']
quad_terms = [f'I({v}**2)' for v in cont_vars]
interact_terms = [f'GW:孕妇BMI', f'年龄:孕妇BMI']
keep_cols = cont_vars + ['IVF妊娠', 'Y染色体浓度']
data = df_male[keep_cols].dropna()

# 3. 探索性分析
g = sns.pairplot(data[['GW', '孕妇BMI', 'Y染色体浓度']])
g.fig.suptitle('孕周、孕妇BMI、Y染色体浓度散点图', y=1.02)
for i, var in enumerate(['GW', '孕妇BMI', 'Y染色体浓度']):
    g.fig.axes[i].set_xlabel(var)
    g.fig.axes[i].set_ylabel(var)
plt.savefig('./P1/pairplot.png')

# 4. 建立线性回归模型


# ---------- 构造公式 ----------
formula = ('Q("Y染色体浓度") ~ ' +
           ' + '.join(cont_vars) +
           ' + ' + ' + '.join(quad_terms) +
           ' + ' + ' + '.join(interact_terms) +
           ' + C(IVF妊娠)')

# ---------- 建模 ----------
model_full = smf.ols(formula, data=data).fit()
print(model_full.summary())

# 5. 残差分析
residuals = model_full.resid
fitted = model_full.fittedvalues

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
_, p_val, _, _ = het_breuschpagan(residuals, model_full.model.exog)
print(f"Breusch-Pagan test p-value: {p_val:.4f}")
if p_val < 0.05:
    print("存在异方差，建议使用稳健标准误")
else:
    print("无异方差")

# 7. 若存在异方差，使用稳健标准误重新估计（公式接口同样支持）
print(model_full.get_robustcov_results(cov_type='HC3').summary())