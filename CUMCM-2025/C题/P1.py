import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV
from scipy import stats
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 1-4 清洗+特征构造 ----------
def parse_gw(gw_str):
    gw_str = str(gw_str).strip().lower().replace(' ','')
    m = re.match(r'^(\d+)(?:w(?:(?:\+)(\d+))?)?$', gw_str)
    if not m: raise ValueError(gw_str)
    weeks = int(m.group(1)); days = int(m.group(2)) if m.group(2) else 0
    return weeks + days/7

def comp_feat(df_in):
    df = df_in.copy()
    ref_AC, ref_UPM = 0.95, df['唯一比对的读段数'].median()*1e6/df['原始读段数'].median()
    df['UPM'] = df['唯一比对的读段数']*1e6/df['原始读段数']
    map_n = (df['在参考基因组上比对的比例']/ref_AC).clip(0,1)
    upm_n = (df['UPM']/ref_UPM).clip(0,1)
    dup_n = (1-df['重复读段的比例']).clip(0,1)
    filt_n = (1-df['被过滤掉读段数的比例']).clip(0,1)
    w1,w2,w3,w4 = 0.30,0.35,0.20,0.15
    df['Seq_QC_score'] = w1*map_n + w2*upm_n + w3*dup_n + w4*filt_n
    gc_dev = df[['13号染色体的GC含量','18号染色体的GC含量','21号染色体的GC含量']].sub(df['GC含量'],axis=0).abs().max(axis=1)
    df['GC_dev_score'] = (1 - gc_dev/0.05).clip(0,1)
    z_norm = np.sqrt((df[['13号染色体的Z值','18号染色体的Z值','21号染色体的Z值']]**2).sum(axis=1))
    df['Aneuploidy_score'] = (z_norm/(z_norm+3.0)).clip(0,1)
    ref_V = df['Y染色体浓度'].median(); ref_U = df['Y染色体的Z值'].abs().median(); ref_W = df['X染色体浓度'].median()
    df['V_norm'] = (df['Y染色体浓度'].fillna(0)/ref_V).clip(0,1)
    df['U_norm'] = (df['Y染色体的Z值'].abs()/ref_U).clip(0,1)
    df['W_norm'] = (df['X染色体浓度']/ref_W).clip(0,1)
    male = 0.7*df['V_norm'] + 0.3*df['U_norm']
    df['Sex_signal_strength'] = np.where(df['Y染色体浓度'].fillna(0)>1e-6, male, 1-df['W_norm']).clip(0,1)
    df['BMI_z'] = (df['孕妇BMI'] - df['孕妇BMI'].mean()) / df['孕妇BMI'].std()
    for col in ['Seq_QC_score', 'GC_dev_score', 'Aneuploidy_score', 'Sex_signal_strength', 'BMI_z']:
        df_in[col] = df[col]
    return df_in

# 读取 & 清洗
df = pd.read_csv('附件_男胎数据.csv')
num_cols = ['年龄', '身高', '体重', '孕妇BMI', '怀孕次数', '生产次数',
            'GC含量', '在参考基因组上比对的比例', '重复读段的比例',
            '唯一比对的读段数', '13号染色体的Z值', '18号染色体的Z值',
            '21号染色体的Z值', 'X染色体的Z值', 'Y染色体浓度']
for col in num_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace('-',''), errors='coerce')
df.columns = df.columns.str.strip()
# 将Y染色体浓度转换为百分比数值（乘以100）
df['Y染色体浓度'] = df['Y染色体浓度'] * 100
df['GW'] = df['检测孕周'].apply(parse_gw)
df = comp_feat(df)
df_male = df[df['Y染色体浓度'].notna()].copy()
df_male['log_unique'] = np.log1p(df_male['唯一比对的读段数'])

# ---------- 数据探索性分析图表 ----------
print("数据探索性分析...")
plt.figure(figsize=(15, 10))

# 1. Y染色体浓度分布
plt.subplot(2, 3, 1)
plt.hist(df_male['Y染色体浓度'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Y染色体浓度 (%)')
plt.ylabel('频数')
plt.title('Y染色体浓度分布')
plt.grid(True, alpha=0.3)

# 2. 孕周分布
plt.subplot(2, 3, 2)
plt.hist(df_male['GW'].dropna(), bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('孕周 (周)')
plt.ylabel('频数')
plt.title('孕周分布')
plt.grid(True, alpha=0.3)

# 3. BMI分布
plt.subplot(2, 3, 3)
plt.hist(df_male['孕妇BMI'].dropna(), bins=20, alpha=0.7, color='salmon', edgecolor='black')
plt.xlabel('BMI')
plt.ylabel('频数')
plt.title('孕妇BMI分布')
plt.grid(True, alpha=0.3)

# 4. Y染色体浓度 vs 孕周
plt.subplot(2, 3, 4)
plt.scatter(df_male['GW'], df_male['Y染色体浓度'], alpha=0.6, c=df_male['孕妇BMI'], cmap='viridis')
plt.colorbar(label='BMI')
plt.xlabel('孕周 (周)')
plt.ylabel('Y染色体浓度 (%)')
plt.title('Y染色体浓度 vs 孕周 (按BMI着色)')
plt.grid(True, alpha=0.3)

# 5. Y染色体浓度 vs BMI
plt.subplot(2, 3, 5)
plt.scatter(df_male['孕妇BMI'], df_male['Y染色体浓度'], alpha=0.6, c=df_male['GW'], cmap='plasma')
plt.colorbar(label='孕周')
plt.xlabel('BMI')
plt.ylabel('Y染色体浓度 (%)')
plt.title('Y染色体浓度 vs BMI (按孕周着色)')
plt.grid(True, alpha=0.3)

# 6. 年龄分布
plt.subplot(2, 3, 6)
plt.hist(df_male['年龄'].dropna(), bins=15, alpha=0.7, color='gold', edgecolor='black')
plt.xlabel('年龄')
plt.ylabel('频数')
plt.title('孕妇年龄分布')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./P1/data_exploration.png', dpi=300, bbox_inches='tight')
plt.close()

print("数据探索性分析图表已保存")

# ---------- 变量相关性热力图 ----------
print("生成变量相关性热力图...")
# 先创建data变量用于相关性分析
corr_data_temp = df_male[['GW','孕妇BMI','年龄','生产次数','唯一比对的读段数',
                         'GC_dev_score','Aneuploidy_score','BMI_z','Y染色体浓度']].dropna()
corr_data_temp['log_unique'] = np.log1p(corr_data_temp['唯一比对的读段数'])

correlation_cols = ['GW', '孕妇BMI', '年龄', '生产次数', 'log_unique',
                   'GC_dev_score', 'Aneuploidy_score', 'BMI_z', 'Y染色体浓度']
corr_data = corr_data_temp[correlation_cols].copy()

plt.figure(figsize=(12, 10))
corr_matrix = corr_data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
           square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('变量相关性热力图', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('./P1/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制Y染色体浓度与其他变量的散点关系矩阵
print("生成散点关系矩阵图...")
scatter_cols = ['GW', '孕妇BMI', '年龄', 'Y染色体浓度']
scatter_data = corr_data_temp[scatter_cols].dropna()

g = sns.PairGrid(scatter_data, diag_sharey=False)
g.map_upper(sns.scatterplot, alpha=0.6)
g.map_lower(sns.scatterplot, alpha=0.6)
g.map_diag(sns.histplot, kde=True)
plt.suptitle('主要变量散点关系矩阵', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig('./P1/scatter_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("相关性分析图表已保存")

# ---------- 5. 正交多项式 + 逐步回归 ----------
data = df_male[['GW','孕妇BMI','年龄','生产次数','log_unique',
                'GC_dev_score','Aneuploidy_score','BMI_z','IVF妊娠','Y染色体浓度']].dropna()


# ① 用sklearn生成多项式特征（正交多项式）
def add_poly_features(df, col, degree, prefix):
    pf = PolynomialFeatures(degree, include_bias=False)
    arr = pf.fit_transform(df[[col]])
    names = [f'{prefix}_poly_{i+1}' for i in range(arr.shape[1])]
    for i, name in enumerate(names):
        df[name] = arr[:, i]
    return df

data_poly = data.copy()
data_poly = add_poly_features(data_poly, 'GW', 3, 'GW')
data_poly = add_poly_features(data_poly, '孕妇BMI', 3, 'BMI')
data_poly = add_poly_features(data_poly, '年龄', 2, 'AGE')

formula_orth = (
    'Q("Y染色体浓度") ~ '
    'GW_poly_1 + GW_poly_2 + GW_poly_3 + '
    'BMI_poly_1 + BMI_poly_2 + BMI_poly_3 + '
    'AGE_poly_1 + AGE_poly_2 + '
    'GC_dev_score + Aneuploidy_score + I(Aneuploidy_score**2) + '
    'BMI_z + 生产次数 + log_unique + C(IVF妊娠)'
)
model_orth = smf.ols(formula_orth, data=data_poly).fit()

# ② 逐步回归（基于 AIC）
def step_aic(model, threshold=1e-4):
    import itertools
    pvals = model.pvalues.drop('Intercept', errors='ignore')
    drop_list = pvals[pvals > threshold].index
    if drop_list.empty:
        return model
    terms = [x for x in model.model.exog_names if x != 'Intercept' and x not in drop_list]
    formula_new = 'Q("Y染色体浓度") ~ ' + ' + '.join(terms)
    return step_aic(smf.ols(formula_new, data=data_poly).fit())

model_final = step_aic(model_orth)
print(model_final.summary())

# ③ 稳健标准误
model_final_hc3 = model_final.get_robustcov_results(cov_type='HC3')
print(model_final_hc3.summary().tables[1])

# ---------- 回归拟合效果可视化 ----------
print("生成回归拟合效果可视化图表...")
plt.figure(figsize=(15, 12))

# 1. 实际值 vs 预测值
plt.subplot(2, 2, 1)
plt.scatter(model_final.fittedvalues, data_poly['Y染色体浓度'], alpha=0.6)
plt.plot([data_poly['Y染色体浓度'].min(), data_poly['Y染色体浓度'].max()],
         [data_poly['Y染色体浓度'].min(), data_poly['Y染色体浓度'].max()],
         'r--', lw=2)
plt.xlabel('预测值')
plt.ylabel('实际值')
plt.title('实际值 vs 预测值')
plt.grid(True, alpha=0.3)

# 2. 残差分布
plt.subplot(2, 2, 2)
residuals = model_final.resid
plt.hist(residuals, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
plt.xlabel('残差')
plt.ylabel('频数')
plt.title('残差分布')
plt.grid(True, alpha=0.3)

# 3. 变量重要性（系数绝对值）
plt.subplot(2, 2, 3)
coef_df = pd.DataFrame({
    'variable': model_final.params.index,
    'coefficient': model_final.params.values,
    'abs_coef': np.abs(model_final.params.values)
})
coef_df = coef_df[coef_df['variable'] != 'Intercept'].sort_values('abs_coef', ascending=True)
plt.barh(coef_df['variable'], coef_df['abs_coef'])
plt.xlabel('系数绝对值')
plt.title('变量重要性（系数绝对值）')
plt.grid(True, alpha=0.3, axis='x')

# 4. 预测误差分布
plt.subplot(2, 2, 4)
prediction_error = data_poly['Y染色体浓度'] - model_final.fittedvalues
plt.hist(prediction_error, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('预测误差')
plt.ylabel('频数')
plt.title('预测误差分布')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./P1/regression_fit_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------- 预测效果验证图表 ----------
print("生成预测效果验证图表...")

# 交叉验证预测效果
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LinearRegression

# 使用线性回归进行交叉验证
X = data_poly.drop('Y染色体浓度', axis=1)
y = data_poly['Y染色体浓度']

# 选择重要的特征（基于最终模型的系数）
important_features = model_final.params.index[model_final.pvalues < 0.05]
important_features = [f for f in important_features if f != 'Intercept' and f in X.columns]

if important_features:
    X_important = X[important_features]

    # 交叉验证预测
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_predictions = cross_val_predict(LinearRegression(), X_important, y, cv=kf)

    plt.figure(figsize=(12, 5))

    # 交叉验证预测 vs 实际值
    plt.subplot(1, 2, 1)
    plt.scatter(cv_predictions, y, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('交叉验证预测值')
    plt.ylabel('实际值')
    plt.title('交叉验证预测效果')
    plt.grid(True, alpha=0.3)

    # 预测误差分布
    plt.subplot(1, 2, 2)
    cv_errors = y - cv_predictions
    plt.hist(cv_errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('交叉验证预测误差')
    plt.ylabel('频数')
    plt.title('交叉验证预测误差分布')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./P1/cross_validation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# ---------- 6. 残差诊断 ----------
residuals = model_final.resid
fitted = model_final.fittedvalues
plt.figure(figsize=(12,4))
plt.subplot(1,2,1); plt.scatter(fitted, residuals); plt.axhline(0, c='r', ls='--'); plt.title('残差 vs 拟合')
plt.subplot(1,2,2); sm.qqplot(residuals, line='45', fit=True); plt.title('Q-Q 图')
plt.tight_layout(); plt.savefig('./P1/opt_resid.png', dpi=300)

lm, lm_pvalue, _, _ = het_breuschpagan(residuals, model_final.model.exog)
print('Breusch-Pagan p =', lm_pvalue)

# ---------- 模型性能指标 ----------
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

y_pred = model_final.fittedvalues
y_true = data_poly['Y染色体浓度']

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"\n模型性能指标:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# ---------- 优化图表样式 ----------
print("优化图表样式...")
# 设置全局图表样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# 创建汇总报告图表
plt.figure(figsize=(15, 10))

# 1. 模型性能指标可视化
plt.subplot(2, 2, 1)
metrics = ['MSE', 'RMSE', 'MAE', 'R²']
values = [mse, rmse, mae, r2]
colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']
bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('数值')
plt.title('模型性能指标')
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# 2. 残差正态性检验
plt.subplot(2, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('残差正态概率图')
plt.grid(True, alpha=0.3)

# 3. 预测误差箱线图
plt.subplot(2, 2, 3)
prediction_errors = y_true - y_pred
plt.boxplot(prediction_errors, vert=False)
plt.xlabel('预测误差')
plt.title('预测误差箱线图')
plt.grid(True, alpha=0.3)

# 4. 变量系数可视化（带置信区间）
plt.subplot(2, 2, 4)
coef_data = pd.DataFrame({
    'variable': model_final.params.index,
    'coef': model_final.params.values,
    'std_err': model_final.bse.values
})
coef_data = coef_data[coef_data['variable'] != 'Intercept'].sort_values('coef', key=abs, ascending=False)
coef_data = coef_data.head(10)  # 显示前10个最重要的变量

y_pos = range(len(coef_data))
plt.barh(y_pos, coef_data['coef'], xerr=coef_data['std_err'],
         alpha=0.7, color='steelblue', ecolor='black', capsize=5)
plt.yticks(y_pos, coef_data['variable'].tolist())
plt.xlabel('系数值')
plt.title('重要变量系数（带标准误）')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('./P1/model_summary_report.png', dpi=300, bbox_inches='tight')
plt.close()

print("所有图表已优化并保存完成！")

# ---------- 7. 特征描述 ----------
print(df[['Seq_QC_score','GC_dev_score','Aneuploidy_score','Sex_signal_strength','BMI_z']].describe())

# ---------- 8. BMI交互项和分组建模 ----------
print("\n=== BMI交互项和分组建模分析 ===")

# (a) 添加GW × BMI交互项
print("\n(a) GW × BMI交互项模型:")
formula_inter = 'Q("Y染色体浓度") ~ GW + I(GW**2) + 孕妇BMI + GW:孕妇BMI'
model_inter = smf.ols(formula_inter, data=data).fit(cov_type="HC3")  # 用稳健误差
print(model_inter.summary())

# (b) BMI分组建模
print("\n(b) BMI分组建模分析:")
# 使用四分位数分组（考虑到孕妇孕期体重变化）
quantiles = data['孕妇BMI'].quantile([0.25, 0.5, 0.75]).tolist()
bins = [data['孕妇BMI'].min()] + quantiles + [data['孕妇BMI'].max()]
labels = ['低BMI组(Q1)', '中低BMI组(Q2)', '中高BMI组(Q3)', '高BMI组(Q4)']
data['BMI_group'] = pd.qcut(data['孕妇BMI'], q=4, labels=labels, duplicates='drop')
print(f"BMI四分位数分组边界: {quantiles}")

# 按组画孕周-浓度曲线
plt.figure(figsize=(12, 8))
sns.lmplot(x='GW', y='Y染色体浓度', hue='BMI_group', data=data, lowess=True, height=8, aspect=1.2)
plt.title("不同BMI组的Y染色体浓度随孕周变化趋势", fontsize=16)
plt.xlabel('孕周 (周)', fontsize=14)
plt.ylabel('Y染色体浓度 (%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./P1/bmi_group_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# 计算各BMI组的统计信息
print("\n各BMI组统计信息:")
bmi_group_stats = data.groupby('BMI_group')['Y染色体浓度'].agg(['count', 'mean', 'std', 'min', 'max'])
print(bmi_group_stats)

# 计算不同阈值下各BMI组的达标情况
print("\n不同阈值下各BMI组的达标情况分析:")

# 尝试多个阈值
thresholds = [0.5, 1.0, 2.0, 3.0, 4.0]  # 从0.5%到4.0%

for threshold in thresholds:
    print(f"\n阈值 {threshold}% 的达标情况:")
    threshold_stats = {}
    for group_name, group_data in data.groupby('BMI_group'):
        above_threshold = group_data[group_data['Y染色体浓度'] >= threshold]
        if len(above_threshold) > 0:
            threshold_stats[group_name] = (
                len(above_threshold),
                len(above_threshold) / len(group_data) * 100,  # 达标比例
                above_threshold['GW'].min() if len(above_threshold) > 0 else None,
                above_threshold['GW'].mean() if len(above_threshold) > 0 else None
            )
        else:
            threshold_stats[group_name] = (0, 0.0, None, None)

    for group, stats in threshold_stats.items():
        if stats[0] > 0:
            print(f"{group}: {stats[0]}人达标({stats[1]:.1f}%), 最早{stats[2]:.1f}周, 平均{stats[3]:.1f}周")
        else:
            print(f"{group}: 无人达标")

# 选择2%作为分析阈值（基于数据分布）
analysis_threshold = 2.0
print(f"\n使用 {analysis_threshold}% 作为分析阈值:")
bmi_threshold_stats = {}
for group_name, group_data in data.groupby('BMI_group'):
    above_threshold = group_data[group_data['Y染色体浓度'] >= analysis_threshold]
    if len(above_threshold) > 0:
        bmi_threshold_stats[group_name] = (
            above_threshold['GW'].min(),
            above_threshold['GW'].mean(),
            above_threshold['GW'].max(),
            len(above_threshold),
            len(above_threshold) / len(group_data) * 100
        )
    else:
        bmi_threshold_stats[group_name] = (None, None, None, 0, 0.0)

print("各BMI组达到2%浓度的孕周统计 (min, mean, max, count, percentage):")
for group, stats in bmi_threshold_stats.items():
    if stats[0] is not None:
        print(f"{group}: {stats[0]:.2f}, {stats[1]:.2f}, {stats[2]:.2f} 周, {stats[3]}人({stats[4]:.1f}%)")
    else:
        print(f"{group}: 未达到{analysis_threshold}%浓度")

# 保存BMI分组结果
bmi_group_stats.to_csv('./P1/bmi_group_statistics.csv')
print("\nBMI分组统计信息已保存到 ./P1/bmi_group_statistics.csv")

print("BMI交互项和分组建模分析完成！")
