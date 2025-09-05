import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV

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
df['GW'] = df['检测孕周'].apply(parse_gw)
df = comp_feat(df)
df_male = df[df['Y染色体浓度'].notna()].copy()
df_male['log_unique'] = np.log1p(df_male['唯一比对的读段数'])

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
print(model_final.summary().tables[1])          # 仅看系数表

# ③ 稳健标准误
model_final_hc3 = model_final.get_robustcov_results(cov_type='HC3')
print(model_final_hc3.summary().tables[1])

# ---------- 6. 残差诊断 ----------
residuals = model_final.resid
fitted = model_final.fittedvalues
plt.figure(figsize=(12,4))
plt.subplot(1,2,1); plt.scatter(fitted, residuals); plt.axhline(0, c='r', ls='--'); plt.title('残差 vs 拟合')
plt.subplot(1,2,2); sm.qqplot(residuals, line='45', fit=True); plt.title('Q-Q 图')
plt.tight_layout(); plt.savefig('./P1/opt_resid.png', dpi=300)

lm, lm_pvalue, _, _ = het_breuschpagan(residuals, model_final.model.exog)
print('Breusch-Pagan p =', lm_pvalue)

# ---------- 7. 特征描述 ----------
print(df[['Seq_QC_score','GC_dev_score','Aneuploidy_score','Sex_signal_strength','BMI_z']].describe())