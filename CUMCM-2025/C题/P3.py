import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('./P3', exist_ok=True)
SEED = 42
np.random.seed(SEED)

# -------------------------------------------------
# ① 读取 & 清洗
# -------------------------------------------------
df = pd.read_csv('附件_男胎数据.csv')

# 孕周解析
def parse_gw(s):
    s = str(s).strip()
    if 'w' not in s: return np.nan
    try:
        if '+' in s:
            w, d = s.split('+')
            return float(w.replace('w', '')) + float(d.replace('d', '')) / 7
        return float(s.replace('w', ''))
    except:
        return np.nan

df['gw'] = df['检测孕周'].apply(parse_gw)
df['qualified'] = df['Y染色体浓度'] >= 0.04
df['mother_id'] = df['孕妇代码'].astype(str)

# 医学质控清洗
print('清洗前样本数:', len(df))
df = df[
    (df['GC含量'] >= 0.4) &
    (df['检测抽血次数'] >= 2) &
    (df['Y染色体浓度'] >= 0.01) &
    (df['gw'] >= 10) &
    (df['gw'] <= 25)
].copy()
print('清洗后样本数:', len(df))

# -------------------------------------------------
# ② 特征工程
# -------------------------------------------------
df['detect_delay'] = (pd.to_datetime(df['检测日期']) - pd.to_datetime(df['末次月经'])).dt.days
df['IVF'] = (df['IVF妊娠'] == '是').astype(int)
# 孕/产次数处理
df['怀孕次数'] = df['怀孕次数'].apply(lambda x: 3 if str(x).startswith('≥') else x)
df['怀孕次数'] = pd.to_numeric(df['怀孕次数'], errors='coerce')
df['生产次数'] = pd.to_numeric(df['生产次数'], errors='coerce')

feat_cols = ['孕妇BMI', '身高', '体重', '怀孕次数', '生产次数', 'detect_delay']
X_base = df[feat_cols].fillna(df[feat_cols].mean())

# -------------------------------------------------
# ③ 临界记录提取（每孕妇最早达标或最后记录）
# -------------------------------------------------
def critical_rec(sub):
    sub = sub.sort_values('gw')
    q = sub[sub['qualified']]
    return q.iloc[0] if not q.empty else sub.iloc[-1]

crit = (df.groupby('mother_id', group_keys=False)
        .apply(critical_rec, include_groups=False)
        .reset_index(drop=True))

# -------------------------------------------------
# ④ 分组策略
# -------------------------------------------------
print('原始样本数:', len(df))
# 4.0 先清 NaN + 重索引，防止空表
crit = crit.dropna(subset=['孕妇BMI']).reset_index(drop=True)

if crit.empty:
    raise ValueError('【错误】dropna 后 crit 为空，请检查清洗条件是否过严！')

# 4.1 BMI 四分位（强制 4 组，容忍重复边缘）
crit['bmi_q'], bins = pd.qcut(
    crit['孕妇BMI'],
    q=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4'],
    duplicates='drop',
    retbins=True
)
print(f'BMI 分位成功：生成 {len(bins)-1} 组，bins={bins}')
q1_raw = crit[crit['bmi_q'] == 'Q1']
print(q1_raw[['孕妇BMI', 'gw', 'Y染色体浓度', 'qualified']].describe())
print(q1_raw[q1_raw['qualified']]['gw'].min())


# -------------------------------------------------
# ⑤ 最佳孕周搜索
# -------------------------------------------------
def find_best_week(sub, candidates=np.arange(10.0, 25.5, 0.5)):
    best_week, best_rate, best_n = None, -1, 0
    for t in candidates:
        late = sub[sub['gw'] >= t]
        if len(late) < 10: continue
        rate = late['qualified'].mean()
        if rate >= 0.95 and (best_week is None or t < best_week):
            best_week, best_rate, best_n = t, rate, len(late)
    if best_week is None:  # 放宽
        for t in candidates:
            late = sub[sub['gw'] >= t]
            if len(late) < 5: continue
            rate = late['qualified'].mean()
            if rate >= 0.80 and (best_week is None or t < best_week):
                best_week, best_rate, best_n = t, rate, len(late)
    if best_week is None:
        rec = sub.iloc[-1]
        best_week, best_rate, best_n = rec['gw'], rec['qualified'], 1
    return best_week, best_rate, best_n

# -------------------------------------------------
# ⑥ 分组 pipeline
# -------------------------------------------------
def pipeline(gp_col, tag):
    res = []
    plt.figure(figsize=(10, 6))
    for g in sorted(crit[gp_col].unique()):
        sub = crit[crit[gp_col] == g]
        week, rate, n = find_best_week(sub)
        res.append({'group': g, '最佳孕周': week,
                   '最大达标概率': rate, '样本量': n})

        # 曲线
        rates = [sub[sub['gw'] >= t]['qualified'].mean()
                 for t in np.arange(10.0, 25.5, 0.5)]
        plt.plot(np.arange(10.0, 25.5, 0.5), rates,
                 label=f'{g} (n={len(sub)})')
    plt.xlabel('孕周阈值'); plt.ylabel('达标率'); plt.title(f'{tag} 达标曲线')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f'./P3/问题三_达标曲线_{tag}.png', dpi=300)
    plt.close()

    out = pd.DataFrame(res)
    out.to_csv(f'./P3/问题三_最佳孕周表_{tag}.csv', index=False, encoding='utf-8-sig')
    print(f'【{tag}】'); print(out)
    return out

_ = pipeline('bmi_q', 'BMI四分位')

# -------------------------------------------------
# ⑦ 蒙特卡洛误差分析
# -------------------------------------------------
def mc_error(gp_col, tag, n_sim=1000):
    recs = []
    for g in sorted(crit[gp_col].unique()):
        sub = crit[crit[gp_col] == g].copy()
        weeks = []
        for _ in range(n_sim):
            noise = np.random.normal(0, sub['Y染色体浓度'] * 0.05)
            sub_sim = sub.copy()
            sub_sim['qualified'] = (sub_sim['Y染色体浓度'] + noise) >= 0.04
            week, _, _ = find_best_week(sub_sim)
            weeks.append(week)
        recs.append({
            'group': g,
            '最佳孕周均值': np.mean(weeks),
            '最佳孕周标准差': np.std(weeks),
            '95%CI下限': np.percentile(weeks, 2.5),
            '95%CI上限': np.percentile(weeks, 97.5)
        })
    df_err = pd.DataFrame(recs)
    df_err.to_csv(f'./P3/问题三_误差分析_{tag}.csv', index=False, encoding='utf-8-sig')
    print(f'【误差分析 {tag}】')
    print(df_err)

mc_error('bmi_q', 'BMI四分位')

print('=== 问题三全部完成，结果已写入 ./P3/ ===')
