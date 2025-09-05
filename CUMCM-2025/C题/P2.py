import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# 中文图例
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ███████████████  1. 参数开关  ███████████████
USE_SMART = True          # False 则退回原版 BMI-KMeans
N_CLUSTERS = 4            # 簇数（智能聚类也会自动在2~8里挑最优）
SEED = 42
# ████████████████████████████████████████████

os.makedirs('./P2', exist_ok=True)

# 1. 读数据
df = pd.read_csv('附件_男胎数据.csv')

# 2. 孕周解析
def gestational_week_to_float(week_str):
    if pd.isna(week_str):
        return np.nan
    week_str = str(week_str).strip()
    if 'w' not in week_str:
        return np.nan
    try:
        if '+' in week_str:
            weeks, days = week_str.split('+')
            weeks = float(weeks.replace('w', ''))
            days = float(days.replace('d', ''))
        else:
            weeks = float(week_str.replace('w', ''))
            days = 0
        return weeks + days / 7.0
    except:
        return np.nan

df['gestational_week_float'] = df['检测孕周'].apply(gestational_week_to_float)
df['qualified'] = df['Y染色体浓度'] >= 0.04
df['mother_id'] = df['孕妇代码'].astype(str)

# 3. 临界记录
def get_critical_record(sub_df):
    sub_df = sub_df.sort_values('gestational_week_float')
    qualified_records = sub_df[sub_df['qualified']]
    if not qualified_records.empty:
        return qualified_records.iloc[0]
    else:
        return sub_df.iloc[-1]

critical_df = (df.groupby('mother_id', group_keys=False)
                 .apply(get_critical_record, include_groups=False)
                 .reset_index(drop=True))

# 4. 特征工程 + 智能聚类
def make_smart_cluster(tmp):
    """返回：新的 critical_df（带 cluster 列） + 簇数"""
    # 4.1 构造特征
    tmp = tmp.copy()
    tmp['detect_delay'] = (pd.to_datetime(tmp['检测日期']) -
                           pd.to_datetime(tmp['末次月经'])).dt.days
    tmp['IVF'] = (tmp['IVF妊娠'] == '是').astype(int)

    # 处理怀孕次数中的字符串值（如'≥3'）
    tmp['怀孕次数'] = tmp['怀孕次数'].apply(lambda x: 3 if str(x).startswith('≥') else x)
    tmp['怀孕次数'] = pd.to_numeric(tmp['怀孕次数'], errors='coerce')
    tmp['生产次数'] = pd.to_numeric(tmp['生产次数'], errors='coerce')

    num_cols = ['孕妇BMI', '年龄', '身高', '体重',
                '怀孕次数', '生产次数', 'detect_delay']
    X = tmp[num_cols].copy()

    # 填充缺失值
    X = X.fillna(X.mean())

    # 4.2 离群清洗
    iso = IsolationForest(contamination=0.03, random_state=SEED)
    mask = iso.fit_predict(X) == 1
    X_clean = X[mask]
    tmp_clean = tmp.loc[mask].copy()

    # 4.3 标准化 + PCA
    X_scaled = StandardScaler().fit_transform(X_clean)
    pca = PCA(n_components=0.95, svd_solver='full')
    X_pca = pca.fit_transform(X_scaled)

    # 4.4 GMM 选最优簇数
    best_n, best_sil = N_CLUSTERS, -1
    for k in range(2, 9):
        gmm = GaussianMixture(n_components=k, random_state=SEED)
        labels = gmm.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        if sil > best_sil:
            best_n, best_sil = k, sil

    gmm = GaussianMixture(n_components=best_n, random_state=SEED)
    tmp_clean['cluster'] = gmm.fit_predict(X_pca)

    # 4.5 离群点设为 -1
    tmp['cluster'] = -1
    tmp.loc[mask, 'cluster'] = tmp_clean['cluster']
    return tmp, best_n

if USE_SMART:
    critical_df, optimal_k = make_smart_cluster(critical_df)
    critical_df['bmi_cluster_label'] = critical_df['cluster'].map(lambda x: f'S{x}')
    print(f'【智能聚类】最优簇数 = {optimal_k}')
else:
    # 原版 BMI-KMeans
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED)
    bmi_vals = critical_df['孕妇BMI'].fillna(critical_df['孕妇BMI'].mean()).values.reshape(-1, 1)
    critical_df['bmi_cluster'] = kmeans.fit_predict(bmi_vals)
    critical_df['bmi_cluster_label'] = critical_df['bmi_cluster'].map(lambda x: f'C{x+1}')

# 5. 最佳孕周搜索函数（通用）
def find_best_week(sub, candidates=np.arange(10.0, 25.5, 0.5)):
    best_week, best_rate, best_n = None, -1, 0
    for t in candidates:
        late = sub[sub['gestational_week_float'] >= t]
        n = len(late)
        if n < 10:
            continue
        rate = late['qualified'].mean()
        if isinstance(rate, pd.Series):
            rate = rate.iloc[0]  # 确保rate是标量值
        if rate >= 0.95 and (best_week is None or t < best_week):
            best_week, best_rate, best_n = t, rate, n
    if best_week is None:
        for t in candidates:
            late = sub[sub['gestational_week_float'] >= t]
            n = len(late)
            if n < 5:
                continue
            rate = late['qualified'].mean()
            if isinstance(rate, pd.Series):
                rate = rate.iloc[0]  # 确保rate是标量值
            if rate >= 0.80 and (best_week is None or t < best_week):
                best_week, best_rate, best_n = t, rate, n
    if best_week is None:
        last = sub.iloc[-1]
        best_week, best_rate, best_n = last['gestational_week_float'], last['qualified'], 1
    return best_week, best_rate, best_n

# 6. 对每种分组跑一遍
def pipeline_one_grouping(critical_df, group_col, tag):
    """group_col: 'bmi_group' 或 'bmi_cluster_label'"""
    results, clus_results = [], []
    plt.figure(figsize=(10, 6))

    for g in sorted(critical_df[group_col].unique()):
        if g == -1:
            continue
        sub = critical_df[critical_df[group_col] == g]
        best_week, best_rate, best_n = find_best_week(sub)

        bucket = {'group': g, '最佳孕周': best_week,
                  '最大达标概率': best_rate, '样本量': best_n}
        (clus_results if 'cluster' in group_col else results).append(bucket)

        # 画曲线
        rates = []
        for t in np.arange(10.0, 25.5, 0.5):
            late = sub[sub['gestational_week_float'] >= t]
            rates.append(late['qualified'].mean() if len(late) else np.nan)
        plt.plot(np.arange(10.0, 25.5, 0.5), rates,
                 label=f'{g} (n={len(sub)})',
                 linestyle='--' if 'cluster' in group_col else '-')

    plt.xlabel('孕周阈值（周）')
    plt.ylabel('Y浓度 ≥ 4% 的比例')
    plt.title(f'{tag} 达标概率曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./P2/问题二_达标概率曲线_{tag}.png', dpi=300)
    plt.close()

    df_out = pd.DataFrame(clus_results if 'cluster' in group_col else results)
    df_out.to_csv(f'./P2/问题二_最佳孕周表_{tag}.csv', index=False, encoding='utf-8-sig')
    print(f'【{tag}】最佳孕周表'); print(df_out)
    return df_out

# 6.1 四分位（仅做一次）
critical_df['bmi_group'] = pd.qcut(critical_df['孕妇BMI'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
_ = pipeline_one_grouping(critical_df, 'bmi_group', '四分位')

# 6.2 聚类（智能 or 原版）
_ = pipeline_one_grouping(critical_df, 'bmi_cluster_label', 'smart聚类')

# 7. 比例差异检验（以聚类为例，四分位同理可复制）
def prop_test_and_plot(critical_df, group_col, tag):
    test_df = []
    for g in sorted(critical_df[group_col].unique()):
        if g == -1:
            continue
        sub = critical_df[critical_df[group_col] == g]
        for t in np.arange(10.0, 25.5, 0.5):
            early = sub[sub['gestational_week_float'] < t]
            late = sub[sub['gestational_week_float'] >= t]
            if len(early) == 0 or len(late) == 0:
                continue
            count = [late['qualified'].sum(), early['qualified'].sum()]
            nobs = [len(late), len(early)]
            if min(count) == 0 or min([n - c for c, n in zip(count, nobs)]) == 0:
                continue
            z, p = proportions_ztest(count, nobs)
            p_diff = count[0]/nobs[0] - count[1]/nobs[1]
            test_df.append({'group': g, '阈值': t, '差值': p_diff,
                            '负logP': -np.log10(p) if p > 0 else np.nan})
    test_df = pd.DataFrame(test_df)
    plt.figure(figsize=(10, 6))
    for g in test_df['group'].unique():
        sub = test_df[test_df['group'] == g]
        plt.plot(sub['阈值'], sub['负logP'], label=g)
    plt.axhline(y=-np.log10(0.05), ls='--', c='gray')
    plt.title(f'{tag} 比例差异检验显著性')
    plt.xlabel('孕周阈值'); plt.ylabel('-log10(P)'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./P2/显著性曲线_{tag}.png', dpi=300)
    plt.close()
    return test_df

_ = prop_test_and_plot(critical_df, 'bmi_cluster_label', 'smart聚类')

# 8. 蒙特卡洛误差分析
def error_analysis(critical_df, group_col, tag, n_sim=1000):
    res = []
    for g in sorted(critical_df[group_col].unique()):
        if g == -1:
            continue
        sub = critical_df[critical_df[group_col] == g]
        weeks_sim = []
        for _ in range(n_sim):
            noise = np.random.normal(0, sub['Y染色体浓度'] * 0.05)
            sub_sim = sub.copy()
            sub_sim['qualified_sim'] = (sub_sim['Y染色体浓度'] + noise) >= 0.04
            best_week, _, _ = find_best_week(sub_sim.rename(columns={'qualified_sim': 'qualified'}).copy())
            weeks_sim.append(best_week)
        res.append({'group': g,
                    '最佳孕周均值': np.mean(weeks_sim),
                    '最佳孕周标准差': np.std(weeks_sim),
                    '95%CI下限': np.percentile(weeks_sim, 2.5),
                    '95%CI上限': np.percentile(weeks_sim, 97.5)})
    df_err = pd.DataFrame(res)
    df_err.to_csv(f'./P2/检测误差分析_{tag}.csv', index=False, encoding='utf-8-sig')
    print(f'【检测误差分析 {tag}】'); print(df_err)
    return df_err

_ = error_analysis(critical_df, 'bmi_cluster_label', 'smart聚类')

print('=== 全部完成，结果已写入 ./P2/ ===')
