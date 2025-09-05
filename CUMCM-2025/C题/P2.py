import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_csv('附件_男胎数据.csv')

# 2. 孕周转数值
def gestational_week_to_float(week_str):
    if pd.isna(week_str):
        return np.nan
    week_str = str(week_str).strip()
    if 'w' not in week_str:
        return np.nan
    weeks, days = 0, 0
    try:
        if '+' in week_str:
            part1, part2 = week_str.split('+')
            weeks = float(part1.replace('w', ''))
            days = float(part2.replace('d', ''))
        else:
            weeks = float(week_str.replace('w', ''))
    except:
        return np.nan
    return weeks + days / 7.0

df['gestational_week_float'] = df['检测孕周'].apply(gestational_week_to_float)
df['qualified'] = df['Y染色体浓度'] >= 0.04
df['mother_id'] = df['孕妇代码'].astype(str)

# 3. 每孕妇保留“临界记录”
def get_critical_record(sub_df):
    sub_df = sub_df.sort_values('gestational_week_float')
    qualified_records = sub_df[sub_df['qualified']]
    if not qualified_records.empty:
        # 最早达标
        return qualified_records.iloc[0]
    else:
        # 最晚未达标
        return sub_df.iloc[-1]

critical_df = df.groupby('mother_id', group_keys=False).apply(get_critical_record).reset_index(drop=True)

# 4. BMI四分位数分组
critical_df['bmi_group'] = pd.qcut(critical_df['孕妇BMI'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 5. 每组搜索最佳孕周阈值
results = []
plt.figure(figsize=(10, 6))

for group in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub = critical_df[critical_df['bmi_group'] == group].copy()
    if sub.empty:
        continue

    candidates = np.arange(10.0, 25.5, 0.5)
    best_week, best_rate, best_n = None, -1, 0

    # 样本量门槛：优先选达标率>=0.95且样本量>=10，取最早满足的
    for t in candidates:
        selected = sub[sub['gestational_week_float'] >= t]
        n = len(selected)
        if n < 10:
            continue
        rate = selected['qualified'].mean()
        if rate >= 0.95 and (best_week is None or t < best_week):
            best_week = t
            best_rate = rate
            best_n = n

    # 若找不到满足条件的，放宽门槛：达标率>=0.80且样本量>=5
    if best_week is None:
        for t in candidates:
            selected = sub[sub['gestational_week_float'] >= t]
            n = len(selected)
            if n < 5:
                continue
            rate = selected['qualified'].mean()
            if rate >= 0.80 and (best_week is None or t < best_week):
                best_week = t
                best_rate = rate
                best_n = n

    # 若仍无，选最晚一条（保底）
    if best_week is None:
        last = sub.iloc[-1]
        best_week = last['gestational_week_float']
        best_rate = last['qualified']
        best_n = 1

    results.append({
        'BMI组': group,
        '最佳孕周': best_week,
        '最大达标概率': best_rate,
        '样本量': best_n
    })

    # 画图
    rates = []
    for t in candidates:
        selected = sub[sub['gestational_week_float'] >= t]
        rates.append(selected['qualified'].mean() if len(selected) > 0 else np.nan)
    plt.plot(candidates, rates, label=f'{group} (n={len(sub)})')

# 6. 保存结果
result_df = pd.DataFrame(results)
print("【问题二结果】")
print(result_df)
os.makedirs('./P2', exist_ok=True)
result_df.to_csv('./P2/问题二_最佳孕周表.csv', index=False, encoding='utf-8-sig')

# 7. 保存图像
plt.xlabel('孕周阈值（周）')
plt.ylabel('Y浓度 ≥ 4% 的比例')
plt.title('各BMI组在不同孕周阈值下的达标概率')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./P2/问题二_达标概率曲线.png', dpi=300)

# 8. 比例差异检验 + 稳健性分析
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import seaborn as sns

# 8.1 对每个BMI组、每个阈值做比例差异检验
test_results = []
for group in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub = critical_df[critical_df['bmi_group'] == group].copy()
    if sub.empty:
        continue
    for t in np.arange(10.0, 25.5, 0.5):
        early = sub[sub['gestational_week_float'] < t]
        late = sub[sub['gestational_week_float'] >= t]
        if len(early) == 0 or len(late) == 0:
            continue
        # 构造列联表
        count = [late['qualified'].sum(), early['qualified'].sum()]
        nobs = [len(late), len(early)]
        z, p = proportions_ztest(count, nobs)
        # 差值及95%CI
        p_diff = count[0]/nobs[0] - count[1]/nobs[1]
        ci_low, ci_high = proportion_confint(count[0], nobs[0], alpha=0.05, method='wilson')
        ci_low -= proportion_confint(count[1], nobs[1], alpha=0.05, method='wilson')[0]
        ci_high -= proportion_confint(count[1], nobs[1], alpha=0.05, method='wilson')[0]
        test_results.append({
            'BMI组': group,
            '阈值': t,
            'late达标率': count[0]/nobs[0],
            'early达标率': count[1]/nobs[1],
            '差值': p_diff,
            'CI_low': p_diff + ci_low,
            'CI_high': p_diff + ci_high,
            'Z值': z,
            'P值': p,
            '负logP': -np.log10(p) if p > 0 else np.nan
        })

test_df = pd.DataFrame(test_results)
test_df.to_csv('./P2/比例差异检验结果.csv', index=False, encoding='utf-8-sig')

# 8.2 画显著性曲线
plt.figure(figsize=(10, 6))
for group in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub = test_df[test_df['BMI组'] == group]
    plt.plot(sub['阈值'], sub['负logP'], label=f'{group}')
plt.axhline(y=-np.log10(0.05), color='gray', linestyle='--', label='P=0.05')
plt.xlabel('孕周阈值')
plt.ylabel('-log10(P)')
plt.title('比例差异检验显著性曲线')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./P2/显著性曲线.png', dpi=300)

# 8.3 森林图：差值 + 95%CI
plt.figure(figsize=(8, 6))
sns.set_style('whitegrid')
sns.set_theme(font='SimHei')
for i, group in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    sub = test_df[test_df['BMI组'] == group]
    # 只画阈值 10~14 的，避免图太长
    sub = sub[(sub['阈值'] >= 10) & (sub['阈值'] <= 14)]
    y = i * 10 + sub['阈值'] - 10
    xerr_low = np.maximum(sub['差值'] - sub['CI_low'], 0)
    xerr_high = np.maximum(sub['CI_high'] - sub['差值'], 0)
    plt.errorbar(sub['差值'], y, xerr=[xerr_low, xerr_high],
                 fmt='o', label=group, capsize=3)
plt.axvline(x=0, color='black', linestyle='--')
plt.xlabel('达标率差值（late - early）')
plt.ylabel('阈值与BMI组')
plt.title('森林图：达标率差值及95%CI')
plt.legend()
plt.tight_layout()
plt.savefig('./P2/森林图.png', dpi=300)

# 检测误差分析模块
def analyze_detection_error(critical_df, n_simulations=1000):
    np.random.seed(42)
    error_results = []

    # 假设检测误差服从正态分布（标准差为真实浓度的5%）
    for group in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = critical_df[critical_df['bmi_group'] == group].copy()
        if sub.empty:
            continue

        # 存储每次模拟的最佳孕周
        best_weeks = []

        for _ in range(n_simulations):
            # 添加随机噪声模拟检测误差
            noise = np.random.normal(0, sub['Y染色体浓度'] * 0.05)
            sub['Y_concentration_sim'] = np.clip(
                sub['Y染色体浓度'] + noise, 0, 1
            )
            sub['qualified_sim'] = sub['Y_concentration_sim'] >= 0.04

            # 重新计算最佳孕周（使用相同逻辑）
            candidates = np.arange(10.0, 25.5, 0.5)
            best_week = None

            for t in candidates:
                selected = sub[sub['gestational_week_float'] >= t]
                if len(selected) < 10:
                    continue
                rate = selected['qualified_sim'].mean()
                if rate >= 0.95 and (best_week is None or t < best_week):
                    best_week = t

            if best_week is None:
                for t in candidates:
                    selected = sub[sub['gestational_week_float'] >= t]
                    if len(selected) < 5:
                        continue
                    rate = selected['qualified_sim'].mean()
                    if rate >= 0.80 and (best_week is None or t < best_week):
                        best_week = t

            if best_week is None:
                best_week = sub['gestational_week_float'].max()

            best_weeks.append(best_week)

        # 计算统计量
        error_results.append({
            'BMI组': group,
            '最佳孕周均值': np.mean(best_weeks),
            '最佳孕周标准差': np.std(best_weeks),
            '95%置信区间下限': np.percentile(best_weeks, 2.5),
            '95%置信区间上限': np.percentile(best_weeks, 97.5)
        })

    return pd.DataFrame(error_results)

# 执行误差分析
error_analysis = analyze_detection_error(critical_df)
print("\n【检测误差分析结果】")
print(error_analysis)
error_analysis.to_csv('./P2/问题二_检测误差分析.csv', index=False, encoding='utf-8-sig')