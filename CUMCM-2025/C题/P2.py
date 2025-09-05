import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_csv('附件_男胎数据.csv')

# 2. 孕周转数值
def parse_gestational_week(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if 'w' not in s:
        return np.nan
    weeks, days = 0, 0
    try:
        if '+' in s:
            part1, part2 = s.split('+')
            weeks = float(part1.replace('w', ''))
            days = float(part2.replace('d', ''))
        else:
            weeks = float(s.replace('w', ''))
    except:
        return np.nan
    return weeks + days / 7.0

df['孕周数值'] = df['检测孕周'].apply(parse_gestational_week)
df['达标'] = df['Y染色体浓度'] >= 0.04
df['孕妇代码'] = df['孕妇代码'].astype(str)

# 3. 每孕妇保留“临界记录”
def extract_critical_record(sub_df):
    sub_df = sub_df.sort_values('孕周数值')
    达标记录 = sub_df[sub_df['达标']]
    if not 达标记录.empty:
        # 最早达标
        return 达标记录.iloc[0]
    else:
        # 最晚未达标
        return sub_df.iloc[-1]

critical_df = df.groupby('孕妇代码').apply(extract_critical_record).reset_index(drop=True)

# 4. BMI四分位数分组
critical_df['BMI_Group'] = pd.qcut(critical_df['孕妇BMI'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 5. 每组搜索最佳孕周阈值
results = []
plt.figure(figsize=(10, 6))

for group in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub = critical_df[critical_df['BMI_Group'] == group].copy()
    if sub.empty:
        continue

    candidates = np.arange(10.0, 25.5, 0.5)
    best_t, best_rate, best_n = None, -1, 0

    # 样本量门槛：优先选达标率>=0.95且样本量>=10，取最早满足的
    for t in candidates:
        selected = sub[sub['孕周数值'] >= t]
        n = len(selected)
        if n < 10:
            continue
        rate = selected['达标'].mean()
        if rate >= 0.95 and (best_t is None or t < best_t):
            best_t = t
            best_rate = rate
            best_n = n

    # 若找不到满足条件的，放宽门槛：达标率>=0.80且样本量>=5
    if best_t is None:
        for t in candidates:
            selected = sub[sub['孕周数值'] >= t]
            n = len(selected)
            if n < 5:
                continue
            rate = selected['达标'].mean()
            if rate >= 0.80 and (best_t is None or t < best_t):
                best_t = t
                best_rate = rate
                best_n = n

    # 若仍无，选最晚一条（保底）
    if best_t is None:
        last = sub.iloc[-1]
        best_t = last['孕周数值']
        best_rate = last['达标']
        best_n = 1

    results.append({
        'BMI组': group,
        '最佳孕周': best_t,
        '最大达标概率': best_rate,
        '样本量': best_n
    })

    # 画图
    rates = []
    for t in candidates:
        selected = sub[sub['孕周数值'] >= t]
        rates.append(selected['达标'].mean() if len(selected) > 0 else np.nan)
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