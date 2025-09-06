import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (confusion_matrix, classification_report,
                             recall_score, make_scorer)
from imblearn.ensemble import BalancedRandomForestClassifier   # 自带欠采样
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读数据
df = pd.read_csv('附件_女胎数据.csv')

# 2. 基础清洗
df = df.drop_duplicates(subset=['孕妇代码'])     # 同一孕妇只保留一条记录
df = df.reset_index(drop=True)

# 3. 构造标签：AB列任意非空即视为异常
abnormal_mask = df['染色体的非整倍体'].notna()
y = abnormal_mask.astype(int)                  # 1=异常，0=正常

# 4. 特征列表（可根据需要增删）
feat_cols = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',
             'X染色体的Z值', 'X染色体浓度',
             '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
             'GC含量', '被过滤掉读段数的比例', '孕妇BMI', '唯一比对的读段数']

X = df[feat_cols].copy()

# 5. 建模管道：缺失值→标准化→模型
pipe = Pipeline(steps=[
    ('imp', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('clf', BalancedRandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        class_weight='balanced_subsample',
        sampling_strategy='all',   # 对多数类欠采样
        replacement=False,
        random_state=42,
        n_jobs=-1))
])

# 6. 交叉验证：以异常类召回率为核心
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'recall_abn': make_scorer(recall_score, pos_label=1),
    'f1_abn': make_scorer(lambda t, p: recall_score(t, p, pos_label=1)),  # 可换成fbeta
    'auc': 'roc_auc'
}
scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)

print('===== 交叉验证结果 =====')
print('异常类召回率（Recall）: {:.3f} ± {:.3f}'.format(
    scores['test_recall_abn'].mean(), scores['test_recall_abn'].std()))
print('AUC: {:.3f} ± {:.3f}'.format(
    scores['test_auc'].mean(), scores['test_auc'].std()))

# 7. 全数据重新训练，输出混淆矩阵及详细指标
pipe.fit(X, y)
y_pred = pipe.predict(X)
print('\n===== 全样本混淆矩阵 =====')
print(confusion_matrix(y, y_pred, labels=[0, 1]))
print('\n===== 分类报告 =====')
print(classification_report(y, y_pred, target_names=['正常', '异常']))

# 8. 特征重要性可视化
clf = pipe.named_steps['clf']
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=np.array(feat_cols)[indices], palette='Blues_r')
plt.title('随机森林特征重要性（女胎非整倍体）')
plt.tight_layout()
plt.savefig('./P4/随机森林特征重要性（女胎非整倍体）.png')
