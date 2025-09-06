import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

# 1. 读数据 ---------------------------------------------------------------
df = pd.read_csv('附件_女胎数据.csv')

# 2. 构造标签：AB 列非空 -> 异常(1)，否则正常(0)
y = (~df['染色体的非整倍体'].isna()).astype(int)

# 3. 选择特征 --------------------------------------------------------------
num_cols = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',
            'X染色体的Z值', 'GC含量', '孕妇BMI', '唯一比对的读段数']

# 处理缺失值：用中位数填充数值特征，保留所有样本
X = df[num_cols].copy()
# 填充数值特征的缺失值
for col in num_cols:
    if X[col].isna().any():
        X[col].fillna(X[col].median(), inplace=True)

# 标签保持不变
y = (~df['染色体的非整倍体'].isna()).astype(int)

# 4. 建模管道：标准化 -> SMOTE -> 逻辑回归 -------------------------------
log_reg = LogisticRegression(max_iter=2000, class_weight=None)  # 类别平衡由 SMOTE 做
pipe = ImbPipeline(steps=[
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', log_reg)
])

# 5. 交叉验证 -------------------------------------------------------------
# 使用分层交叉验证确保每个fold都有异常样本
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

precs, recs, f1s = [], [], []
cms = np.zeros((2, 2), dtype=int)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # 检查训练集中是否有异常样本，如果没有则跳过SMOTE
    if y_tr.sum() == 0:
        print(f'Fold{fold}: 训练集中无异常样本，跳过SMOTE')
        # 使用简单的标准化和逻辑回归（不使用SMOTE）
        simple_pipe = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
        ])
        simple_pipe.fit(X_tr, y_tr)
        y_pred = simple_pipe.predict(X_va)
    else:
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_va)

    p, r, f1, _ = precision_recall_fscore_support(y_va, y_pred, pos_label=1, average='binary', zero_division=0)
    precs.append(p); recs.append(r); f1s.append(f1)
    cms += confusion_matrix(y_va, y_pred)

    print(f'Fold{fold}: precision={p:.3f} recall={r:.3f} F1={f1:.3f}')

print('\n===== 5 折平均性能 =====')
print(f'Precision: {np.mean(precs):.3f} ± {np.std(precs):.3f}')
print(f'Recall:    {np.mean(recs):.3f} ± {np.std(recs):.3f}')
print(f'F1-score:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}')
print('混淆矩阵（总和）:')
print(cms)

# 6. 在全量数据上重新训练并保存模型 ---------------------------------------
pipe.fit(X, y)
joblib.dump(pipe, './P4/female_lr_smote.pkl')
print('\n全量模型已保存：female_lr_smote.pkl')

# 7. 给出系数（标准化后的系数大小可反映特征重要性） -----------------------
coef = pd.Series(pipe.named_steps['clf'].coef_[0], index=num_cols)
print('\n逻辑回归系数（标准化后）:')
print(coef.sort_values(ascending=False))
