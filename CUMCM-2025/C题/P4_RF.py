import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.impute import SimpleImputer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
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

# 5. 定义模型和管道
pipelines = {
    "Logistic Regression": ImbPipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42, class_weight='balanced'))
    ]),
    "Random Forest": ImbPipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
}

# 6. K折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for model_name, pipeline in pipelines.items():
    print(f"--- 评估 {model_name} ---")
    y_preds, y_probs = [], []
    y_tests_all = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline.fit(X_train, y_train)
        y_preds.extend(pipeline.predict(X_test))
        y_probs.extend(pipeline.predict_proba(X_test)[:, 1])
        y_tests_all.extend(y_test)

    results[model_name] = {
        'y_true': y_tests_all,
        'y_pred': y_preds,
        'y_prob': y_probs
    }

# 7. 生成和保存报告
report_content = "# 模型比较分类报告\n\n"
report_content += "## 数据集信息\n"
report_content += f"- 数据文件: 附件_女胎数据.csv\n"
report_content += f"- 样本数量: {len(X)}\n"
report_content += f"- 特征数量: {len(feat_cols)}\n"
report_content += f"- 异常样本比例: {y.mean():.2%}\n\n"

for model_name, data in results.items():
    # 分类报告（使用字符串格式）
    report_str = str(classification_report(data['y_true'], data['y_pred'], target_names=['正常 (0)', '异常 (1)']))
    report_content += f"## {model_name}\n\n"
    report_content += "### 分类报告\n\n"
    report_content += "```\n"
    report_content += report_str
    report_content += "\n```\n\n"

    # 混淆矩阵
    cm = confusion_matrix(data['y_true'], data['y_pred'])
    report_content += "### 混淆矩阵\n\n"
    report_content += "|  | 预测正常 | 预测异常 |\n"
    report_content += "|--|----------|----------|\n"
    report_content += f"| 实际正常 | {cm[0,0]} (TN) | {cm[0,1]} (FP) |\n"
    report_content += f"| 实际异常 | {cm[1,0]} (FN) | {cm[1,1]} (TP) |\n\n"

    # ROC曲线信息
    fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
    roc_auc = auc(fpr, tpr)
    report_content += f"### ROC曲线指标\n"
    report_content += f"- AUC: {roc_auc:.3f}\n\n"

with open("./P4/模型比较分类报告.md", "w", encoding="utf-8") as f:
    f.write(report_content)
print("分类报告已保存到 './P4/模型比较分类报告.md'")

# 8. 生成和保存图表
# ROC曲线
plt.figure(figsize=(10, 8))
for model_name, data in results.items():
    fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线比较')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig("./P4/roc_curve_comparison.png")
print("ROC曲线图已保存到 './P4/roc_curve_comparison.png'")
plt.close()

# Precision-Recall曲线
plt.figure(figsize=(10, 8))
for model_name, data in results.items():
    precision, recall, _ = precision_recall_curve(data['y_true'], data['y_prob'])
    plt.plot(recall, precision, label=f'{model_name}')
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('Precision-Recall曲线比较')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("./P4/pr_curve_comparison.png")
print("PR曲线图已保存到 './P4/pr_curve_comparison.png'")
plt.close()

# 9. 特征重要性可视化（仅针对Random Forest）
rf_pipeline = pipelines["Random Forest"]
rf_pipeline.fit(X, y)
rf_classifier = rf_pipeline.named_steps['classifier']
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=np.array(feat_cols)[indices], palette='Blues_r')
plt.title('随机森林特征重要性（女胎非整倍体）')
plt.tight_layout()
plt.savefig('./P4/随机森林特征重要性（女胎非整倍体）.png')
print("特征重要性图已保存到 './P4/随机森林特征重要性（女胎非整倍体）.png'")
