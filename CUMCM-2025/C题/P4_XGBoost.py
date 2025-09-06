import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, fbeta_score, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import joblib
from collections import Counter

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载女胎数据
data = pd.read_csv('附件_女胎数据.csv')

# 构造标签：异常为1，正常为0
y = data['染色体的非整倍体'].notnull().astype(int)
print(f"类别分布: {Counter(y)}")
print(f"异常比例: {y.mean():.3f}")

# 基础特征
base_features = ['21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值',
                'X染色体的Z值', 'GC含量', '孕妇BMI', '唯一比对的读段数']

X = data[base_features].copy()

# 填充数值特征的缺失值
for col in base_features:
    if X[col].isna().any():
        median_val = X[col].median()
        X.loc[:, col] = X[col].fillna(median_val)

# 特征工程：构造新特征
X['Z值最大绝对值'] = X[['21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值']].abs().max(axis=1)
X['Z值均值'] = X[['21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值']].mean(axis=1)
X['Z值标准差'] = X[['21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值']].std(axis=1)
X['GC含量偏差'] = abs(X['GC含量'] - 0.5)  # 偏离正常GC含量
X['X染色体Z值绝对值'] = abs(X['X染色体的Z值'])
X['Z值异常标志'] = ((X['21号染色体的Z值'].abs() > 3) |
                   (X['18号染色体的Z值'].abs() > 3) |
                   (X['13号染色体的Z值'].abs() > 3)).astype(int)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 计算类别权重
class_weight = len(y[y==0]) / len(y[y==1])
print(f"类别权重 (正常:异常): {class_weight:.2f}")

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 定义参数网格
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

# 使用网格搜索进行参数优化
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=class_weight,
    random_state=42
)

# 使用分层交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("开始网格搜索...")
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.3f}")

# 使用最佳参数训练最终模型
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# 预测概率
y_prob = best_model.predict_proba(X_test)[:, 1]

# 寻找最佳阈值（优先召回率）
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# 寻找多个目标召回率的最佳阈值
target_recalls = [0.6, 0.7, 0.8]
best_thresholds = {}
best_f2_scores = {}

for target_recall in target_recalls:
    best_threshold = 0.5
    best_f2 = 0

    for i in range(len(thresholds)):
        if recalls[i] >= target_recall:
            current_f2 = fbeta_score(y_test, (y_prob >= thresholds[i]).astype(int), beta=2)
            if current_f2 > best_f2:
                best_f2 = current_f2
                best_threshold = thresholds[i]

    best_thresholds[target_recall] = best_threshold
    best_f2_scores[target_recall] = best_f2

print("\n不同目标召回率的最佳阈值:")
for recall, threshold in best_thresholds.items():
    print(f"召回率 ≥ {recall}: 阈值 = {threshold:.3f}, F2分数 = {best_f2_scores[recall]:.3f}")

# 选择召回率 ≥ 0.7 的阈值
selected_threshold = best_thresholds[0.7]
y_pred_optimized = (y_prob >= selected_threshold).astype(int)

# 输出评估结果
print(f"\n=== 模型性能 (阈值={selected_threshold:.3f}) ===")
print("分类报告:")
print(classification_report(y_test, y_pred_optimized))

print("混淆矩阵:")
cm = confusion_matrix(y_test, y_pred_optimized)
print(cm)

# 计算各种评估指标
f2_score_val = fbeta_score(y_test, y_pred_optimized, beta=2)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_optimized, average='binary')

print(f"精确率: {precision:.3f}")
print(f"召回率: {recall:.3f}")
print(f"F1分数: {f1:.3f}")
print(f"F2分数 (beta=2): {f2_score_val:.3f}")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(15, 5))

# ROC曲线
plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# 精确率-召回率曲线
plt.subplot(1, 3, 2)
plt.plot(recalls[:-1], precisions[:-1], color='blue', lw=2)
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('精确率-召回率曲线')
plt.grid(True, alpha=0.3)

# 阈值曲线
plt.subplot(1, 3, 3)
plt.plot(thresholds, precisions[:-1], label='精确率', linewidth=2)
plt.plot(thresholds, recalls[:-1], label='召回率', linewidth=2)
plt.axvline(x=selected_threshold, color='r', linestyle='--', label=f'选择阈值 ({selected_threshold:.3f})')
plt.xlabel('阈值')
plt.ylabel('分数')
plt.title('精确率-召回率 vs 阈值')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./P4/model_evaluation_curves.png', dpi=300, bbox_inches='tight')

# 特征重要性
feature_importance = best_model.feature_importances_
feature_names_cn = ['21号染色体Z值', '18号染色体Z值', '13号染色体Z值',
                   'X染色体Z值', 'GC含量', '孕妇BMI', '唯一比对读段数',
                   'Z值最大绝对值', 'Z值均值', 'Z值标准差', 'GC含量偏差',
                   'X染色体Z值绝对值', 'Z值异常标志']

feat_imp = pd.Series(feature_importance, index=feature_names_cn)
plt.figure(figsize=(12, 8))
feat_imp.sort_values(ascending=True).plot(kind='barh', title='XGBoost特征重要性')
plt.tight_layout()
plt.savefig('./P4/xgboost_feature_importance.png', dpi=300, bbox_inches='tight')

# 保存模型和相关信息
joblib.dump(best_model, './P4/female_xgboost.pkl')
joblib.dump(scaler, './P4/scaler.pkl')
joblib.dump(best_thresholds, './P4/best_thresholds.pkl')

print("\n最终模型已保存: female_xgboost.pkl")
print("最佳阈值已保存: best_thresholds.pkl")

# 输出详细的性能分析
print("\n=== 详细性能分析 ===")
print(f"测试集中异常样本数量: {y_test.sum()}")
print(f"测试集中正常样本数量: {len(y_test) - y_test.sum()}")
print(f"模型检测到的异常样本: {y_pred_optimized.sum()}")
print(f"真正例 (TP): {cm[1, 1]}")
print(f"假正例 (FP): {cm[0, 1]}")
print(f"假反例 (FN): {cm[1, 0]}")
print(f"真反例 (TN): {cm[0, 0]}")

# 计算临床相关指标
sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
ppv = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
npv = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0

print(f"\n临床指标:")
print(f"灵敏度 (召回率): {sensitivity:.3f}")
print(f"特异度: {specificity:.3f}")
print(f"阳性预测值 (精确率): {ppv:.3f}")
print(f"阴性预测值: {npv:.3f}")
