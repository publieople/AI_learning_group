import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# 加载优化后的模型和标准化器
model = joblib.load('./P4/female_xgboost.pkl')
scaler = joblib.load('./P4/scaler.pkl')
best_thresholds = joblib.load('./P4/best_thresholds.pkl')

# 加载测试数据
data = pd.read_csv('附件_女胎数据.csv')
y_true = data['染色体的非整倍体'].notnull().astype(int)

# 特征工程函数（与训练时一致）
def prepare_features(data):
    base_features = ['21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值',
                    'X染色体的Z值', 'GC含量', '孕妇BMI', '唯一比对的读段数']

    X = data[base_features].copy()

    # 填充缺失值
    for col in base_features:
        if X[col].isna().any():
            median_val = X[col].median()
            X.loc[:, col] = X[col].fillna(median_val)

    # 构造新特征
    X['Z值最大绝对值'] = X[['21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值']].abs().max(axis=1)
    X['Z值均值'] = X[['21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值']].mean(axis=1)
    X['Z值标准差'] = X[['21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值']].std(axis=1)
    X['GC含量偏差'] = abs(X['GC含量'] - 0.5)
    X['X染色体Z值绝对值'] = abs(X['X染色体的Z值'])
    X['Z值异常标志'] = ((X['21号染色体的Z值'].abs() > 3) |
                       (X['18号染色体的Z值'].abs() > 3) |
                       (X['13号染色体的Z值'].abs() > 3)).astype(int)

    return X

# 准备特征
X = prepare_features(data)

# 标准化特征
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 使用不同阈值进行预测
print("=== 不同阈值下的预测结果 ===")

for target_recall, threshold in best_thresholds.items():
    print(f"\n阈值 = {threshold:.3f} (目标召回率 ≥ {target_recall})")

    # 预测概率
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # 使用指定阈值进行预测
    y_pred = (y_prob >= threshold).astype(int)

    # 计算评估指标
    from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    print(f"精确率: {precision:.3f}")
    print(f"召回率: {recall:.3f}")
    print(f"F1分数: {f1:.3f}")
    print(f"F2分数: {f2:.3f}")

# 使用推荐阈值（召回率 ≥ 0.7）
recommended_threshold = best_thresholds[0.7]
print(f"\n=== 推荐使用阈值: {recommended_threshold:.3f} ===")

y_prob = model.predict_proba(X_scaled)[:, 1]
y_pred = (y_prob >= recommended_threshold).astype(int)

print("全量数据分类报告:")
print(classification_report(y_true, y_pred))

print("混淆矩阵:")
print(confusion_matrix(y_true, y_pred))

# 输出预测结果统计
print(f"\n预测结果统计:")
print(f"总样本数: {len(y_pred)}")
print(f"预测为异常: {y_pred.sum()}")
print(f"预测为正常: {len(y_pred) - y_pred.sum()}")
print(f"实际异常: {y_true.sum()}")
print(f"实际正常: {len(y_true) - y_true.sum()}")

# 特征重要性展示
feature_importance = model.feature_importances_
feature_names = ['21号染色体Z值', '18号染色体Z值', '13号染色体Z值',
                'X染色体Z值', 'GC含量', '孕妇BMI', '唯一比对读段数',
                'Z值最大绝对值', 'Z值均值', 'Z值标准差', 'GC含量偏差',
                'X染色体Z值绝对值', 'Z值异常标志']

feat_imp = pd.Series(feature_importance, index=feature_names)
print("\n=== 特征重要性排序 ===")
print(feat_imp.sort_values(ascending=False))

# 单个样本预测示例
print(f"\n=== 单个样本预测示例 ===")
sample_idx = 0  # 第一个样本
sample_features = X_scaled.iloc[sample_idx:sample_idx+1]
sample_prob = model.predict_proba(sample_features)[:, 1][0]
sample_pred = (sample_prob >= recommended_threshold).astype(int)

print(f"样本 {sample_idx} 预测概率: {sample_prob:.3f}")
print(f"样本 {sample_idx} 预测结果: {'异常' if sample_pred == 1 else '正常'}")
print(f"样本 {sample_idx} 实际标签: {'异常' if y_true.iloc[sample_idx] == 1 else '正常'}")
