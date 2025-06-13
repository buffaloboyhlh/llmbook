# scikit-learn 基础教程

## 第一部分：基础入门

### 1. Scikit-learn 简介
Scikit-learn 是基于 Python 的科学计算库（NumPy/SciPy）构建的机器学习库，具有以下特点：
- 简单高效的数据挖掘和数据分析工具
- 开源、商业可用 - BSD 许可证
- 建立在 NumPy、SciPy 和 matplotlib 之上
- 支持监督学习和无监督学习算法

### 2. 安装与配置
```bash
# 基础安装
pip install scikit-learn

# 完整安装（包含绘图功能）
pip install scikit-learn[alldeps]

# 验证安装
python -c "import sklearn; print(sklearn.__version__)"
```

### 3. 基本工作流程
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 模型训练
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 模型评估
y_pred = clf.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")
```

## 第二部分：中级应用

### 1. 数据预处理进阶
#### 管道(Pipeline)使用
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# 数值型特征处理
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 类别型特征处理
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 组合处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['age', 'income']),
        ('cat', categorical_transformer, ['gender', 'city'])
    ])
```

#### 特征选择
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

# 查看选择的特征
selected_features = [iris.feature_names[i] for i in selector.get_support(indices=True)]
print("选择的特征:", selected_features)
```

### 2. 模型调优技术
#### 网格搜索与随机搜索
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import uniform

# 网格搜索
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 随机搜索
param_dist = {
    'C': uniform(loc=0, scale=4),
    'gamma': uniform(loc=0, scale=1),
    'kernel': ['rbf', 'linear']
}
random_search = RandomizedSearchCV(SVC(), param_dist, n_iter=100, cv=5, n_jobs=-1)
random_search.fit(X_train, y_train)
```

#### 交叉验证策略
```python
from sklearn.model_selection import (KFold, StratifiedKFold, 
                                   TimeSeriesSplit, 
                                   cross_val_score)

# 普通K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 分层K折交叉验证（适用于分类问题）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 使用交叉验证评估模型
scores = cross_val_score(RandomForestClassifier(), X, y, cv=skf, scoring='accuracy')
print(f"交叉验证平均准确率: {scores.mean():.2f} ± {scores.std():.2f}")
```

## 第三部分：高级精通

### 1. 自定义评估指标
```python
from sklearn.metrics import make_scorer

def custom_metric(y_true, y_pred):
    """自定义评估指标示例"""
    true_pos = sum((y_true == 1) & (y_pred == 1))
    false_pos = sum((y_true == 0) & (y_pred == 1))
    return true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0

custom_scorer = make_scorer(custom_metric)

# 在网格搜索中使用自定义指标
grid_search = GridSearchCV(
    SVC(), param_grid, cv=5, scoring=custom_scorer
)
```

### 2. 自定义转换器
```python
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    """对数变换自定义转换器"""
    
    def __init__(self, add_one=True):
        self.add_one = add_one
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if self.add_one:
            return np.log1p(X)
        else:
            return np.log(X)

# 在管道中使用
pipeline = Pipeline([
    ('log', LogTransformer()),
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier())
])
```

### 3. 集成学习与堆叠
```python
from sklearn.ensemble import (VotingClassifier, 
                            StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 投票分类器
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('svm', SVC(probability=True)),
        ('dt', DecisionTreeClassifier())
    ],
    voting='soft'  # 'hard' 或 'soft'
)

# 堆叠分类器
stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('svm', SVC(probability=True)),
        ('dt', DecisionTreeClassifier())
    ],
    final_estimator=LogisticRegression(),
    stack_method='auto',
    n_jobs=-1
)
```

### 4. 处理类别不平衡
```python
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

# 计算类别权重
classes = np.unique(y)
weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight = dict(zip(classes, weights))

# 在模型中使用
model = RandomForestClassifier(class_weight=class_weight)

# 使用SMOTE过采样
smote_pipeline = make_pipeline(
    SMOTE(random_state=42),
    RandomForestClassifier()
)
```

## 第四部分：实战项目

### 项目1：客户流失预测
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 加载数据
data = pd.read_csv('customer_churn.csv')

# 特征工程
X = data.drop(['customerID', 'Churn'], axis=1)
y = data['Churn'].map({'Yes': 1, 'No': 0})

# 数据预处理
X = pd.get_dummies(X, drop_first=True)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 训练模型
gbm = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
gbm.fit(X_train, y_train)

# 评估
y_pred = gbm.predict(X_test)
y_proba = gbm.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC分数: {roc_auc_score(y_test, y_proba):.2f}")

# 特征重要性
pd.Series(gbm.feature_importances_, index=X.columns).sort_values().plot(kind='barh')
```

### 项目2：时间序列预测
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

# 创建时间序列特征
def create_features(df, lags=5):
    for i in range(1, lags+1):
        df[f'lag_{i}'] = df['value'].shift(i)
    return df.dropna()

# 准备数据
series = ...  # 你的时间序列数据
df = create_features(series)

X = df.drop('value', axis=1)
y = df['value']

# 多步预测
X_train, X_test = X.iloc[:-100], X.iloc[-100:]
y_train, y_test = y.iloc[:-100], y.iloc[-100:]

# 使用多输出回归器进行多步预测
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100))
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
```

## 第五部分：性能优化

### 1. 并行处理
```python
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# 使用n_jobs参数
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)  # 使用所有CPU核心

# 使用HistGradientBoosting（内存效率更高）
hgb = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
```

### 2. 增量学习
```python
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

# 适用于大数据集的增量学习
sgd = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)

# 分批训练
for batch in pd.read_csv('large_data.csv', chunksize=1000):
    X_batch = batch.drop('target', axis=1)
    y_batch = batch['target']
    sgd.partial_fit(X_batch, y_batch, classes=np.unique(y))
```

### 3. 模型持久化
```python
import joblib

# 保存模型
joblib.dump(model, 'model.joblib')

# 加载模型
loaded_model = joblib.load('model.joblib')

# 保存整个pipeline
joblib.dump(pipeline, 'pipeline.joblib')
```

## 第六部分：最佳实践

1. **特征工程比算法选择更重要**：数据质量决定模型上限
2. **从简单模型开始**：先尝试线性模型，再逐步复杂化
3. **理解业务需求选择指标**：准确率不是万能的
4. **模型可解释性**：使用SHAP或LIME解释模型决策
5. **监控模型性能**：建立模型性能下降的预警机制

## 学习资源推荐

1. 官方文档：https://scikit-learn.org/stable/
2. Scikit-learn 教程仓库：https://github.com/justmarkham/scikit-learn-videos
3. 书籍：《Python机器学习手册》- Andreas Müller
4. 竞赛平台：Kaggle (https://www.kaggle.com/)

通过本教程的系统学习，您应该已经掌握了 scikit-learn 从基础到高级的各项技能。接下来建议：
1. 参与实际项目积累经验
2. 阅读 scikit-learn 源码理解实现原理
3. 关注机器学习领域最新进展
4. 尝试将模型部署到生产环境