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


## 第七部分：`sklearn.datasets` 模块详解

`sklearn.datasets` 是 Scikit-learn 中的一个重要模块，它提供了**加载常用数据集、生成模拟数据集、从外部数据源导入数据集**的功能，非常适合用于学习和测试机器学习算法。


### 一、常用数据集加载函数（`load_*`）

这些函数加载的是**内置的标准小型数据集**，通常作为学习和测试用：

| 函数名 | 描述 |
|--------|------|
| `load_iris()` | 鸢尾花数据集（分类） |
| `load_digits()` | 手写数字数据集（分类） |
| `load_wine()` | 葡萄酒分类数据集 |
| `load_breast_cancer()` | 乳腺癌诊断数据集 |
| `load_diabetes()` | 糖尿病数据集（回归） |
| `load_linnerud()` | 体能训练数据集（多输出回归） |

**返回类型**：`Bunch` 对象，类似字典（有 `.data`、`.target`、`.feature_names`、`.DESCR`）

#### ✅ 示例代码

```python
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data.shape)         # 特征数据
print(iris.target.shape)       # 标签数据
print(iris.feature_names)      # 特征名称
print(iris.target_names)       # 标签名称
```

### 二、数据集生成函数（make_*）


用于生成**可控特征的模拟数据集**，常用于测试算法性能和可视化。


| 函数名 | 描述 |
|--------|------|
| `make_classification()` | 生成分类问题数据 |
| `make_regression()` | 生成回归问题数据 |
| `make_blobs()` | 生成聚类问题数据 |
| `make_moons()` | 生成月牙型分类数据 |
| `make_circles()` | 生成圆环型分类数据 |
| `make_multilabel_classification()` | 多标签分类数据 |
| `make_sparse_coded_signal()` | 稀疏信号数据（用于稀疏学习） |

#### ✅ 示例代码

```python
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 生成分类数据
X, y = make_classification(
    n_samples=100,     # 样本数量
    n_features=2,      # 特征数量
    n_redundant=0,     # 冗余特征数量
    n_classes=2        # 类别数量
)

# 可视化分类数据
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Classification Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

###  三、外部大型数据集加载（fetch_*）

适用于**下载大型真实数据集**（需联网，首次下载后缓存本地）。

| 函数名 | 描述 |
|--------|------|
| `fetch_20newsgroups()` | 新闻文本分类数据 |
| `fetch_olivetti_faces()` | 人脸识别数据 |
| `fetch_lfw_people()` | LFW人脸数据集 |
| `fetch_covtype()` | 覆盖类型数据（分类） |
| `fetch_california_housing()` | 加州房价数据（回归） |

这些函数通常用于：

- 自然语言处理（如 `fetch_20newsgroups`）
- 图像识别（如人脸识别数据集）
- 实际回归与分类建模问题

> ⚠️ 注意：`fetch_*` 函数首次运行时会下载数据集，可能需要联网，并会缓存到本地。

#### ✅ 示例代码：新闻文本分类数据

```python
from sklearn.datasets import fetch_20newsgroups

# 加载训练集（也可用 subset='test'）
data = fetch_20newsgroups(subset='train')

# 数据内容与标签
print(f"数据总数：{len(data.data)}")
print(f"目标类别：{data.target_names}")
print("\n第一篇文章内容：")
print(data.data[0])
```

#### ✅ 示例代码：加州房价数据

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing(as_frame=True)
df = housing.frame  # 转为 DataFrame 格式

print(df.head())     # 查看前几行
print(housing.DESCR) # 查看数据集描述
```

###  四、Bunch 对象说明（类似字典）

加载的标准数据集如 load_iris() 返回的是一个 Bunch 类型对象，它的使用类似于字典。

```python
from sklearn.datasets import load_wine
data = load_wine()

print(data.keys())          # 查看所有键
print(data.data[:5])        # 查看前五个样本
print(data.target_names)    # 标签的文字说明
```

###  五、综合示例：手写数字识别

```python
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

plt.gray()
plt.matshow(digits.images[0])  # 显示第一张图片
plt.title(f'Label: {digits.target[0]}')
plt.show()
```

### 🧾 总结

| 类型 | 函数前缀 | 用途 |
|------|----------|------|
| 小型内置数据集 | `load_` | 快速演示与教学 |
| 模拟数据集生成 | `make_` | 测试算法和可视化 |
| 大型真实数据 | `fetch_` | 实际应用与模型训练 |


## 第八部分：model_selection模块

`sklearn.model_selection` 是 Scikit-learn 中用于**模型选择与评估**的核心模块，提供了数据划分、交叉验证、模型评估与超参数搜索等功能。


### 1. 数据划分：`train_test_split`

将数据划分为训练集与测试集（或验证集），常用于模型训练与评估。

#### ✅ 常用参数

- `test_size`：测试集比例（如 0.2）
- `train_size`：训练集比例（默认是 1 - test_size）
- `shuffle`：是否打乱数据（默认 True）
- `random_state`：随机种子（保证可复现）
- `stratify`：按标签分层划分（分类任务推荐）

#### ✅ 示例

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 2. 交叉验证：cross_val_score 与 cross_validate

#### ✅ cross_val_score：快速评估模型性能

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier()

scores = cross_val_score(model, X, y, cv=5)
print("交叉验证得分：", scores)
print("平均得分：", scores.mean())
```

#### ✅ cross_validate：获取更多评估指标

```python
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC

result = cross_validate(SVC(), X, y, cv=5, return_train_score=True)
print(result['train_score'])
print(result['test_score'])
```

### 3. K 折划分器：KFold、StratifiedKFold 等

用于自定义交叉验证的数据划分策略，适用于更精细的控制。

| 划分器 | 说明 |
|--------|------|
| `KFold` | 将数据均分为 K 折，适用于回归任务或类别均衡的数据 |
| `StratifiedKFold` | 每一折中保持类别比例一致（推荐用于分类问题） |
| `GroupKFold` | 同一组的样本不会被划分到不同折（如患者编号） |

#### ✅ `KFold` 示例

```python
from sklearn.model_selection import KFold
import numpy as np

X = np.arange(10).reshape((5, 2))  # 5个样本，2个特征
kf = KFold(n_splits=3)

for train_index, test_index in kf.split(X):
    print("Train:", train_index, "Test:", test_index)
```

### 4. 超参数搜索：GridSearchCV 与 RandomizedSearchCV

#### ✅ GridSearchCV：网格搜索（穷举法）

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=3)
grid.fit(X, y)

print("最佳参数：", grid.best_params_)
print("最佳得分：", grid.best_score_)
```

#### ✅ RandomizedSearchCV：随机搜索（节省时间）

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10)
}

rand_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, n_iter=10, cv=3)
rand_search.fit(X, y)

print("最佳参数：", rand_search.best_params_)
```

### 5. 学习曲线与验证曲线：learning_curve 与 validation_curve

#### ✅ learning_curve：查看训练集规模对模型性能的影响

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

train_sizes, train_scores, test_scores = learning_curve(SVC(), X, y, cv=5)

plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Test')
plt.legend()
plt.title("Learning Curve")
plt.show()
```

#### ✅ validation_curve：查看某个超参数的变化对性能影响

```python
from sklearn.model_selection import validation_curve

param_range = [0.1, 1, 10]
train_scores, test_scores = validation_curve(
    SVC(), X, y, param_name="C", param_range=param_range, cv=5
)

plt.plot(param_range, train_scores.mean(axis=1), label='Train')
plt.plot(param_range, test_scores.mean(axis=1), label='Test')
plt.xscale("log")
plt.xlabel("C")
plt.title("Validation Curve")
plt.legend()
plt.show()
```

### 🧾 总结表

| 功能类别       | 对应函数 |
|----------------|----------|
| 数据划分       | `train_test_split` |
| 交叉验证评估   | `cross_val_score`，`cross_validate` |
| 自定义划分器   | `KFold`, `StratifiedKFold`, `GroupKFold` |
| 超参数搜索     | `GridSearchCV`, `RandomizedSearchCV` |
| 学习曲线分析   | `learning_curve`, `validation_curve` |


## 第九部分：`sklearn.preprocessing` 模块详解

`sklearn.preprocessing` 模块包含了一系列用于数据预处理的工具，帮助我们对数据进行转换、归一化、标准化等操作，提升机器学习模型的表现和收敛速度。


### 1. 常用预处理方法

#### 1.1 标准化：`StandardScaler`

将特征数据转换为均值为0，标准差为1的分布。

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 1.2 归一化：MinMaxScaler

将数据缩放到指定区间（默认 [0,1]）。

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

#### 1.3 最大绝对值缩放：MaxAbsScaler

将数据按最大绝对值缩放到 [-1, 1]，适合稀疏数据。

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)
```

#### 1.4 归一化（范数缩放）：Normalizer

按行进行缩放，使每个样本的范数（L1、L2等）为1。

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm='l2')
X_normalized = normalizer.fit_transform(X)
```

### 2. 编码类别数据

#### 2.1 标签编码：LabelEncoder

将类别标签转换为数字编码。

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = ['cat', 'dog', 'cat', 'bird']
y_encoded = le.fit_transform(y)
```

#### 2.2 独热编码（One-Hot Encoding）：OneHotEncoder

将类别变量转换成独热编码矩阵。

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False)
X_cat = [['Male'], ['Female'], ['Female']]
X_encoded = enc.fit_transform(X_cat)
```

### 3. 特征生成与多项式特征

#### 3.1 多项式特征：PolynomialFeatures

生成多项式及交叉特征，用于扩展模型能力。

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

### 4. 自定义转换器：FunctionTransformer

通过用户自定义函数对数据进行转换。

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def log_transform(x):
    return np.log1p(x)

transformer = FunctionTransformer(log_transform)
X_transformed = transformer.fit_transform(X)
```

### 5. 缺失值处理

虽然主要缺失值处理在 sklearn.impute 模块，但 preprocessing 中也可以配合使用数据转换。


### 🧾 总结

| 功能          | 常用类/函数               | 说明                         |
|---------------|--------------------------|------------------------------|
| 标准化        | `StandardScaler`         | 零均值单位方差               |
| 归一化        | `MinMaxScaler`           | 线性缩放到指定范围           |
| 最大绝对值缩放| `MaxAbsScaler`           | 缩放到[-1,1]                 |
| 样本归一化    | `Normalizer`             | 按范数缩放样本               |
| 标签编码      | `LabelEncoder`           | 类别标签转数字               |
| 独热编码      | `OneHotEncoder`          | 类别变量转独热码             |
| 多项式特征    | `PolynomialFeatures`     | 生成多项式与交叉特征         |
| 自定义转换    | `FunctionTransformer`    | 通过函数自定义转换           |


## 第十部分：sklearn.ensemble模块

`sklearn.ensemble` 模块包含了多个**集成学习方法（Ensemble Methods）**，主要分为 **Bagging（装袋）**、**Boosting（提升）** 和 **Stacking（堆叠）** 三大类。它通过结合多个基学习器的预测结果，提高模型的准确性与鲁棒性。


### 1. Bagging 方法：并行训练多个模型

#### ✅ `BaggingClassifier` 与 `BaggingRegressor`

使用同一个基学习器在不同数据子集上训练多个模型，然后进行平均（回归）或投票（分类）。

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)
bag.fit(X_train, y_train)
print("Accuracy:", bag.score(X_test, y_test))
```

### 2. Boosting 方法：逐步改进错误

#### ✅ AdaBoostClassifier 与 AdaBoostRegressor

适用于弱模型（如决策树桩），通过调整样本权重来迭代训练。

```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
print("Accuracy:", ada.score(X_test, y_test))
```

#### ✅ GradientBoostingClassifier 与 GradientBoostingRegressor

通过梯度下降的方法优化残差，效果好，参数多，适合调优。

```python
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gbc.fit(X_train, y_train)
print("Accuracy:", gbc.score(X_test, y_test))
```

### 3. 随机森林（Random Forest）

#### ✅ RandomForestClassifier 与 RandomForestRegressor

集成多棵决策树，通过 Bagging 和特征随机子集来增强泛化能力。

```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rfc.fit(X_train, y_train)
print("Accuracy:", rfc.score(X_test, y_test))
```

特点：

+ 抗过拟合能力强
+ 可评估特征重要性（feature_importances_）
+ 支持并行训练

### 4. Stacking 方法（堆叠）

#### ✅ StackingClassifier 与 StackingRegressor

将多个模型的输出作为新特征，再训练一个元模型（meta model）进行最终预测。

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svc', SVC(probability=True))
]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack.fit(X_train, y_train)
print("Accuracy:", stack.score(X_test, y_test))
```

### 🧾 总结表

| 方法类别   | 主要类名                                     | 说明                         |
|------------|----------------------------------------------|------------------------------|
| Bagging    | `BaggingClassifier`, `BaggingRegressor`      | 并行训练多个基学习器，减少方差 |
| Boosting   | `AdaBoost*`, `GradientBoosting*`             | 顺序训练基学习器，优化残差     |
| 随机森林   | `RandomForestClassifier`, `RandomForestRegressor` | Bagging 的改进版，加入特征随机性 |
| Stacking   | `StackingClassifier`, `StackingRegressor`    | 多模型融合，用元学习器组合预测结果 |

## 第十一部分：sklearn.metrics

`sklearn.metrics` 模块提供了用于模型**评估和度量指标**的丰富函数，支持分类、回归、聚类、排序、距离、概率评分等多种任务。

### 1️⃣ 分类评估指标

#### ✅ 准确率（Accuracy）

```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 2, 2]
y_pred = [0, 2, 1, 2]
accuracy_score(y_true, y_pred)  # 输出 0.5
```

#### ✅ 精确率 / 召回率 / F1 分数

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y_true, y_pred, average='macro')  # 平均精确率
recall_score(y_true, y_pred, average='macro')      # 平均召回率
f1_score(y_true, y_pred, average='macro')          # 平均F1
```
+ average='macro'：对每个类计算得分再平均。
+ average='micro'：累计全局 TP/FP/FN 再计算。
+ average='weighted'：考虑类别样本数量的加权平均。

#### ✅ 混淆矩阵（Confusion Matrix）

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_true, y_pred)
```

#### ✅ 分类报告（classification_report）

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```

#### ✅ ROC 曲线 & AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score

y_true = [0, 0, 1, 1]
y_score = [0.1, 0.4, 0.35, 0.8]
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc_score(y_true, y_score)  # 输出 AUC 值
```

### 2️⃣ 回归评估指标

#### ✅ MSE / RMSE / MAE

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

mean_squared_error(y_true, y_pred)  # MSE
np.sqrt(mean_squared_error(y_true, y_pred))  # RMSE
mean_absolute_error(y_true, y_pred)  # MAE
```

#### ✅ R² 分数（决定系数）

```python
from sklearn.metrics import r2_score

r2_score(y_true, y_pred)  # 越接近1越好
```

### 3️⃣ 聚类评估指标

```python
from sklearn.metrics import adjusted_rand_score, silhouette_score

# 假设 labels_true 和 labels_pred 是聚类标签
adjusted_rand_score(labels_true, labels_pred)
silhouette_score(X, cluster_labels)  # 轮廓系数
```

### 4️⃣ 距离度量

```python
from sklearn.metrics import pairwise_distances

pairwise_distances([[0, 1]], [[1, 0]], metric='euclidean')  # 欧氏距离
```

### 5️⃣ 自定义打分函数：make_scorer

用于在交叉验证或 GridSearchCV 中自定义评估指标。

```python
from sklearn.metrics import make_scorer

custom_scorer = make_scorer(mean_squared_error, greater_is_better=False)
```

### 🧾 总结表

| 任务类型 | 常用指标函数                                                                 | 说明                           |
|----------|------------------------------------------------------------------------------|--------------------------------|
| 分类     | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `classification_report` | 评估分类模型性能（精度、召回、AUC 等） |
| 回归     | `mean_squared_error`, `mean_absolute_error`, `r2_score`                     | 评估回归模型误差和拟合程度     |
| 聚类     | `adjusted_rand_score`, `silhouette_score`                                   | 无监督聚类与真实标签一致性     |
| 距离     | `pairwise_distances`                                                        | 计算样本对之间的距离           |
| 自定义评分 | `make_scorer`                                                               | 自定义用于交叉验证和搜索的评分函数 |


## 第十二部分：sklearn 其他模块 

除了常用的模型与评估模块，`scikit-learn` 还包含许多辅助模块，支持数据处理、模型持久化、管道组合等任务。

### 1️⃣ sklearn.pipeline：构建数据处理与模型训练流程

可将数据预处理与模型打包成流水线，便于自动化训练与调参。

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
pipe.fit(X_train, y_train)
```

### 2️⃣ sklearn.compose：组合预处理器

支持对不同列应用不同的转换（如分类特征 vs 数值特征）。

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [0, 1]),
        ('cat', OneHotEncoder(), [2])
    ]
)
```

### 3️⃣ sklearn.feature_selection：特征选择

用于选出最相关的特征，提高模型效率。

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)
```

### 4️⃣ sklearn.feature_extraction：特征提取（文本/图像）

+ 文本：CountVectorizer, TfidfVectorizer
+ 图像：image.extract_patches_2d

```python
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X_text = vec.fit_transform(["I love sklearn", "scikit-learn is powerful"])
```

### 5️⃣ sklearn.impute：缺失值处理

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X_missing)
```

### 6️⃣ sklearn.decomposition：降维方法（如 PCA）

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

### 7️⃣ sklearn.manifold：流形学习（如 t-SNE）

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(X)
```

### 8️⃣ sklearn.utils：实用工具函数

包括打乱数据、设置随机种子、稀疏矩阵转换等。

```python
from sklearn.utils import shuffle

X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
```

### 🧾 总结表

| 模块                     | 功能说明                                | 常用类/函数示例                          |
|--------------------------|-----------------------------------------|------------------------------------------|
| `pipeline`               | 构建训练流水线                          | `Pipeline`, `make_pipeline`              |
| `compose`                | 不同列的预处理组合                      | `ColumnTransformer`                      |
| `feature_selection`      | 特征选择                                | `SelectKBest`, `RFE`, `SelectFromModel`  |
| `feature_extraction`     | 文本/图像特征提取                       | `CountVectorizer`, `TfidfVectorizer`     |
| `impute`                 | 缺失值填充                              | `SimpleImputer`, `KNNImputer`            |
| `decomposition`          | 降维                                    | `PCA`, `TruncatedSVD`                    |
| `manifold`               | 非线性降维 / 可视化                     | `TSNE`, `Isomap`                         |
| `utils`                  | 工具函数                                | `shuffle`, `resample`, `Bunch`           |





