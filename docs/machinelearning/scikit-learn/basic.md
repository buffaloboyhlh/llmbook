# scikit-learn 手册

## 第一部分：基础

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


| 步骤 | 方法 |
|------|------|
| 1️⃣ 数据预处理 | `preprocessing` |
| 2️⃣ 划分训练集测试集 | `train_test_split()` |
| 3️⃣ 选择模型 | `LogisticRegression()`、`SVC()` 等 |
| 4️⃣ 模型训练 | `.fit(X_train, y_train)` |
| 5️⃣ 模型预测 | `.predict(X_test)` |
| 6️⃣ 模型评估 | `accuracy_score`、`classification_report` 等 |

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

## 第二部分：核心模块

| 模块名 | 说明 |
|--------|------|
| `sklearn.datasets` | 提供内置数据集、模拟数据生成器、外部数据加载工具 |
| `sklearn.model_selection` | 数据划分、交叉验证、网格搜索等模型选择工具 |
| `sklearn.preprocessing` | 特征缩放、归一化、编码、缺失值填充等预处理方法 |
| `sklearn.linear_model` | 各类线性模型，如线性回归、逻辑回归、Lasso、Ridge |
| `sklearn.tree` | 决策树模型，包括分类与回归树 |
| `sklearn.ensemble` | 集成方法，如随机森林、梯度提升、投票法等 |
| `sklearn.svm` | 支持向量机，用于分类和回归任务 |
| `sklearn.naive_bayes` | 朴素贝叶斯分类器，如高斯、伯努利、多项式贝叶斯 |
| `sklearn.neighbors` | K近邻算法，用于分类与回归 |
| `sklearn.metrics` | 评估指标模块，支持分类、回归、聚类等评估 |
| `sklearn.pipeline` | 构建数据处理 + 模型训练的一体化流程 |

### 1. sklearn.datasets 

#### 🧠 一、datasets 是什么？

sklearn.datasets 提供了：

+ ✅ 内置经典数据集（如鸢尾花、波士顿房价）
+ ✅ 下载真实数据集（如 20 类新闻文本）
+ ✅ 生成模拟数据（用于分类、回归、聚类等）

#### 📚 二、内置小型数据集（常用于入门练习）

| 函数名 | 描述 |
|--------|------|
| `load_iris()` | 鸢尾花数据集（分类） |
| `load_digits()` | 手写数字数据集 |
| `load_diabetes()` | 糖尿病数据集（回归） |
| `load_wine()` | 葡萄酒分类数据 |
| `load_breast_cancer()` | 乳腺癌二分类数据 |
| `load_linnerud()` | 人体指标回归数据 |


#### 🌐 三、从外部下载数据集（真实世界）

| 函数名 | 描述 |
|--------|------|
| `fetch_20newsgroups()` | 20 类新闻文本（NLP） |
| `fetch_california_housing()` | 加州房价数据（回归） |
| `fetch_covtype()` | 森林覆盖类型数据集 |
| `fetch_olivetti_faces()` | Olivetti 人脸识别图像数据 |
| `fetch_lfw_people()` | LFW（Labeled Faces in the Wild）人脸图像 |
| `fetch_lfw_pairs()` | LFW 脸部配对图像（人脸验证） |


#### 🧪 四、生成模拟数据（建模与测试用）

| 函数 | 用途 |
|------|------|
| `make_classification()` | 生成用于分类的样本数据 |
| `make_regression()` | 生成用于回归的样本数据 |
| `make_blobs()` | 生成用于聚类测试的高斯分布数据 |
| `make_moons()` | 生成双月形状的非线性分类数据 |
| `make_circles()` | 生成同心圆形的二分类数据 |
| `make_multilabel_classification()` | 生成多标签分类问题数据 |
| `make_sparse_spd_matrix()` | 生成稀疏对称正定矩阵（如图模型结构） |


### 2. sklearn.model_selection

#### 📌 一、model_selection 是什么？

这是 scikit-learn 中负责：

+	数据划分（训练集/测试集）
+	模型评估（交叉验证）
+	超参数搜索（网格搜索、随机搜索）
+	模型选择与验证策略的核心模块


#### 🧩 二、常用功能分类与作用

| 功能类别 | 常用函数 | 作用 |
|----------|----------|------|
| 数据集划分 | `train_test_split` | 训练集 / 测试集划分 |
| 交叉验证 | `cross_val_score`, `cross_validate`, `KFold`, `StratifiedKFold` | 多折评估模型 |
| 超参数搜索 | `GridSearchCV`, `RandomizedSearchCV` | 网格/随机搜索参数 |
| 学习曲线 | `learning_curve`, `validation_curve` | 模型学习过程可视化 |
| 预定义验证 | `ShuffleSplit`, `LeaveOneOut` 等 | 控制验证集划分方式 |


#### ✂️ 三、数据划分：train_test_split()

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)
```

+	test_size：测试集比例（如 0.2）
+	stratify=y：按标签比例分层采样（分类常用）


#### 🔁 四、交叉验证

##### 1️⃣ cross_val_score()

快速评估模型交叉验证得分：

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

scores = cross_val_score(LogisticRegression(), X, y, cv=5)
print("平均准确率：", scores.mean())
```

##### 2️⃣ cross_validate()（支持更多输出）

```python
from sklearn.model_selection import cross_validate

result = cross_validate(LogisticRegression(), X, y,
                        scoring=['accuracy', 'f1_macro'],
                        return_train_score=True,
                        cv=5)
print(result)
```

#### 🔀 五、交叉验证策略类（KFold 等）

| 类名 | 说明 |
|------|------|
| `KFold` | 简单均匀划分为 K 折 |
| `StratifiedKFold` | 保持类别分布的 K 折（适用于分类问题） |
| `ShuffleSplit` | 多次随机划分训练/测试集 |
| `LeaveOneOut` | 留一法（每次留一个样本做测试集） |
| `GroupKFold` | 按组划分，确保同一组数据不在训练和验证集中同时出现 |


```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skf.split(X, y):
    print(train_idx, test_idx)
```

#### 🔍 六、超参数搜索

##### ✅ GridSearchCV（网格搜索）

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("最优参数：", grid.best_params_)
print("最优得分：", grid.best_score_)
```

##### ✅ RandomizedSearchCV（随机搜索）

更适合大搜索空间：

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

param_dist = {'C': uniform(0.1, 10)}
rand_search = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=10, cv=5)
rand_search.fit(X_train, y_train)
```

#### 📈 七、学习曲线 & 验证曲线（可视化模型学习能力）

##### 学习曲线：随着样本量增加的性能变化

```python
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

train_sizes, train_scores, test_scores = learning_curve(LogisticRegression(), X, y, cv=5)

plt.plot(train_sizes, train_scores.mean(axis=1), label='train')
plt.plot(train_sizes, test_scores.mean(axis=1), label='test')
plt.legend()
plt.title("Learning Curve")
plt.show()
```

##### 验证曲线：随着超参数变化模型性能如何

```python
from sklearn.model_selection import validation_curve

param_range = [0.01, 0.1, 1, 10]
train_scores, test_scores = validation_curve(
    LogisticRegression(), X, y,
    param_name="C", param_range=param_range,
    cv=5)

plt.plot(param_range, train_scores.mean(axis=1), label="Train")
plt.plot(param_range, test_scores.mean(axis=1), label="Test")
plt.xscale('log')
plt.title("Validation Curve")
plt.legend()
plt.show()
```

### 3. sklearn.preprocessing

#### ✅ 一、模块作用概述

`sklearn.preprocessing` 提供各种数据预处理方法：

| 预处理类别 | 典型操作 |
|------------|----------|
| 特征缩放 | 标准化、归一化、最大最小缩放 |
| 非数值特征编码 | LabelEncoding、OneHotEncoding |
| 非线性转换 | 对数、幂次、分位数变换 |
| 缺失值处理 | 插值、填充 |

#### 📊 二、常用预处理器总览

| 名称 | 类/函数 | 用途 |
|------|---------|------|
| 标准化 | `StandardScaler` | 转换为均值 0 方差 1 |
| 归一化 | `MinMaxScaler` | 缩放到 [0, 1] 区间 |
| 最大绝对缩放 | `MaxAbsScaler` | 缩放到 [-1, 1] |
| 稀疏缩放 | `RobustScaler` | 缩放不受异常值影响 |
| 单位向量化 | `Normalizer` | 将每行样本缩放为单位范数 |
| 类别编码 | `LabelEncoder`, `OneHotEncoder` | 标签编码或独热编码 |
| 缺失值填充 | `SimpleImputer` | 用均值/中位数/众数填补缺失 |
| 功能转换 | `FunctionTransformer`, `PowerTransformer`, `QuantileTransformer` | 自定义变换、Box-Cox/Yeo-Johnson、分位数变换 |


#### 📐 三、常用缩放类详解

##### 1️⃣ StandardScaler：标准化

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

##### 2️⃣ MinMaxScaler：归一化

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

##### 3️⃣ RobustScaler：抗异常值缩放

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

##### 4️⃣ Normalizer：单位范数缩放（行方向）

```python
from sklearn.preprocessing import Normalizer

scaler = Normalizer()
X_scaled = scaler.fit_transform(X)
```

#### 🔤 四、类别变量编码器

##### LabelEncoder：标签编码

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(['yes', 'no', 'yes', 'no'])
```

##### OneHotEncoder：独热编码

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False)
X = [['Male'], ['Female'], ['Male']]
X_encoded = enc.fit_transform(X)
```

#### 🧪 五、缺失值处理

##### SimpleImputer：均值/中位数填补

```python
from sklearn.impute import SimpleImputer
import numpy as np

imp = SimpleImputer(strategy='mean')
X = np.array([[1, 2], [np.nan, 3], [7, 6]])
X_filled = imp.fit_transform(X)
```

#### 🔁 六、非线性变换

##### PowerTransformer：Box-Cox / Yeo-Johnson

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)
```

##### QuantileTransformer：分位数变换

```python
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution='normal')
X_transformed = qt.fit_transform(X)
```

#### 🧾 七、小结：预处理器选择建议表

| 场景 | 推荐方法 |
|------|-----------|
| 数据标准化 | `StandardScaler` |
| 异常值存在 | `RobustScaler` |
| 特征在不同量纲 | `MinMaxScaler` |
| 样本为向量数据 | `Normalizer` |
| 分类标签转换 | `LabelEncoder`, `OneHotEncoder` |
| 数据缺失 | `SimpleImputer` |
| 特征非正态分布 | `PowerTransformer`, `QuantileTransformer` |


#### ✅ 八、配合 Pipeline 使用


```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)
```

### 4. sklearn.metrics


#### ✅ 一、模块用途

`sklearn.metrics` 是用于评估模型性能的模块，支持：

- 分类任务评估指标（如准确率、F1 值）
- 回归任务评估指标（如均方误差、R²）
- 聚类评估指标（如轮廓系数、ARI）
- 多标签、多输出、多分类任务支持

#### 🎯 二、分类任务评估指标

| 函数 | 含义 | 用法举例 |
|------|------|----------|
| `accuracy_score()` | 准确率 | 分类正确的样本数 / 总样本数 |
| `precision_score()` | 精确率 | 正类预测中正确的比例 |
| `recall_score()` | 召回率 | 实际正类中被正确识别的比例 |
| `f1_score()` | F1 值 | 精确率和召回率的调和平均 |
| `classification_report()` | 分类整体报告 | 多个指标综合 |
| `confusion_matrix()` | 混淆矩阵 | TP/FP/FN/TN |
| `roc_auc_score()` | ROC 曲线下的面积 | 二分类性能 |
| `log_loss()` | 对数损失 | 概率预测误差 |


```python
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

y_true = [0, 1, 1, 1, 0, 0, 1]
y_pred = [0, 1, 0, 1, 0, 1, 1]

print("准确率：", accuracy_score(y_true, y_pred))
print("F1 值：", f1_score(y_true, y_pred))
print("报告：\n", classification_report(y_true, y_pred))
print("混淆矩阵：\n", confusion_matrix(y_true, y_pred))
```

#### 📉 三、回归任务评估指标


| 函数 | 含义 | 用法 |
|------|------|------|
| `mean_squared_error()` | 均方误差（MSE） | 均方误差越小越好 |
| `mean_absolute_error()` | 平均绝对误差（MAE） | 更少受离群点影响 |
| `r2_score()` | 决定系数 R² | 越接近 1 越好 |
| `mean_squared_log_error()` | 均方对数误差 | 适合对数回归问题 |

```python
from sklearn.metrics import mean_squared_error, r2_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

print("MSE:", mean_squared_error(y_true, y_pred))
print("R²:", r2_score(y_true, y_pred))
```

#### 🧠 四、聚类任务评估指标

| 函数 | 含义 | 特点 |
|------|------|------|
| `adjusted_rand_score()` | 调整后的兰德指数（ARI） | 无监督聚类标签相似性 |
| `normalized_mutual_info_score()` | 归一化互信息 | 聚类标签一致性 |
| `homogeneity_score()` | 同质性评分 | 每个聚类只包含一种标签 |
| `completeness_score()` | 完整性评分 | 每种标签只在一个聚类中 |
| `silhouette_score()` | 轮廓系数 | 用于度量聚类分离程度 |


```python
from sklearn.metrics import adjusted_rand_score

labels_true = [0, 0, 1, 1, 2, 2]
labels_pred = [0, 0, 1, 2, 2, 2]

print("ARI:", adjusted_rand_score(labels_true, labels_pred))
```

#### 🧾 五、常用指标速查表

##### 📊 分类指标速查表

| 指标 | 函数 | 说明 |
|------|------|------|
| 准确率 | `accuracy_score` | 总体分类正确率 |
| 精确率 | `precision_score` | 正类中预测正确的比例 |
| 召回率 | `recall_score` | 实际正类中被预测正确的比例 |
| F1 值 | `f1_score` | 平衡精确率与召回率 |
| 混淆矩阵 | `confusion_matrix` | 预测 vs 真实标签 |
| 分类报告 | `classification_report` | 多个指标汇总展示 |


##### 📈 回归指标速查表

| 指标 | 函数 | 说明 |
|------|------|------|
| MSE | `mean_squared_error` | 对误差平方的平均值 |
| MAE | `mean_absolute_error` | 绝对误差的平均值 |
| R² | `r2_score` | 回归拟合优度，越接近 1 越好 |

##### 🧠 聚类指标速查表

| 指标 | 函数 | 说明 |
|------|------|------|
| ARI | `adjusted_rand_score` | 聚类标签一致性 |
| NMI | `normalized_mutual_info_score` | 标签信息共享程度 |
| Homogeneity | `homogeneity_score` | 每个簇是否只包含一个类 |
| Completeness | `completeness_score` | 每个类是否集中在一个簇 |
| Silhouette | `silhouette_score` | 聚类的分离性和紧密度 |

#### 🧠 六、建议使用场景总结

| 任务类型 | 推荐指标 |
|----------|----------|
| 二分类 | 准确率、精确率、召回率、F1、AUC |
| 多分类 | 分类报告、混淆矩阵 |
| 回归 | MSE、MAE、R² |
| 聚类 | ARI、NMI、轮廓系数 |

#### ✅ 七、补充说明：多标签/多分类设置参数

- `average='binary'`：默认，用于二分类
- `average='micro'`：全局累计 TP/FP/FN
- `average='macro'`：各类别指标简单平均
- `average='weighted'`：加权平均（按支持度）
- `labels=`：指定参与计算的标签类别索引


### 5. sklearn.linear_model

