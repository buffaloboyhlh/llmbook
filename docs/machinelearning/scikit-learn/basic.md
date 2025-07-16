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

| 模块 | 作用 |
|------|------|
| `sklearn.datasets` | 加载内置数据集 |
| `sklearn.model_selection` | 训练集/测试集划分，交叉验证，网格搜索 |
| `sklearn.preprocessing` | 数据标准化、归一化、编码等 |
| `sklearn.metrics` | 模型评估 |
| `sklearn.linear_model` | 线性模型，如线性回归、逻辑回归 |
| `sklearn.tree` | 决策树、随机森林等 |
| `sklearn.svm` | 支持向量机 |
| `sklearn.cluster` | 聚类算法，如 KMeans |
| `sklearn.decomposition` | PCA 等降维算法 |

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
