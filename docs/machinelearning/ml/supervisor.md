# 第三章：监督学习模型

## 一、支持向量机（SVM）

### 1.1 什么是 SVM？

SVM（Support Vector Machine，支持向量机）是一种**监督学习的分类算法**。

它的目标是：
> 找到一条最佳的“分界线”（也称为**超平面**），把不同类别的数据分开，并尽可能保持距离最大（安全间隔最大）。


### 1.2 为什么要使用 SVM？

SVM 特别适合以下场景：

+ 样本数量不多，但特征很多（高维数据）
+ 分类任务要求高准确率
+ 想要找到“鲁棒性强”的分类边界

### 1.3  SVM 的工作原理

#### 线性可分情况：

SVM 找一条“最佳直线”（或超平面），把两类点（例如猫 vs 狗）完全分开，同时：

✅ 保证两边的点离分界线**尽可能远**

✅ 中间的“安全带”（间隔）**越宽越好**

![svm.png](../../imgs/ml/svm.png)

#### 数学形式

假设我们要找的超平面是：

```text
w·x + b = 0
```
其中：

+ w 是法向量，决定方向
+ b 是偏置，决定距离原点多远

分类规则：

+ 正类：w·x + b ≥ +1
+ 负类：w·x + b ≤ -1

目标是：**最大化间隔 = 2 / ||w||**

### 1.4 硬间隔 vs 软间隔

#### 硬间隔（Hard Margin）：

+ 要求数据完全可分
+ 不允许有任何分类错误
+ 适合干净数据，但现实中很少

#### 软间隔（Soft Margin）：

+ 允许部分错误分类（引入松弛变量 ξ）
+ 平衡“间隔最大”和“分类错误最小”
+ 更适合真实数据

### 1.5 核函数（处理非线性问题）

有些数据用直线根本分不开怎么办？

!!! info

    👉 答案是：用核函数把数据“映射”到高维空间，在高维空间里分开！

### 1.6 Python 实现 SVM 分类（线性核）

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 生成简单数据
X, y = datasets.make_classification(n_samples=100, n_features=2, 
                                     n_redundant=0, n_clusters_per_class=1, random_state=42)

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型，使用线性核
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# 输出准确率
print("准确率:", clf.score(X_test, y_test))

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("支持向量机分类效果")
plt.show()
```

### 1.7 SVM 的优缺点

#### ✅ 优点

+ 分类准确率高，特别是高维数据
+ 对小样本有效
+ 可以使用核函数扩展到非线性分类
+ 理论基础强，泛化能力好

#### ❌ 缺点

+ 对参数敏感（如 C、核参数）
+ 不适合大数据集（训练慢）
+ 对噪声和重叠数据不太稳健

## 二、逻辑回归

