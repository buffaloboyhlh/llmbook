# 第三章：监督学习

## 一、线性模型

### 1.1 线性回归（Linear Regression）

线性回归是一种最基础但非常重要的机器学习算法，主要用于预测连续值。

#### 1. 什么是线性回归？

线性回归的目标是：
> 找到一条最佳的直线，拟合一组点的趋势，从而进行预测。

**比如：**

+ 根据房子的面积预测房价
+ 根据学生的学习时间预测考试成绩

这就是线性回归的经典应用。

#### 2. 一元线性回归公式

假设你有一个变量 x，我们想预测 y，那么模型如下：

$y = wx + b$

+ w：斜率（weight）
+ b：截距（bias）
+ 这就是我们要学的参数

####  3. 目标是什么？

我们希望找到最合适的 w 和 b，使得：

+ 预测值 $\hat{y}$ 尽可能接近真实值 y
+ 损失函数（误差）最小

常用的损失函数是 **均方误差（MSE）**：

$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$

####  4. 模型训练（梯度下降）

我们可以使用 **梯度下降（Gradient Descent）** 来不断调整 w 和 b，直到 MSE 最小。

#### 5. Python 实战演示（sklearn）

```python
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 模拟数据：房屋面积 vs 房价
X = np.array([[50], [60], [80], [100], [120]])  # 面积
y = np.array([150, 180, 240, 280, 310])        # 房价（单位：万）

# 创建模型并训练
model = LinearRegression()
model.fit(X, y)

# 预测
X_test = np.array([[90]])
y_pred = model.predict(X_test)
print("预测 90 平米房子的价格：", y_pred[0], "万")

# 可视化
plt.scatter(X, y, color='blue', label='真实数据')
plt.plot(X, model.predict(X), color='red', label='拟合线')
plt.xlabel('面积（平方米）')
plt.ylabel('价格（万）')
plt.title('线性回归：面积 vs 价格')
plt.legend()
plt.show()
```

####  6. 多元线性回归

当我们有多个变量（例如面积、卧室数量、楼层等），模型变成：

$y = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$

Python 训练方式完全相同，只需提供多个特征即可。

####  7. 线性回归的优缺点

##### 优点：

+ 简单直观，易于理解
+ 训练速度快
+ 可解释性强（每个变量的权重可解释）

##### 缺点：

+ 只能处理线性关系
+ 对异常值敏感
+ 不能处理多峰或非线性数据

#### 8. 常见变种

| 模型                         | 特点                                   |
|------------------------------|----------------------------------------|
| 岭回归（Ridge）              | 加入 L2 正则化，防止过拟合             |
| Lasso 回归                   | 加入 L1 正则化，可用于特征选择         |
| 多项式回归                   | 可拟合曲线（非线性）                   |
| 逻辑回归（Logistic Regression） | 虽叫“回归”，其实是分类算法             |


----

### 1.2 逻辑回归（Logistic Regression）

#### 1. 什么是逻辑回归？

逻辑回归是一种**用于二分类问题的监督学习算法**，尽管名字中有“回归”，但它其实是用于**分类任务**的。

- 输入：一组特征  
- 输出：某个类别（例如：0 或 1）  
- 目标：学习一个函数，输入特征后预测样本属于类别 1 的概率  

#### 2. 应用场景

| 应用领域     | 示例                  |
|--------------|-----------------------|
| 医疗诊断     | 判断肿瘤是良性或恶性 |
| 金融风控     | 判断是否可能违约     |
| 市场营销     | 判断用户是否会点击广告 |
| 社会调查分析 | 判断用户是否支持某观点 |

#### 3. 模型公式与原理

##### ✅ 线性部分

逻辑回归首先使用一个线性模型：

$$
z = w^\top x + b
$$

其中：

- \( x \)：特征向量  
- \( w \)：权重向量  
- \( b \)：偏置项  

##### ✅ Sigmoid 函数

为了把 \( z \) 映射到 [0, 1] 范围，我们使用 **Sigmoid** 函数：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

输出就是属于类别 1 的概率 \( P(y=1|x) \)。

##### ✅ 分类判定规则

$$
\hat{y} = \begin{cases}
1, & \text{if } \sigma(z) \geq 0.5 \\
0, & \text{otherwise}
\end{cases}
$$

#### 4. 损失函数（对数损失）

逻辑回归使用 **对数损失函数（Log Loss）**：

$$
L(y, \hat{y}) = -y \log(\hat{y}) - (1 - y) \log(1 - \hat{y})
$$

目标是**最小化所有样本的平均损失**。

#### 5. 模型训练过程

1. 初始化权重 \( w \) 和偏置 \( b \)
2. 对每一轮训练（epoch）：
   - 计算预测值 \( \hat{y} \)
   - 计算损失函数
   - 使用梯度下降更新参数：
     $$
     w := w - \eta \cdot \nabla_w L, \quad b := b - \eta \cdot \nabla_b L
     $$
   - 直到损失收敛或达到迭代上限

#### 6. Python 实现（使用 scikit-learn）

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据：使用 Iris 数据集中两个类别
iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 模型训练
model = LogisticRegression()
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 评估准确率
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
```

#### 7.  优点与缺点

| 优点                         | 缺点                     |
|------------------------------|--------------------------|
| 简单高效，适用于线性问题     | 对非线性问题表现较差     |
| 概率输出，模型可解释性强     | 容易欠拟合               |
| 训练速度快，适合大样本       | 对异常值敏感             |

----

### 1.3 岭回归（Ridge Regression）和 Lasso回归

#### 1. 为什么需要正则化？

在使用普通线性回归时，可能会遇到以下问题：

+ 特征数量多（高维）
+ 特征之间存在共线性（Multicollinearity）
+ 模型容易过拟合（Overfitting）

为了解决这些问题，我们引入了**正则化方法**：

+ Ridge Regression（岭回归）：L2 正则化
+ Lasso Regression：L1 正则化

#### 2.  岭回归（Ridge Regression）

在普通最小二乘的损失函数基础上，加上了 L2 正则项：

$$
\text{Loss}{\text{Ridge}} = \sum{i=1}^n (y_i - \hat{y}i)^2 + \lambda \sum{j=1}^p w_j^2
$$

其中：

+ $\lambda$：正则化强度（超参数）
+ $w_j$：回归系数

**特点**

+ 缩小系数，但不会使其变为 0
+ 更适用于所有特征都有贡献的情况

#### 3. Lasso 回归（Lasso Regression）

Lasso 在损失函数中加入了 L1 正则项：

$$
\text{Loss}{\text{Lasso}} = \sum{i=1}^n (y_i - \hat{y}i)^2 + \lambda \sum{j=1}^p |w_j|
$$

**特点**

+ 可以让某些系数变为 0（变量选择）
+ 更适用于特征多、但部分特征不重要的场景

#### 4.  Ridge vs Lasso 对比

| 对比点       | Ridge 回归             | Lasso 回归              |
|--------------|-------------------------|--------------------------|
| 正则化类型   | L2 正则化               | L1 正则化                |
| 是否稀疏     | 否                      | 是（可进行特征选择）     |
| 系数为 0     | 不可能                  | 可能                     |
| 应用场景     | 所有特征都相关          | 部分特征相关             |

#### 5. Python 实现（scikit-learn）

###### 📌 岭回归示例

```python
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 加载数据
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 建立模型
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# 预测与评估
print("Ridge score:", ridge.score(X_test, y_test))
```

###### 📌 Lasso 回归示例

```python
from sklearn.linear_model import Lasso

# 建立模型
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# 预测与评估
print("Lasso score:", lasso.score(X_test, y_test))
```

#### 6. 正则化参数 $\alpha$ 的选择

+ $\alpha$ 越大 → 正则化越强 → 模型更简单，可能欠拟合
+ $\alpha$ 越小 → 正则化越弱 → 模型更复杂，可能过拟合

可以使用 GridSearchCV 或 RidgeCV、LassoCV 自动选择最优参数。


## 二、决策树与集成方法

### 2.1 决策树（Decision Tree）

### 2.2 随机森林（Random Forest）

### 2.3 梯度提升树（Gradient Boosting）

## 三、支持向量机（SVM）

### 3.1 线性SVM

### 3.2 核方法（Kernel SVM）


## 四、贝叶斯方法

### 4.1 朴素贝叶斯（Naive Bayes）

### 4.2 贝叶斯网络

## 五、 最近邻算法

### 5.1 K近邻（K-Nearest Neighbors, KNN）

## 六、神经网络与深度学习

### 6.1 多层感知机（MLP）

### 6.2 卷积神经网络（CNN）

### 6.3 循环神经网络（RNN/LSTM）

