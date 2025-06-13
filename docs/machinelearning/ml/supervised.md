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

### 1.2 逻辑回归（Logistic Regression）

### 1.3 岭回归（Ridge Regression）和 Lasso回归

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

