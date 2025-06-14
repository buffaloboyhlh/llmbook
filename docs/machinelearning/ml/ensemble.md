# 第十二章：集成学习

---

## 一、概念简介

**集成学习**是一种将多个模型（通常是弱模型）组合成一个强模型的方法，主要目标是提高整体的泛化能力和预测准确率。

常见术语：

| 术语       | 含义                             |
|------------|----------------------------------|
| 弱学习器   | 表现略优于随机猜测的模型（如决策树） |
| 强学习器   | 精度较高的模型，是多个弱学习器组合后的结果 |
| 基学习器   | 被集成的基础模型                   |
| 集成方法   | 组合方式，如投票、加权平均等         |

---

## 二、集成学习三大主要方法

### 1. Bagging（Bootstrap Aggregating）

- **思想**：多个模型并行训练，平均或投票结果。
- **目标**：降低方差（Variance），防止过拟合。
- **代表算法**：Random Forest

#### 公式

假设有 T 个模型 $begin:math:text$ h_1(x), h_2(x), \\dots, h_T(x) $end:math:text$，最终输出：

- 分类：  
  $begin:math:display$
  H(x) = \\text{majority\\_vote}(h_1(x), ..., h_T(x))
  $end:math:display$
- 回归：  
  $begin:math:display$
  H(x) = \\frac{1}{T} \\sum_{t=1}^T h_t(x)
  $end:math:display$

#### 示例代码

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))
```

---

### 2. Boosting

- **思想**：多个模型串行训练，后一个模型专注于修正前一个模型的错误。
- **目标**：降低偏差（Bias），提高准确率。
- **代表算法**：AdaBoost、Gradient Boosting、XGBoost、LightGBM、CatBoost

#### 公式（以 AdaBoost 为例）

给定训练样本 $begin:math:text$ (x_i, y_i) $end:math:text$，每轮训练一个分类器 $begin:math:text$ h_t(x) $end:math:text$，并分配权重 $begin:math:text$ \\alpha_t $end:math:text$，最终模型为：

$begin:math:display$
H(x) = \\text{sign}\\left(\\sum_{t=1}^{T} \\alpha_t h_t(x)\\right)
$end:math:display$

#### 示例代码（使用 AdaBoost）

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50
)
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))
```

---

### 3. Stacking（Stacked Generalization）

- **思想**：将多个不同类型的模型作为一层，输出作为下一层模型的输入。
- **目标**：综合多种模型优点，提升性能。
- **特点**：模型间可异构，通常分两层或多层。

#### 示例代码（使用 sklearn Stacking）

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

estimators = [
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC(probability=True))
]
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)
stack_model.fit(X_train, y_train)
print("Accuracy:", stack_model.score(X_test, y_test))
```

---

## 三、方法对比总结

| 方法     | 是否并行 | 侧重方向 | 是否容易过拟合 | 可扩展性 |
|----------|----------|----------|----------------|----------|
| Bagging  | ✅       | 减少方差 | 较低           | 中等     |
| Boosting | ❌       | 减少偏差 | 容易过拟合     | 高       |
| Stacking | 可并可串 | 综合提升 | 中等           | 高       |

---

## 四、进阶：主流 Boosting 框架介绍

| 框架     | 特点                           |
|----------|--------------------------------|
| XGBoost  | 正则化强，支持稀疏数据         |
| LightGBM | 速度快，适合大数据              |
| CatBoost | 原生支持类别特征，调参简单     |

#### 示例代码（使用 XGBoost）

```python
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
```

---

## 五、集成学习的实用建议

- 弱模型可选：决策树、KNN、朴素贝叶斯、线性模型等
- 模型差异性越大，集成效果越好
- 可用于不平衡数据、异常检测、时间序列预测等多种任务
- 可结合交叉验证、模型融合进一步优化

---

## 六、学习路线建议

1. **入门阶段**
   - 掌握 Bagging、Boosting 的基本思想与区别
   - 熟练使用 sklearn 的集成模型（如 RandomForest、AdaBoost）

2. **进阶阶段**
   - 理解 GBDT 的损失函数优化原理
   - 学习 Stacking 的多层模型构建与融合技巧

3. **实战应用**
   - 使用 XGBoost/LightGBM 调参、交叉验证
   - 多模型融合提升竞赛成绩（如 Kaggle）

---

如需配套 Jupyter Notebook 文件、可视化集成效果图，或使用集成学习做中文文本分类、图像识别等任务，请告诉我，我可以帮你定制。