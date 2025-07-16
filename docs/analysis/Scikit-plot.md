# Scikit-plot 教程 

## 一、Scikit-plot 是什么？

scikit-plot 是一个专为 scikit-learn 设计的模型可视化库，它封装了许多常用可视化图，比如混淆矩阵、ROC 曲线、学习曲线、特征重要性等，让你能非常简单地进行模型评估。

+ GitHub：https://github.com/reiinakano/scikit-plot
+ 优点：开箱即用、对 sklearn 模型兼容性极高。

##  二、安装

```bash
pip install scikit-plot
```
依赖：matplotlib、scikit-learn、numpy


##  三、快速入门示例

```python
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 数据加载与模型训练
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 预测并可视化混淆矩阵
y_pred = clf.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_test, y_pred)
plt.show()
```

##  四、常用可视化功能汇总

| 图类型         | 函数名                                 | 说明                             |
|----------------|------------------------------------------|----------------------------------|
| 混淆矩阵       | `skplt.metrics.plot_confusion_matrix()` | 分类模型表现                     |
| ROC 曲线       | `skplt.metrics.plot_roc()`              | 多类/二类分类器的 ROC 曲线      |
| 精确率-召回曲线 | `skplt.metrics.plot_precision_recall()` | 精确率 vs 召回率曲线            |
| 学习曲线       | `skplt.estimators.plot_learning_curve()`| 模型训练与验证误差对比           |
| 验证曲线       | `skplt.estimators.plot_validation_curve()` | 超参数变化对模型性能影响     |
| 特征重要性     | `skplt.estimators.plot_feature_importances()` | 特征对模型的重要性排序    |
| KMeans 可视化  | `skplt.cluster.plot_elbow_curve()`      | 用于选择最佳聚类数 K             |
| Silhouette 图  | `skplt.cluster.plot_silhouette()`       | 聚类簇分布紧密度与分离度评估     |
| Calibration 图 | `skplt.metrics.plot_calibration_curve()`| 预测概率的校准情况（如过拟合）   |
| 决策边界       | `skplt.estimators.plot_2d_classification()` | 二维分类模型的决策边界可视化 |


##  五、分类模型可视化示例

#### 1️⃣ 混淆矩阵

```python
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
plt.show()
```

#### 2️⃣ ROC 曲线

```python
y_probas = clf.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, y_probas)
plt.show()
```

#### 3️⃣ 精确率-召回率曲线

```python
skplt.metrics.plot_precision_recall(y_test, y_probas)
plt.show()
```

## 六、学习曲线

```python
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

skplt.estimators.plot_learning_curve(SVC(), X, y, cv=StratifiedKFold(5), scoring='accuracy')
plt.show()
```

## 七、特征重要性可视化

```python
skplt.estimators.plot_feature_importances(clf, feature_names=None, title="特征重要性")
plt.show()
```
注意：模型必须具有 .feature_importances_ 属性，如随机森林、决策树、XGBoost 等。

## 八、聚类算法可视化

#### 1️⃣ Elbow 曲线：选择最佳聚类数 K

```python
from sklearn.cluster import KMeans

skplt.cluster.plot_elbow_curve(KMeans(), X)
plt.show()
```

#### 2️⃣ Silhouette 分数图

```python
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
skplt.cluster.plot_silhouette(X, kmeans.labels_)
plt.show()
```

## 九、模型校准曲线（概率输出校准）

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

gnb = GaussianNB()
calibrated = CalibratedClassifierCV(gnb, cv=3)
calibrated.fit(X_train, y_train)

skplt.metrics.plot_calibration_curve(y_test, [gnb.fit(X_train, y_train), calibrated], model_names=['Naive Bayes', 'Calibrated NB'])
plt.show()
```


