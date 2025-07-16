# Seaborn 教程

## 一、简介

Seaborn 是基于 Matplotlib 构建的高级可视化库，专为 统计图表 而设计。相比 Matplotlib，它更美观、更易用，尤其适合数据分析工作流。

+ 官网：https://seaborn.pydata.org
+ 核心优势：更美观的默认样式、对 Pandas DataFrame 友好、支持分类数据与统计图

##  二、安装

```bash
pip install seaborn
```

## 三、快速使用

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载内置数据集
tips = sns.load_dataset("tips")

# 可视化
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.title("账单与小费关系")
plt.show()
```

## 四、常用图表类型

| 图表类型       | 函数名                | 描述                          |
|----------------|-----------------------|-------------------------------|
| 散点图         | `sns.scatterplot()`   | 两个数值型变量的关系         |
| 折线图         | `sns.lineplot()`      | 趋势线                       |
| 柱状图         | `sns.barplot()`       | 类别 → 数值的平均             |
| 条形图         | `sns.countplot()`     | 类别 → 频数                  |
| 箱线图         | `sns.boxplot()`       | 展示中位数、四分位数、异常值 |
| 小提琴图       | `sns.violinplot()`    | 类似箱线图但带 KDE 曲线       |
| 分布图         | `sns.histplot()`      | 直方图                       |
| KDE 密度图     | `sns.kdeplot()`       | 分布的核密度估计             |
| 热力图         | `sns.heatmap()`       | 数据矩阵热度图               |
| 配对图         | `sns.pairplot()`      | 所有变量两两可视化           |
| 分面图（子图） | `sns.FacetGrid()`     | 多子图分组展示               |
| 条件回归图     | `sns.lmplot()`        | 散点 + 回归线                |


## 五、内置数据集与风格设置

```python
# 加载数据集
df = sns.load_dataset("iris")  # 还有 tips、flights、titanic 等

# 设置风格
sns.set_style("whitegrid")  # white, dark, ticks
sns.set_palette("pastel")   # 可选 deep, bright, dark, colorblind
```

## 六、分类图表：分类变量 VS 数值变量

```python
# 箱线图
sns.boxplot(x="day", y="total_bill", data=tips)

# 小提琴图
sns.violinplot(x="day", y="total_bill", data=tips)

# 条形图（平均值）
sns.barplot(x="day", y="total_bill", data=tips)

# 计数图（频数）
sns.countplot(x="day", data=tips)
```

## 七、数值变量分布分析

```python
# 单变量分布
sns.histplot(tips["total_bill"], kde=True)

# 核密度估计（KDE）
sns.kdeplot(data=tips["tip"], shade=True)
```

## 八、变量关系探索

```python
# 散点图
sns.scatterplot(x="total_bill", y="tip", hue="sex", data=tips)

# 带回归线的图（线性模型）
sns.lmplot(x="total_bill", y="tip", hue="sex", data=tips)

# 多变量分布（成对变量）
sns.pairplot(tips, hue="sex")
```

##  九、热力图与透视表

```python
# 创建透视表
pivot = tips.pivot_table(values="tip", index="day", columns="time", aggfunc="mean")

# 热力图
sns.heatmap(pivot, annot=True, cmap="YlGnBu")
```

##  十、子图布局（FacetGrid）

```python
g = sns.FacetGrid(tips, col="sex", row="time")
g.map(sns.histplot, "total_bill")
```

## 十一、自定义美化

```python
sns.set_context("notebook")  # 其他：paper, talk, poster
sns.set_style("darkgrid")

# 自定义调色板
sns.set_palette("Set2")
```

##  十二、保存图像

```python
plt.savefig("seaborn_plot.png", dpi=300, bbox_inches="tight")
```

