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


## 十三、实战

#### 1. 散点图

| 参数         | 说明                                         |
|--------------|----------------------------------------------|
| `x`, `y`     | 指定要绘制的变量名                           |
| `data`       | pandas DataFrame 数据源                      |
| `hue`        | 根据类别变量设置点的颜色（自动生成图例）     |
| `style`      | 设置点的样式（如圆形、三角等）               |
| `size`       | 设置点的大小                                 |
| `palette`    | 设置颜色调色板（如 `"pastel"`、`"deep"`）     |
| `alpha`      | 设置透明度（0~1）                             |
| `s`          | 设置所有点的固定大小（如果不使用 `size`）     |
| `marker`     | 设置统一点的形状，如 `"o"`、`"s"`、`"^"` 等   |

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['STHeiTi']

# 加载示例数据集
tips = sns.load_dataset("tips")
# 绘制散点图：总账单 vs 小费
sns.scatterplot(x='total_bill',y='tip',data=tips,hue="sex") # hue 更具性别设置不同颜色
plt.title('散点图')
plt.show()
```
![散点图.png](../imgs/seaborn/%E6%95%A3%E7%82%B9%E5%9B%BE.png)

**带 style样式**

style="smoker" 用 smoker 列的值区分点的形状，增强分类信息

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['STHeiTi']

# 加载示例数据集
tips = sns.load_dataset("tips")
# 绘制散点图：总账单 vs 小费
sns.scatterplot(x='total_bill',y='tip',data=tips,hue="sex",style="smoker") # hue 更具性别设置不同颜色
plt.title('散点图')
plt.show()
```
![带样式散点图.png](../imgs/seaborn/%E5%B8%A6%E6%A0%B7%E5%BC%8F%E6%95%A3%E7%82%B9%E5%9B%BE.png)


#### 2.折线图 

| 参数       | 说明                                            |
|------------|-------------------------------------------------|
| `x`, `y`   | 指定横轴和纵轴的数据列名                        |
| `data`     | DataFrame 数据源                                 |
| `hue`      | 根据分类变量分组，画多条线，每条线不同颜色       |
| `style`    | 根据分类变量设置线条样式（如虚线、点线等）       |
| `size`     | 控制线的粗细（可根据变量变化）                   |
| `markers`  | 是否显示点标记，或传入具体的样式（如 `True` 或 `['o','s']`） |
| `dashes`   | 设置虚线样式（如 `[True, False]`）               |
| `palette`  | 设置调色板，如 `"pastel"`、`"deep"`              |

```python
sns.lineplot(x="size", y="tip", hue="sex", style="smoker", data=tips, markers=True)
```

+ hue="sex"：男女使用不同颜色
+ style="smoker"：吸烟与否使用不同线型
+ markers=True：在数据点位置加上点标记

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['STHeiTi']

# 示例数据
tips = sns.load_dataset("tips")

# 折线图：x=人数，y=小费
sns.lineplot(x="size", y="tip", data=tips)

plt.title("线性趋势：就餐人数 vs 小费")
plt.show()
```

![折线图.png](../imgs/seaborn/%E6%8A%98%E7%BA%BF%E5%9B%BE.png)

**误差线（error bar）或误差带（误差阴影区域）**

| 用法                  | 效果说明                                       |
|-----------------------|------------------------------------------------|
| `errorbar=None`       | 不显示误差线，只画折线                         |
| `errorbar='sd'`       | 显示标准差的误差带                             |
| `errorbar=('ci', 95)` | 显示 95% 的置信区间（默认行为）                |
| `errorbar=('pi', 90)` | 显示 90% 的预测区间（Seaborn 0.12+ 支持）      |
| `errorbar=('se', 1)`  | 显示 1 倍标准误差                              |

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['STHeiTi']

# 示例数据
tips = sns.load_dataset("tips")

# 折线图：x=人数，y=小费
sns.lineplot(x="size", y="tip", data=tips,errorbar=None)

plt.title("线性趋势：就餐人数 vs 小费")
plt.show()
```
![折线图带error.png](../imgs/seaborn/%E6%8A%98%E7%BA%BF%E5%9B%BE%E5%B8%A6error.png)

#### 3. 柱状图




