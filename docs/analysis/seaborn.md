# Seaborn 教程

## 一、简介

Seaborn 是基于 Matplotlib 构建的高级可视化库，专为 统计图表 而设计。相比 Matplotlib，它更美观、更易用，尤其适合数据分析工作流。

+ 官网：https://seaborn.pydata.org
+ 核心优势：更美观的默认样式、对 Pandas DataFrame 友好、支持分类数据与统计图

##  二、安装

```bash
pip install seaborn
```

## 三、常用图

| 图表类型 | 函数 | 功能 |
|----------|------|------|
| 直方图 | `histplot()` | 显示数值分布频率 |
| 密度图 | `kdeplot()` | 显示数值的概率密度 |
| 散点图 | `scatterplot()` | 显示两变量之间关系 |
| 折线图 | `lineplot()` | 显示趋势或时间序列 |
| 回归图 | `regplot()` | 带回归线的散点图 |
| 箱型图 | `boxplot()` | 显示中位数、四分位数和异常值 |
| 小提琴图 | `violinplot()` | 显示分布密度与箱型图合并 |
| 条形图 | `barplot()` | 类别变量对应数值的均值等统计量 |
| 计数图 | `countplot()` | 类别变量的频数统计 |
| 热力图 | `heatmap()` | 显示变量之间相关性 |
| 矩阵图 | `pairplot()` | 显示成对变量关系 |


### 散点图

| 参数名     | 说明 |
|------------|------|
| `x`        | 横坐标变量名 |
| `y`        | 纵坐标变量名 |
| `hue`      | 控制点的颜色分类变量（如性别、时间） |
| `style`    | 控制点的形状分类变量（如吸烟与否） |
| `size`     | 控制点的大小变量（如数量、等级） |
| `data`     | 数据源，通常是一个 DataFrame |
| `palette`  | 调色板（如 `"pastel"`、`"dark"`、`"Set2"` 等） |
| `s`        | 点的大小（标量值，默认大小 40） |
| `marker`   | 点的形状（如 `"o"` 表示圆形，`"s"` 表示方形） |
| `alpha`    | 点的透明度（范围 0~1） |
| `legend`   | 是否显示图例（默认 `"auto"`，可设为 `False`） |
| `edgecolor`| 点的边框颜色（如 `"w"` 表示白色） |

```python
import seaborn as sns
import matplotlib.pyplot as plt 

# 加载数据集
tips = sns.load_dataset("tips")
# hue:根据是否抽烟设置不同颜色 sex: 根据性别设置不同形状 day:根据这列控制大小
sns.scatterplot(data=tips,x="total_bill",y="tip",hue="smoker",style="sex",size="day")
plt.title("总消费与小费之间的关系")
plt.show()
```

![散点图.png](../imgs/seaborn/%E6%95%A3%E7%82%B9%E5%9B%BE.png)

###  条形图

| 参数名        | 说明 |
|---------------|------|
| `x`           | 分类变量（横轴） |
| `y`           | 数值变量（纵轴） |
| `hue`         | 分组变量，显示多组条形 |
| `data`        | 数据集，通常是 Pandas 的 DataFrame |
| `palette`     | 设置颜色方案（如 `"Set2"`、`"pastel"`） |
| `ci`          | 误差棒（默认是 95% 置信区间，设为 `None` 可取消） |
| `estimator`   | 聚合函数，默认是 `np.mean`，也可设为 `np.sum`、`len` 等 |
| `order`       | 分类变量显示顺序（列表） |
| `hue_order`   | hue 的分类顺序（列表） |
| `orient`      | 条形图方向：`"v"`（默认）竖直方向 或 `"h"` 横向 |

