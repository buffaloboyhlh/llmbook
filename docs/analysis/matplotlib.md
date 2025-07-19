# Matplotlib 教程

## 一、简介

Matplotlib 是 Python 中最常用的数据可视化库，提供类似 MATLAB 的绘图 API，可以绘制各种静态、动态、交
互式图表。

+ 官网：https://matplotlib.org
+ 核心模块：matplotlib.pyplot（通常简写为 plt）

## 二、安装

```shell
pip install matplotlib
```

## 三、快速入门

```python
import matplotlib.pyplot as plt

# 数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 绘图
plt.plot(x, y)
plt.title("简单折线图")
plt.xlabel("x轴")
plt.ylabel("y轴")
plt.show()
```

![简单折线图.png](../imgs/analysis/matplotlib/%E7%AE%80%E5%8D%95%E6%8A%98%E7%BA%BF%E5%9B%BE.png)

## 四、常见图表类型

| 类型         | 函数                 | 描述                  |
|--------------|----------------------|-----------------------|
| 折线图       | `plt.plot()`         | 折线                  |
| 散点图       | `plt.scatter()`      | 点状分布              |
| 条形图       | `plt.bar()`          | 柱状（竖）            |
| 条形图（横） | `plt.barh()`         | 柱状（横）            |
| 直方图       | `plt.hist()`         | 数据分布              |
| 饼图         | `plt.pie()`          | 占比                  |
| 箱线图       | `plt.boxplot()`      | 四分位/异常值         |
| 填充图       | `plt.fill_between()` | 区域阴影图            |
| 图像显示     | `plt.imshow()`       | 显示图片或热力图      |

## 五、图表元素定制

#### 1. 添加标题和坐标轴标签

```python
plt.title("标题")
plt.xlabel("x标签")
plt.ylabel("y标签")
```

#### 2. 图例 legend

```python
plt.plot(x, y, label="线1")
plt.legend(loc="best")  # 可选: upper right, lower left 等
```

#### 3. 坐标范围

```python
plt.xlim(0, 10)
plt.ylim(0, 20)
```

#### 4. 样式设置（颜色/线型/标记）

```python
plt.plot(x, y, color='r', linestyle='--', marker='o')  # 红色虚线圆点
```

## 六、样式与主题

```python
plt.style.use("ggplot")   # 其他样式如：seaborn, fivethirtyeight, classic
```

查看所有可用样式：

```python
print(plt.style.available)
```

##  七、子图 subplot

```python
plt.subplot(2, 1, 1)  # 2行1列，第1个图
plt.plot([1, 2, 3], [1, 2, 3])

plt.subplot(2, 1, 2)  # 第2个图
plt.plot([1, 2, 3], [3, 2, 1])
plt.tight_layout()
plt.show()
```

## 八、保存图像

```python
plt.savefig("figure.png", dpi=300, bbox_inches="tight")
```

## 九、结合 NumPy 使用示例

```python
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("正弦曲线")
plt.grid(True)
plt.show()
```

## 十、动画与交互（进阶）

```python
from matplotlib import animation

fig, ax = plt.subplots()
x, y = [], []
line, = ax.plot([], [], 'r-')

def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return line,

def update(frame):
    x.append(frame)
    y.append(frame)
    line.set_data(x, y)
    return line,

ani = animation.FuncAnimation(fig, update, frames=range(10), init_func=init, blit=True)
plt.show()
```

## 十一、常见问题

| 问题                          | 解决方法                                                   |
|-------------------------------|------------------------------------------------------------|
| 中文乱码                      | `plt.rcParams['font.sans-serif'] = ['SimHei']`             |
| 负号显示为方框                | `plt.rcParams['axes.unicode_minus'] = False`               |
| 图像显示不完整                | `plt.tight_layout()`                                       |
| 图表显示但一闪而过（在某些 IDE 中） | `plt.show()` 放在最后显示图表                              |
| Jupyter 中图表不显示         | 在开头添加 `%matplotlib inline`                           |
| 保存图像模糊或被裁剪         | 使用 `dpi=300` 和 `bbox_inches='tight'` 参数配合 `savefig` |


## 十二、实战

####  1. 折线图（Line Chart）

```python
import matplotlib.pyplot as plt 
%matplotlib inline

plt.rcParams['font.family'] = ["STHeiTi"]

x = [1, 2, 3, 4]
y = [10, 20, 15, 25]

plt.plot(x,y,color="red",linestyle='--',marker='o',label="线-1")
plt.title('简单折线图')
plt.xlabel('x 轴')
plt.ylabel('y 轴')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```

![折线图.png](../imgs/analysis/matplotlib/%E6%8A%98%E7%BA%BF%E5%9B%BE.png)


#### 2. 散点图（Scatter Plot）

```python
import matplotlib.pyplot as plt 

plt.rcParams['font.family'] = ['STHeiTi']
plt.rcParams['axes.unicode_minus'] = False # 减号显示问题

x = [1, 2, 3, 4]
y = [10, 12, 14, 13]

plt.scatter(x=x,y=y,color="blue",label='散点')
plt.title("散点图",color="red",loc="left")
plt.legend()
plt.show()
```

![散点图.png](../imgs/analysis/matplotlib/%E6%95%A3%E7%82%B9%E5%9B%BE.png)

#### 3. 柱状图（Bar Chart）

```python
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['STHeiTi']

# 数据
labels = ['A', 'B', 'C', 'D']
x = np.arange(len(labels))  # x坐标
y1 = [5, 7, 6, 8]
y2 = [3, 4, 2, 5]
y3 = [2, 1, 4, 3]

# 绘图
plt.bar(x, y1, label='类别1')
plt.bar(x, y2, bottom=y1, label='类别2')  # y2 在 y1 上面
plt.bar(x, y3, bottom=np.array(y1)+np.array(y2), label='类别3')  # y3 累加
# 添加标签
plt.xticks(x, labels)
plt.ylabel('数值')
plt.title('叠加柱状图')
plt.legend()

plt.tight_layout()
plt.show()
```

![叠加柱状图.png](../imgs/analysis/matplotlib/%E5%8F%A0%E5%8A%A0%E6%9F%B1%E7%8A%B6%E5%9B%BE.png)

```python
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['STHeiTi']

# 组标签
labels = ['A', 'B', 'C', 'D']
x = np.arange(len(labels))  # x轴位置

# 每个类别的数据
y1 = [5, 6, 7, 8]   # 类别1
y2 = [4, 5, 6, 7]   # 类别2
y3 = [3, 4, 5, 6]   # 类别3

bar_width = 0.2  # 每个柱的宽度

plt.bar(x-bar_width,y1,width=bar_width,label="类别 1")
plt.bar(x,y2,width=bar_width,label='类别 2')
plt.bar(x+bar_width,y3,width=bar_width,label='类别 3')

# 标签设置
plt.xticks(x,labels)
plt.ylabel("数值")
plt.title("分组柱状图")
plt.legend()
plt.tight_layout()
plt.show()
```

![分组柱状图.png](../imgs/analysis/matplotlib/%E5%88%86%E7%BB%84%E6%9F%B1%E7%8A%B6%E5%9B%BE.png)

```python
import matplotlib.pyplot as plt
import numpy as np

# 分组标签
labels = ['A', 'B', 'C', 'D']
y = np.arange(len(labels))  # y轴坐标 [0,1,2,3]

# 每组的子类别数据
x1 = [5, 6, 7, 8]
x2 = [4, 5, 6, 7]
x3 = [3, 4, 5, 6]

bar_height = 0.2  # 每个子类别的条形高度

# 绘图：y轴坐标向下偏移，避免重叠
plt.barh(y - bar_height, x1, height=bar_height, label='类别1')
plt.barh(y,              x2, height=bar_height, label='类别2')
plt.barh(y + bar_height, x3, height=bar_height, label='类别3')

# 设置y轴标签
plt.yticks(y, labels)
plt.xlabel('数值')
plt.title('水平分组柱状图')
plt.legend()
plt.tight_layout()
plt.show()
```

![水平分组柱状图.png](../imgs/analysis/matplotlib/%E6%B0%B4%E5%B9%B3%E5%88%86%E7%BB%84%E6%9F%B1%E7%8A%B6%E5%9B%BE.png)


#### 4. 直方图

```python
import matplotlib.pyplot as plt 
import numpy as np

plt.rcParams['font.family'] = ['STHeiTi']

# 生成模拟数据（正态分布）
data = np.random.normal(loc=100, scale=15, size=200)

# density：是否标准化为概率密度   bins：分成多少个区间（也可以传 list 自定义）
plt.hist(data,bins=10,density=True,color='skyblue',edgecolor='black')
plt.xlabel('数值区间')
plt.ylabel("频数")
plt.title("直方图示例")
plt.tight_layout()
plt.show()
```

![直方图.png](../imgs/analysis/matplotlib/%E7%9B%B4%E6%96%B9%E5%9B%BE.png)


```python
from scipy.stats import norm

# 拟合正态分布曲线
mu, std = norm.fit(data)
x = np.linspace(min(data), max(data), 100)
pdf = norm.pdf(x, mu, std)

# 绘图
plt.hist(data, bins=10, density=True, color='skyblue', edgecolor='black')
plt.plot(x, pdf, 'r-', label='正态分布拟合')
plt.legend()
plt.show()
```

![正太直方图.png](../imgs/analysis/matplotlib/%E6%AD%A3%E5%A4%AA%E7%9B%B4%E6%96%B9%E5%9B%BE.png)

#### 5. 饼图

```python
import matplotlib.pyplot as plt 

# 数据
labels = ['苹果', '香蕉', '橙子', '葡萄']
sizes = [30, 25, 20, 25]

# autopct：显示百分比，如 %1.1f%% 表示保留1位小数  startangle：起始角度（默认从 x 轴正方向）
plt.pie(x=sizes,labels=labels,autopct='%1.1f%%',startangle=90)

plt.title('水果销量占比')
plt.axis('equal') # 保证为正圆形
plt.show()
```

![饼状图.png](../imgs/analysis/matplotlib/%E9%A5%BC%E7%8A%B6%E5%9B%BE.png)


```python
import matplotlib.pyplot as plt 

plt.rcParams['font.family'] = "STHeiTi"
plt.rcParams['axes.unicode_minus'] = False

labels = ['苹果', '香蕉', '橙子', '葡萄']
sizes = [30, 25, 20, 25]
explode = [0, 0.1, 0, 0]  # 突出香蕉

plt.pie(x=sizes,labels=labels,autopct="%1.1f%%",startangle=90,explode=explode,shadow=True)
plt.title("水果销量比例")
plt.axis('equal') # 设置 x 轴与 y 轴的单位长度比例相同
plt.tight_layout()
plt.show()
```

![突出饼状图.png](../imgs/analysis/matplotlib/%E7%AA%81%E5%87%BA%E9%A5%BC%E7%8A%B6%E5%9B%BE.png)


#### 6. 箱线图

| 组成部分        | 含义说明                                                                 |
|-----------------|--------------------------------------------------------------------------|
| 最小值（Minimum） | 非异常数据中的最小值，通常为 Q1 - 1.5×IQR 以内的数据最小值                |
| 下四分位数（Q1） | 将数据按大小顺序排列后，处于 25% 位置的值                                 |
| 中位数（Q2）     | 将数据按大小顺序排列后，处于 50% 位置的值（箱体中间的线）                   |
| 上四分位数（Q3） | 将数据按大小顺序排列后，处于 75% 位置的值                                 |
| 最大值（Maximum）| 非异常数据中的最大值，通常为 Q3 + 1.5×IQR 以内的数据最大值                |
| IQR             | 四分位数间距，IQR = Q3 - Q1                                              |
| 箱体（Box）     | 从 Q1 到 Q3 的区域，表示中间 50% 的数据分布                               |
| 中位线（Median Line） | 箱体内部的一条线，表示中位数的位置                                     |
| 须（Whiskers）  | 从 Q1 延伸到最小值、Q3 延伸到最大值的线，表示非异常值的范围                   |
| 异常值（Outliers）| 超出 Q1 - 1.5×IQR 或 Q3 + 1.5×IQR 范围的点，通常用单独的点来表示              |

![箱型图示例.png](../imgs/analysis/matplotlib/%E7%AE%B1%E5%9E%8B%E5%9B%BE%E7%A4%BA%E4%BE%8B.png)


```python
plt.boxplot(data, notch=False, vert=True, patch_artist=False, labels=None)
```


| 参数名         | 类型           | 默认值      | 说明 |
|----------------|----------------|-------------|------|
| `x`            | array-like     | 必须参数     | 输入数据，可以是一组数值、二维数组（多组）或带标签的结构。 |
| `notch`        | bool           | `False`     | 是否绘制缺口箱线图（notched boxplot）。缺口可用于粗略显示中位数差异是否显著。 |
| `vert`         | bool           | `True`      | 是否垂直显示箱线图。`False` 表示水平箱线图。 |
| `patch_artist` | bool           | `False`     | 是否填充箱体颜色。 |
| `meanline`     | bool           | `False`     | 是否用线条显示均值。 |
| `showmeans`    | bool           | `False`     | 是否显示均值点。 |
| `showbox`      | bool           | `True`      | 是否显示箱体。 |
| `showfliers`   | bool           | `True`      | 是否显示异常值（outliers）。 |
| `showcaps`     | bool           | `True`      | 是否显示上下须（whisker caps）。 |
| `whis`         | float or tuple | `1.5`       | 控制须的长度：`1.5` 表示1.5倍四分位距以外的点为异常值。 |
| `labels`       | list           | `None`      | 设置各组数据的标签。 |
| `flierprops`   | dict           | `None`      | 异常值的样式，如颜色、形状等，使用 `marker`、`color` 等字段设置。 |
| `boxprops`     | dict           | `None`      | 箱体样式设置，如颜色、线宽等。 |
| `capprops`     | dict           | `None`      | 须端横线（caps）的样式设置。 |
| `whiskerprops` | dict           | `None`      | 须（whiskers）的样式设置。 |
| `medianprops`  | dict           | `None`      | 中位数线条的样式设置。 |
| `meanprops`    | dict           | `None`      | 均值点或线的样式设置。 |
| `widths`       | float or array | `0.5`       | 箱体的宽度。 |
| `manage_ticks` | bool           | `True`      | 是否自动处理刻度标签。 |

> 🔍 `plt.boxplot()` 返回的是一个字典，包含绘图中各元素的 `Line2D` 或 `Patch` 对象，可用于自定义样式。

```python
import matplotlib.pyplot as plt
import numpy as np

# 模拟数据：3组成绩
data = [
    np.random.normal(80, 10, 100),
    np.random.normal(70, 15, 100),
    np.random.normal(60, 20, 100)
]

labels = ['语文', '数学', '英语']
plt.boxplot(data,labels=labels,patch_artist=True)
plt.title("学生成绩箱线图")
plt.ylabel("分数")
plt.grid(True)
plt.show()
```

![箱线图.png](../imgs/analysis/matplotlib/%E7%AE%B1%E7%BA%BF%E5%9B%BE.png)



#### 7. 填充图

**`plt.fill_between()` 参数详解**

| 参数名       | 说明                                                                 |
|--------------|----------------------------------------------------------------------|
| `x`          | 横坐标数组，表示填充区域的起始横坐标                                |
| `y1`         | 第一个 y 值数组，填充区域的上边界（或下边界）                        |
| `y2`         | （可选）第二个 y 值数组，填充区域的下边界，默认与 x 轴（y=0）填充     |
| `where`      | （可选）布尔数组，指示哪些区域需要填充                               |
| `interpolate`| 是否插值以确保在边界点填充，默认 False                               |
| `color`      | 填充颜色，例如 'skyblue'、'#ff9999'                                   |
| `alpha`      | 透明度，范围 [0,1]，越小越透明                                       |
| `label`      | 图例标签                                                               |
| `linewidth`  | 填充边缘线宽，默认无边缘线  


```python
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ["STHeiTi"]
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(0,10,500)
y1 = np.sin(x)
y2 = np.sin(x) + 0.5

plt.plot(x,y1,label='y1')
plt.plot(x,y2,label='y2')

plt.fill_between(x,y1,y2,color='skyblue',alpha=0.8,label="Filled Area")

plt.legend()
plt.title("Fill_between 示例")
plt.show()
```

![填充图.png](../imgs/analysis/matplotlib/%E5%A1%AB%E5%85%85%E5%9B%BE.png)


```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 200)
y = np.sin(x)

plt.plot(x, y, label='y = sin(x)')

# 只在 y > 0 的地方填充
plt.fill_between(x, y, 0, where=(y > 0), color='green', alpha=0.3, label='y > 0')
# 只在 y < 0 的地方填充
plt.fill_between(x, y, 0, where=(y < 0), color='red', alpha=0.3, label='y < 0')

plt.axhline(0, color='black', linewidth=1)
plt.legend()
plt.title("fill_between + where 示例")
plt.show()
```

![填充图+where.png](../imgs/analysis/matplotlib/%E5%A1%AB%E5%85%85%E5%9B%BE%2Bwhere.png)


#### 8. 图像显示

| 参数            | 说明 |
|-----------------|------|
| `X`             | 输入的二维数组（图像数据） |
| `cmap`          | 颜色映射表，例如 `'gray'`, `'viridis'`, `'hot'` 等 |
| `interpolation`| 插值方法，如 `'nearest'`, `'bilinear'`, `'bicubic'` 等，控制图像放缩方式 |
| `norm`          | 归一化方式，如 `Normalize()`，用于调整显示范围 |
| `vmin`, `vmax`  | 手动设置颜色映射范围的最小值和最大值 |
| `alpha`         | 透明度（0~1） |
| `extent`        | 指定图像边界范围 `[xmin, xmax, ymin, ymax]` |
| `origin`        | 原点位置：`'upper'`（默认，左上）或 `'lower'`（左下） |
| `aspect`        | 图像长宽比，常见为 `'auto'`, `'equal'` |

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.rand(10, 10)  # 10x10 随机灰度图
plt.imshow(data, cmap='viridis', interpolation='nearest',extent=[0, 5, 0, 5])
plt.colorbar()  # 显示颜色条
plt.title("示例图像")
plt.show()
```

![图像显示.png](../imgs/analysis/matplotlib/%E5%9B%BE%E5%83%8F%E6%98%BE%E7%A4%BA.png)
















