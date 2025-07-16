# Matplotlib 教程

## 一、简介

Matplotlib 是 Python 中最常用的数据可视化库，提供类似 MATLAB 的绘图 API，可以绘制各种静态、动态、交互式图表。

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