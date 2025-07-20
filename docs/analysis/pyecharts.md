# pyecharts 教程

## 一、简介

## 二、基础

## 三、实战

### 1. 柱状图

```python
from pyecharts.charts import Bar
from pyecharts import options as opts 
from pyecharts.faker import Faker
from pyecharts.globals import ThemeType

bar = Bar(init_opts=opts.InitOpts(
    theme=ThemeType.LIGHT,
    width='1000px',
    height='600px'
))

bar.add_xaxis(Faker.choose()) 

# stack值一样的系列会堆叠在一起 所以 A 和 B 堆叠在一起
bar.add_yaxis('A',Faker.values(),stack="stack1")
bar.add_yaxis('B',Faker.values(),stack='stack1')
bar.add_yaxis('C',Faker.values(),stack="stack2")

bar.set_global_opts(toolbox_opts=opts.ToolboxOpts(is_show=True))
bar.render_notebook()
```
![柱状图.png](../imgs/pyecharts/%E6%9F%B1%E7%8A%B6%E5%9B%BE.png)

```python
from pyecharts import options as opts 
from pyecharts.charts import Bar
from pyecharts.faker import Faker
from pyecharts.globals import ThemeType

bar = Bar(init_opts=opts.InitOpts(
    theme=ThemeType.LIGHT,
))

bar.add_xaxis(Faker.choose())
bar.add_yaxis('',Faker.values())
bar.set_series_opts(label_opts=opts.LabelOpts(
    position='insideLeft',
    formatter="{b}:{c}"
))
# 隐藏坐标系
bar.set_global_opts(xaxis_opts=opts.AxisOpts(is_show=False),
                    yaxis_opts=opts.AxisOpts(is_show=False),
                    toolbox_opts=opts.ToolboxOpts(is_show=True)
                    )
bar.reversal_axis() # 反转坐标系
bar.render_notebook()
```
![柱状图关闭坐标轴.png](../imgs/pyecharts/%E6%9F%B1%E7%8A%B6%E5%9B%BE%E5%85%B3%E9%97%AD%E5%9D%90%E6%A0%87%E8%BD%B4.png)


