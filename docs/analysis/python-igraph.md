# Python-igraph 教程

## 一、igraph 简介

**igraph** 是一个专门用于图论分析的开源库，支持：

- 创建图（Graph）和网络（Network）
- 计算网络指标（如度、聚类系数、最短路径等）
- 社区发现（community detection）
- 网络可视化

安装：

```bash
pip install igraph
pip install cairocffi  # 若需要可视化支持
```

## 二、基本概念与图创建

#### 1. 导入与基本结构

```python
import igraph as ig
```

#### 2. 创建图（无向图 / 有向图）

```python
# 创建一个无向图
g = ig.Graph()

# 添加 5 个节点
g.add_vertices(5)

# 添加边（顶点编号从0开始）
g.add_edges([(0,1), (1,2), (2,3), (3,4), (4,0)])
```

#### 3. 使用有向图

```python
g = ig.Graph(directed=True)
g.add_vertices(["A", "B", "C"])
g.add_edges([("A", "B"), ("B", "C")])
```

## 三、图的属性设置

```python
# 节点名称
g.vs["label"] = ["A", "B", "C", "D", "E"]

# 边权重
g.es["weight"] = [1.0, 2.5, 1.2, 0.5, 3.0]

# 自定义属性
g.vs["color"] = ["red", "green", "blue", "yellow", "pink"]
```

## 四、图的常用分析函数

#### 1. 节点和边的数量

```python
g.vcount()  # 顶点数量
g.ecount()  # 边数量
```

#### 2. 度（Degree）

```python
g.degree()               # 所有节点的度
g.degree(0)              # 单个节点的度
g.degree(mode="in")      # 入度（有向图）
g.degree(mode="out")     # 出度（有向图）
```

#### 3. 邻接节点

```python
g.neighbors(0)  # 返回第0号节点的邻居
```

#### 4. 最短路径

```python
g.shortest_paths(source=0, target=[2,4])
```

#### 5. 聚类系数

```python
g.transitivity_undirected()
```

##  五、社区发现算法

#### 1. Louvain 社区划分

```python
g = ig.Graph.Famous("Zachary")
communities = g.community_multilevel()
print(communities.membership)
```

#### 2. Label Propagation

```python
communities = g.community_label_propagation()
```


## 六、图的可视化

```python
layout = g.layout("circle")  # 其他布局有：fr、kk、grid_fr等

ig.plot(g,
        layout=layout,
        vertex_label=g.vs["label"],
        vertex_color=g.vs["color"],
        edge_width=[2 + w for w in g.es["weight"]])
```

**保存图像：**

```python
ig.plot(g, target="graph.png", layout=layout)
```

## 七、示例：构建并分析社交网络

```python
g = ig.Graph(directed=False)
g.add_vertices(["Alice", "Bob", "Claire", "Dennis", "Emma"])
g.add_edges([("Alice", "Bob"), ("Alice", "Claire"), ("Bob", "Dennis"), ("Claire", "Emma")])
g.vs["label"] = g.vs["name"]

# 度中心性
print(g.degree())

# 可视化
layout = g.layout("fr")
ig.plot(g, layout=layout, vertex_label=g.vs["label"])
```

## 八、常用内置图与数据导入

#### 1. 生成常见图

```python
g = ig.Graph.Famous("Zachary")  # 查克利空手道俱乐部图
```

#### 2. 从边列表创建

```python
edges = [("A", "B"), ("B", "C")]
g = ig.Graph.TupleList(edges, directed=False)
```

#### 3. 从邻接矩阵导入

```python
import numpy as np
adj = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 0]
])
g = ig.Graph.Adjacency((adj > 0).tolist())
```

##  九、进阶：图指标与中心性

```python
# 度中心性
g.degree()

# 接近中心性
g.closeness()

# 中介中心性（Betweenness）
g.betweenness()

# Pagerank
g.pagerank()

# Cliques（完全子图）
g.cliques(min=3)
```

## 十、总结表格（常用函数速查）

| 功能             | 函数                                |
|------------------|-------------------------------------|
| 添加节点         | `g.add_vertices()`                   |
| 添加边           | `g.add_edges()`                      |
| 顶点数 / 边数    | `g.vcount()` / `g.ecount()`          |
| 度中心性         | `g.degree()`                         |
| 接近中心性       | `g.closeness()`                      |
| 中介中心性       | `g.betweenness()`                    |
| PageRank         | `g.pagerank()`                       |
| 最短路径         | `g.shortest_paths()`                 |
| 邻居节点         | `g.neighbors()`                      |
| 社区发现（Louvain） | `g.community_multilevel()`          |
| 社区发现（标签传播） | `g.community_label_propagation()`  |
| 聚类系数         | `g.transitivity_undirected()`        |
| 可视化图         | `igraph.plot()`                      |
| 预定义图         | `igraph.Graph.Famous()`              |
| 从边列表创建图   | `igraph.Graph.TupleList()`           |
| 从邻接矩阵创建图 | `igraph.Graph.Adjacency()`           |



