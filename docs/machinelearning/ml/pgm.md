# 第六章：概率图模型

---

### 1️⃣ 什么是概率图模型？

概率图模型（PGM）是一种结合图论和概率论的方法，用于表示多个随机变量之间的联合概率分布及其条件依赖关系。它能够有效建模复杂系统中的不确定性。

---

### 2️⃣ 类型划分

| 模型类型                               | 图结构           | 主要特点               |
|--------------------------------------|------------------|------------------------|
| 贝叶斯网络（Bayesian Network, BN）  | 有向无环图（DAG） | 表示因果关系，方向明确 |
| 马尔可夫网络（Markov Random Field） | 无向图           | 表示对称关系，无方向性 |

---

### 3️⃣ 图结构表示

图中的每个节点代表一个随机变量，边代表变量间的依赖关系。

例如：

Weather → WetRoad → Accident

表示天气影响路面湿滑，路面湿滑影响交通事故。

---

### 4️⃣ 贝叶斯网络（Bayesian Network）

贝叶斯网络由有向无环图和条件概率分布组成。

#### 联合概率分布公式

设随机变量集合为 \(X_1, X_2, \dots, X_n\)，则它们的联合概率分布可表示为：

$$
P(X_1, X_2, \dots, X_n) = \prod_{i=1}^n P\big(X_i \mid \mathrm{Pa}(X_i)\big)
$$

其中，\(\mathrm{Pa}(X_i)\) 表示节点 \(X_i\) 的父节点集合。

---

### 5️⃣ 马尔可夫网络（Markov Random Field）

马尔可夫网络使用无向图，利用团的势函数定义联合概率。

#### 联合概率分布公式

$$
P(X) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \psi_C(X_C)
$$

其中：

- \(\mathcal{C}\) 是图中所有最大团的集合  
- \(\psi_C(X_C)\) 是对应团上的势函数  
- \(Z\) 是归一化常数，称为配分函数（partition function）

---

### 6️⃣ 条件独立性判断

| 模型类型     | 条件独立性判断方法        |
|--------------|---------------------------|
| 贝叶斯网络   | d-分离（d-separation）     |
| 马尔可夫网络 | 图分割（Graph Separation） |

---

### 7️⃣ 推理（Inference）

推理是根据已知变量的观测值，计算其他变量的后验概率。

| 推理类型 | 常见方法                           | 说明                       |
|----------|----------------------------------|----------------------------|
| 精确推理 | 变量消除（Variable Elimination）、信念传播（Belief Propagation） | 计算精确，但计算复杂度高   |
| 近似推理 | 马尔可夫链蒙特卡洛（MCMC）、变分推断（Variational Inference） | 适合大规模模型，近似计算   |

---

### 8️⃣ 学习（Learning）

学习分为参数学习和结构学习：

| 学习类型 | 内容描述                         | 常用方法                           |
|----------|--------------------------------|----------------------------------|
| 参数学习 | 在给定图结构下估计概率参数       | 极大似然估计（MLE）、EM算法、贝叶斯估计 |
| 结构学习 | 学习变量间依赖关系图结构         | 搜索算法、评分函数（如BIC、AIC）   |

---

### 9️⃣ Python 示例：贝叶斯网络构建与推理

```python
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 定义网络结构
model = BayesianNetwork([('Weather', 'WetRoad'), ('WetRoad', 'Accident')])

# 定义条件概率分布
cpd_weather = TabularCPD(variable='Weather', variable_card=2, values=[[0.7], [0.3]])
cpd_wetroad = TabularCPD(variable='WetRoad', variable_card=2,
                        values=[[0.9, 0.4],
                                [0.1, 0.6]],
                        evidence=['Weather'],
                        evidence_card=[2])
cpd_accident = TabularCPD(variable='Accident', variable_card=2,
                         values=[[0.95, 0.2],
                                 [0.05, 0.8]],
                         evidence=['WetRoad'],
                         evidence_card=[2])

# 添加 CPD
model.add_cpds(cpd_weather, cpd_wetroad, cpd_accident)

# 验证模型正确性
model.check_model()

# 推理
inference = VariableElimination(model)
posterior_accident = inference.query(variables=['Accident'], evidence={'Weather': 1})
print(posterior_accident)
```

