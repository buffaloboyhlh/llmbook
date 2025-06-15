# 第十一章：强化学习

## 一、强化学习基础

### 1.1 初探强化学习

#### 1️⃣ 简介

强化学习（Reinforcement Learning, RL）是一种重要的机器学习范式，它通过**智能体（Agent）与环境（Environment）之间的交互**，让智能体通过试错的方式学习行为策略，从而实现特定目标或最大化长期收益。

强化学习被广泛应用于以下领域：

- 游戏智能（如 AlphaGo、Dota2 AI）
- 机器人控制
- 自动驾驶
- 智能推荐系统
- 金融交易策略

强化学习强调的是在未知环境中通过行动获得反馈，并基于反馈不断优化决策。

####  2️⃣ 什么是强化学习？

强化学习是一个**基于奖励反馈**的学习过程。智能体在每一步会根据当前状态采取一个动作，环境根据该动作返回一个奖励值和下一个状态，智能体的目标是学习一种策略，使得它在长期内获得尽可能多的奖励。

##### 强化学习的基本构成

| 术语 | 含义 |
|------|------|
| **Agent（智能体）** | 做出决策的学习系统 |
| **Environment（环境）** | 智能体所交互的外部世界 |
| **State（状态）** | 当前的环境情况 |
| **Action（动作）** | 智能体采取的操作 |
| **Reward（奖励）** | 环境对动作的反馈 |
| **Policy（策略）** | 决定在每个状态下选择什么动作的准则 |
| **Value Function（价值函数）** | 评估一个状态或状态-动作组合的未来回报 |


#### 3️⃣ 强化学习的数学目标

强化学习通常使用马尔可夫决策过程（MDP）来建模：

$$
\text{MDP} = (S, A, P, R, \gamma)
$$

其中：

- $S$：状态空间；
- $A$：动作空间；
- $P(s'|s,a)$：状态转移概率；
- $R(s,a)$：奖励函数；
- $\gamma \in (0,1]$：折扣因子，衡量未来奖励的重要性。

强化学习的目标是找到一个最优策略 $\pi^*$，使得期望累积折扣奖励最大化：

$$
\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

#### 4️⃣ 强化学习与监督学习的对比

| 特征 | 监督学习 | 强化学习 |
|------|----------|----------|
| 数据类型 | 输入-输出对（已标注） | 与环境交互数据 |
| 学习目标 | 拟合函数映射 | 学习策略最大化长期回报 |
| 反馈方式 | 立即、明确 | 延迟、稀疏、间接 |
| 应用示例 | 图像识别、文本分类 | 博弈、机器人控制 |


#### 5️⃣ 小结

强化学习是一种能够通过环境反馈不断自我优化的学习方式，适用于具有**序列决策特性**的问题。它与监督学习和无监督学习互为补充，是构建通用人工智能的重要组成部分。

### 1.2 多臂老虎机问题

**多臂老虎机问题（Multi-Armed Bandit, MAB）** 是强化学习中最基础的问题之一。它描述了一个**探索与利用（exploration vs exploitation）**的核心困境：

> 一位赌徒面对 $K$ 个老虎机，每个老虎机的奖励分布不同。赌徒的目标是在有限次数的尝试中，最大化总奖励。

#### 目标定义


在第 $t$ 步选择一个臂 $A_t$，获得奖励 $R_t$，目标是最大化总期望奖励：

$$
\max \mathbb{E} \left[ \sum_{t=1}^{T} R_t \right]
$$

等价于最小化**后悔（Regret）**：

$$
\text{Regret}(T) = T \mu^* - \mathbb{E} \left[ \sum_{t=1}^{T} R_t \right]
$$

其中：

- $\mu^*$ 是最优臂的期望奖励；
- $T$ 是总轮数；
- $R_t$ 是第 $t$ 步获得的奖励。

####  策略算法

##### 1️⃣ ε-Greedy 策略

- **思想**：大部分时间选择当前估计最优臂（利用），小部分时间随机探索其他臂（探索）。

- **算法步骤**：
  1. 初始化每个臂的平均奖励估计值；
  2. 在每一轮：
     - 以概率 $1 - \varepsilon$ 选择当前估计最优臂；
     - 以概率 $\varepsilon$ 随机选择一个臂；
  3. 更新所选臂的估计值。

     - **更新公式**（均值估计）：
       $$
       \hat{\mu}_i \leftarrow \hat{\mu}_i + \frac{1}{N_i}(r - \hat{\mu}_i)
       $$


##### 2️⃣ Upper Confidence Bound（UCB）

- **思想**：选择置信上界最大的臂，即“既考虑均值也考虑不确定性”。

- **选择臂公式**：
  $$
  a_t = \arg\max_i \left[ \hat{\mu}_i + \sqrt{\frac{2 \ln t}{N_i}} \right]
  $$

- 解释：
    - $\hat{\mu}_i$：臂 $i$ 的当前平均奖励；
    - $N_i$：臂 $i$ 被拉过的次数；
    - $\ln t$：对不确定性的惩罚逐步减弱。

##### 3️⃣ 汤普森采样（Thompson Sampling）


- **思想**：为每个臂维护一个概率分布，从中采样并选择最大值臂。

- **过程**：
    - 对每个臂 $i$ 建立贝叶斯分布（如 Beta 分布）；
    - 每一轮对每个臂从其分布中采样；
    - 选择最大值对应的臂；
    - 根据实际奖励更新该臂的分布。

- **二项奖励时**：
  $$
  \text{Beta}(\alpha_i, \beta_i) \quad \text{更新为：}
  \begin{cases}
    \alpha_i \leftarrow \alpha_i + 1 & \text{如果获奖} \\
    \beta_i \leftarrow \beta_i + 1 & \text{否则}
  \end{cases}
  $$

####  Python 简易代码示例（ε-greedy）

```python
import numpy as np

# 模拟老虎机
true_means = [0.2, 0.5, 0.75]  # 三个臂的真实概率
n_arms = len(true_means)
n_rounds = 1000
epsilon = 0.1

# 初始化
counts = np.zeros(n_arms)
values = np.zeros(n_arms)
rewards = []

for t in range(n_rounds):
    if np.random.rand() < epsilon:
        action = np.random.choice(n_arms)
    else:
        action = np.argmax(values)

    reward = np.random.rand() < true_means[action]  # 二值奖励
    counts[action] += 1
    values[action] += (reward - values[action]) / counts[action]
    rewards.append(reward)

print(f"平均奖励: {np.mean(rewards):.3f}")
```

#### 应用场景

+ 在线广告投放
+ 推荐系统（AB 测试）
+ 动态定价
+ 医疗试验（多药测试）

### 1.3 马尔可夫决策过程

#### 1️⃣ 什么是马尔可夫决策过程？


马尔可夫决策过程（MDP）是强化学习中建模环境和智能体交互行为的数学框架。  
其核心思想是：

> 一个智能体在每个时刻处于某个状态中，根据策略选择动作，环境根据动作转移到新状态并给予奖励。


#### 2️⃣ MDP 的五元组定义

MDP 通常由以下五部分组成：

| 符号 | 含义 |
|------|------|
| \( S \) | 状态空间（States）：环境中所有可能的状态集合 |
| \( A \) | 动作空间（Actions）：智能体在每个状态可采取的动作集合 |
| \( P(s' \mid s, a) \) | 状态转移概率：在状态 \(s\) 下采取动作 \(a\) 转移到状态 \(s'\) 的概率 |
| \( R(s, a) \) | 奖励函数：采取动作 \(a\) 后获得的即时奖励 |
| \( \gamma \) | 折扣因子（Discount factor）：未来奖励的重要性，取值 \(0 \leq \gamma \leq 1\) |


#### 3️⃣ 马尔可夫性（Markov Property）

MDP 满足马尔可夫性质：

> 当前状态的转移概率只与当前状态和当前动作有关，与过去历史无关。

数学表达式如下：

\[
P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_1, a_1, \dots, s_t, a_t)
\]


#### 4️⃣ 目标：最优策略

目标是找到一个策略 \( \pi \)，使得期望累积奖励最大化：

\[
\pi^* = \arg\max_{\pi} \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right]
\]

#### 5️⃣ 值函数（Value Function）

##### 5.1 状态值函数 \( V^\pi(s) \)

\[
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \mid s_0 = s \right]
\]

##### 5.2 动作值函数 \( Q^\pi(s, a) \)

\[
Q^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \mid s_0 = s, a_0 = a \right]
\]

#### 6️⃣ 贝尔曼方程（Bellman Equation）


##### 6.1 策略下的贝尔曼方程：

\[
V^\pi(s) = \sum_{a} \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]
\]

##### 6.2 最优贝尔曼方程：

\[
V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]
\]

#### 7️⃣ 求解 MDP 的方法

##### 7.1 动态规划（DP）类（已知模型）

- 值迭代（Value Iteration）
- 策略迭代（Policy Iteration）

##### 7.2 强化学习（RL）类（未知模型）

- Q-learning
- SARSA
- DQN（深度 Q 网络）

#### 8️⃣ 代码示例（FrozenLake）

```python
import gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

V = np.zeros(n_states)
gamma = 0.9

# 值迭代
for i in range(1000):
    V_prev = V.copy()
    for s in range(n_states):
        Q_sa = []
        for a in range(n_actions):
            q = 0
            for prob, s_prime, reward, done in env.P[s][a]:
                q += prob * (reward + gamma * V_prev[s_prime])
            Q_sa.append(q)
        V[s] = max(Q_sa)
    if np.max(np.abs(V - V_prev)) < 1e-4:
        break
```

### 1.4 动态规划算法




## 二、强化学习进阶



## 三、强化学习前沿

