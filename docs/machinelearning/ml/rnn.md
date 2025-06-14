# 第十章：循环神经网络

### 1️⃣ 什么是 RNN？

循环神经网络（Recurrent Neural Network, RNN）是一类用于处理**序列数据**的神经网络。它通过“记忆”前面时间步的状态，使得模型具备处理时间上下文的能力，适用于语音、文本、时间序列等任务。


### 2️⃣ RNN 原理与结构

**单个时间步公式：**

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h) \\
y_t = W_{hy} h_t + b_y
$$

- \( x_t \)：当前输入  
- \( h_{t-1} \)：上一步隐藏状态（记忆）  
- \( h_t \)：当前隐藏状态  
- \( y_t \)：输出  
- \( W \)：权重矩阵，\( b \)：偏置项  

**图示结构（展开形式）：**

```text
x₁ ──▶[RNN]──▶ h₁ ──▶ y₁
▲
x₂ ──▶[RNN]──▶ h₂ ──▶ y₂
▲
x₃ ──▶[RNN]──▶ h₃ ──▶ y₃
```

###  3️⃣ RNN 存在的问题

| 问题                | 描述                                     |
|---------------------|------------------------------------------|
| 梯度消失 / 爆炸     | 难以训练长序列，学习长期依赖困难         |
| 短期记忆能力有限    | 只能捕捉最近的信息                       |
| 参数更新不稳定      | 因为每一步都要反向传播，优化困难         |

✅ **改进方案**：LSTM / GRU

###  4️⃣ LSTM：长短期记忆网络


LSTM（Long Short-Term Memory）通过**门控机制**解决了 RNN 的长期依赖问题。

#### 门控结构

1. **遗忘门**：  
   $$ f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) $$
2. **输入门**：  
   $$ i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) $$
3. **候选记忆**：  
   $$ \tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) $$
4. **更新记忆**：  
   $$ C_t = f_t * C_{t-1} + i_t * \tilde{C}_t $$
5. **输出门**：  
   $$ o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) $$
6. **隐藏状态**：  
   $$ h_t = o_t * \tanh(C_t) $$

✅ 支持长期依赖  
✅ 应用于大规模 NLP、语音、时间序列任务


### 5️⃣ Seq2Seq 模型（编码器-解码器）

**Sequence-to-Sequence（序列到序列）模型**用于将输入序列转换为输出序列（如翻译、摘要、对话系统）。

#### 基本结构：

```text
输入序列 → 编码器RNN → 上下文向量 → 解码器RNN → 输出序列
```

- **编码器（Encoder）**：读取输入序列，输出最后隐藏状态 \( h_T \)
- **上下文向量（Context）**：用于初始化解码器
- **解码器（Decoder）**：以 context 启动，逐步生成输出序列

| 模块     | 作用                             |
|----------|----------------------------------|
| 编码器   | 压缩整个输入为一个隐藏状态       |
| 解码器   | 使用隐藏状态生成目标序列         |
| 问题     | 上下文向量压缩所有信息，效果受限 |


###  6️⃣ 注意力机制（Attention）

注意力机制让 Seq2Seq 更强大，解码器每一步都能**选择性关注输入序列的不同部分**。

#### 原理：

- 计算当前解码器隐藏状态 \( h_t \) 与所有编码器输出 \( h_i \) 的相关性
- 使用 softmax 得到注意力权重 \( \alpha_{t,i} \)
- 计算上下文向量：

$$
\alpha_{t,i} = \text{softmax}(score(h_t, h_i)) \\
c_t = \sum_i \alpha_{t,i} h_i
$$

| 模型类型         | 描述                              |
|------------------|-----------------------------------|
| 加性注意力       | Bahdanau attention                |
| 点积注意力       | Luong attention                   |
| 自注意力（Self-Attention） | Transformer 中核心组件 |

✅ 支持长序列建模  
✅ 动态上下文建模能力强  
✅ 是 Transformer 的核心构建块


### 7️⃣ PyTorch 实现简易版（LSTM）

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])  # 使用最后隐藏状态
        return out

model = LSTMClassifier(input_dim=100, hidden_dim=128, output_dim=2)
```

### 8️⃣ 各模型对比总结

| 模型       | 特点                                | 是否支持长依赖 | 应用场景              |
|------------|-------------------------------------|----------------|-----------------------|
| RNN        | 结构简单，短期记忆                  | 否             | 简单文本、传感器数据  |
| LSTM       | 有门控结构，支持长期依赖            | 是             | NLP、语音、时间序列   |
| GRU        | 简化版 LSTM，效率更高               | 是             | NLP、序列分类         |
| Seq2Seq    | 编码-解码结构，适合序列生成         | 有限           | 翻译、对话生成        |
| Seq2Seq + Attention | 解决信息瓶颈问题          | 强             | 翻译、摘要、问答系统  |

### 9️⃣ 应用场景

- 📖 机器翻译（LSTM + Seq2Seq + Attention）
- 🧠 情感分析
- 📈 股票/天气预测（时间序列）
- 🗣️ 语音识别
- 📝 文本摘要、文本生成
- 💬 ChatBot、对话系统

###  🔟 总结

| 项目             | 描述                                               |
|------------------|----------------------------------------------------|
| 输入类型         | 序列数据（文本、语音、时间序列）                  |
| 基本结构         | RNN → LSTM / GRU → Seq2Seq → 注意力机制            |
| 训练方式         | 前向传播 + 反向传播 + 时间反向传播（BPTT）        |
| 框架实现         | PyTorch、TensorFlow、Keras                         |
| 进阶替代方案     | Transformer（完全基于注意力，无 RNN）             |

