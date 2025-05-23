# 注意力机制

---

## 一、缩放点积注意力（SDPA）

2014 年《Neural Machine Translation by Jointly Learning to Align and Translate》提出的单头注意力，输入的 Query、Key 和 Value
矩阵都是完整的张量。

缩放点积注意力早于 Transformer 被提出，受到的关注并不多，其内部只实现了 $q,k,v$ 的注意力计算。

+ 输入是 query 和 key-value，注意力机制首先计算 query 与每个 key 的关联性
+ 每个关联性作为每个 value 的权重 (weight)，各个权重与 value 的乘积相加得到输出。
+ **SDPA 可以被认为是 MHA 的中间步骤！**

#### 1.1 注意力机制的基本数学形式

最基础的注意力机制是 Scaled Dot-Product Attention（缩放点积注意力），公式如下：

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V
$$

+ Q：Query（查询）
+ K：Key（键）
+ V：Value（值）
+ d_k：Key 的维度（用于缩放）

#### 1.2 注意力掩码（Attention Mask）

在注意力机制中，**掩码（Mask）用于控制注意力的计算范围**，以便模型能够忽略某些 token，或避免访问未来的信息。

##### 🧩 核心作用：

在注意力公式中加入 mask，使得某些位置的注意力分数变为极小值（负无穷），从而在 softmax 后变成 0，不参与加权求和：

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} + \text{mask} \right) V
$$

##### 📚 常见类型

| Mask 名称                | 作用                            | 应用场景（模型）                  |
|------------------------|-------------------------------|---------------------------|
| `padding_mask`         | 忽略填充符$（<PAD>）$的位置             | 所有 Encoder/Decoder 模型     |
| `causal_mask`          | 防止注意力“偷看”未来的信息                | GPT / Transformer Decoder |
| `combined_mask`        | 同时考虑 padding 和 causal         | 解码器中的自注意力层                |
| `cross_attention_mask` | 控制 cross-attention 中 key 的有效性 | Transformer 解码器交叉注意力      |

###### 🔎 一、Padding Mask（填充掩码）

在 NLP 中，输入长度常不同，需要统一成定长，用 <PAD> 进行填充。为了不让模型关注这些填充值，需要屏蔽掉它们。

原始输入：

```python
input_1 = [我, 爱, 你]
input_2 = [你, 好]
```

填充后（长度统一为 4）：

```python
input_1 = [我, 爱, 你, < PAD >]
input_2 = [你, 好, < PAD >, < PAD >]
```

对应的 padding_mask：

```python
mask_1 = [1, 1, 1, 0]
mask_2 = [1, 1, 0, 0]
```

###### ⏳ 二、Causal Mask（因果掩码）

在自回归生成任务中（如 GPT），当前单词不能看到未来单词。Causal Mask 保证了信息只从过去流向现在，避免“未来泄漏”。

生成一个 下三角矩阵（Lower Triangular Mask）：

```python
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

注意力计算中，会将上三角区域的 attention scores 设置为 -∞。

```python
seq_len = 4
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
```

###### 🔀 三、Combined Mask（联合掩码）

在实际使用中，常常需要同时使用 padding mask 和 causal mask，尤其是在 Transformer 的解码器中。

假设有一个句子：

```python
input = [我, 爱, < PAD >]
padding_mask = [1, 1, 0]

causal_mask =
[[1, 0, 0],
 [1, 1, 0],
 [1, 1, 1]]
```

合并后：

```python
combined_mask = causal_mask & padding_mask_broadcast
```

实现方式如下：

```python
combined_mask = causal_mask & padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
```

#### 1.3 实现代码

```python
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, casual_mask=None, padding_mask=None):
        '''
        query, key, value 形状: (batch_size, seq_len, hidden_size)
        :param query:
        :param key:
        :param value:
        :param casual_mask: 因果掩码
        :param padding_mask: 填充掩码
        :return:
        '''

        d_k = query.size(-1)  # 获取 hidden_size

        # 计算注意力分数
        # key.transpose(-1, -2) 将最后两个维度进行转置，以进行点积
        # attention_scores 形状: (batch_size, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32))

        # 添加注意力掩码（seq_len, seq_len），掩码位置（1）的值为负无穷
        if casual_mask is not None:
            attention_scores += casual_mask * -1e9  # -1e9 代表负无穷

        # 添加填充位置的掩码，每个句子不一样（batch_size, seq_len)
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)  # 扩展成与注意力权重矩阵可广播的形状
            attention_scores += padding_mask * -1e9

        # 对注意力分数进行归一化，得到注意力概率
        attention_probs = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        # 计算注意力输出，通过注意力概率加权值

        attention_output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, hidden_size)

        return attention_output
```

**验证attention**

```python
def test_attention():
    batch_size = 16
    seq_length = 32
    hidden_size = 512

    query = torch.randn(batch_size, seq_length, hidden_size)
    key = torch.randn(batch_size, seq_length, hidden_size)
    value = torch.randn(batch_size, seq_length, hidden_size)

    attention = ScaledDotProductAttention()
    output = attention(query, key, value, casual_mask=None, padding_mask=None)

    print("Query shape:", query.shape)
    print("Key shape:", key.shape)
    print("Value shape:", value.shape)
    print("Output shape:", output.shape)


test_attention()
```

**输出**

```text
Query shape: torch.Size([16, 32, 512])
Key shape: torch.Size([16, 32, 512])
Value shape: torch.Size([16, 32, 512])
Output shape: torch.Size([16, 32, 512])
```

## 二、多头注意力（MHA）

多头注意力机制是 Transformer 模型中的核心组件。在其设计中，「多头」意味着该机制并不只计算一种注意力权重，而是并行计算多种权重，每种权重都从不同的「视角」捕获输入的不同信息。具体步骤如下：

1、为输入序列中计算 Q, K, V ，这是通过将输入词向量与三个权重矩阵相乘实现的:

$$
\begin{aligned} & Q = X W_q \\ & K = X W_k \\ & V = X W_v \end{aligned}
$$

2、计算 Q, K 注意力得分，其中， $d_k$ 是$k$的维度：

$$
\operatorname{score}(Q, K) = \frac{Q \cdot K^T}{\sqrt{d_k}}
$$

3、使用 Softmax 得到注意力权重：

$$
\operatorname{Attention}(Q, K) = \operatorname{softmax}(\operatorname{score}(Q, K))=\operatorname{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)
$$

4、使用注意力权重和V ，计算输出：

$$
\text{Output} = \operatorname{Attention}(Q, K) \cdot V = \operatorname{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V
$$

5、 拼接多头输出，乘以 $W_O$，得到最终输出：

$$
\text{MultiHeadOutput} = \text{Concat} (\text{Output}^1, \text{Output}^2, \ldots, \text{Output}^H) W_O
$$


```python
import torch
from torch import nn

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 每个头的维度，二者必须整除
        
        # 初始化 Q、K、V 的投影矩阵，将输入词向量线性变换为 Q、K、V，维度保持一致
        self.q_linear = nn.Linear(hidden_size, hidden_size) 
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        # 输出线性层，将拼接后的多头注意力输出变换为所需的输出维度，这里维度保持一致
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_state, causal_mask=None, padding_mask=None):
        # hidden_state 形状: (batch_size, seq_len, hidden_size)
        batch_size = hidden_state.size(0)  # 获取批量大小

        # 计算 Q、K、V，线性变换，得到形状：(batch_size, seq_len, hidden_size)
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)
        
        # 将每个头的维度拆分出来，得到形状：(batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数，attention_scores 形状: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) \
        / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 添加因果注意力掩码（seq_len, seq_len），掩码位置（1）的值为负无穷，自动广播
        if causal_mask is not None:
            attention_scores += causal_mask * -1e9
        
        # 添加填充位置的掩码，每个句子不一样（batch_size, seq_len)
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            attention_scores += padding_mask * -1e9
            
        # 对注意力分数进行归一化，得到注意力概率
        attention_probs = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        # 如果有 dropout 操作就加在这，self.dropout(attention_probs)，也可以在函数外面加

        # 计算注意力输出，通过注意力概率加权值
        output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 对多头注意力输出进行拼接，将形状调整为 (batch_size, seq_len, hidden_size)
        # 先 output.transpose(1, 2) 将 num_heads 和 seq_len 维度转置
        output = output.transpose(1, 2).view(batch_size, -1, self.head_dim * self.num_heads)
        
        # 通过线性层将拼接后的输出变换为所需的输出维度
        output = self.o_linear(output)  # (batch_size, seq_len, hidden_size)
        return output

def test_MHA():
    batch_size = 128
    seq_len = 512
    hidden_size = 1024
    num_heads = 8
    
    # 随机生成输入数据
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)  # (batch_size, seq_len, hidden_size)
    
    # 生成因果掩码（下三角矩阵），这里就不刻意生成 padding_mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    # 创建多头注意力模块
    mha = MultiHeadAttention(hidden_size, num_heads)
    
    # 计算多头注意力输出
    output = mha(hidden_state, causal_mask=causal_mask)
    
    print("Input shape:", hidden_state.shape)
    print("Output shape:", output.shape)
    
if __name__ == "__main__":
	test_MHA()
```

## 三、MHA with KV Cache

键值缓存（KV Cache）主要用于 Decoder 的 Next Token Prediction 时减少重复计算，通过缓存并逐步更新键（Key）和值（Value），来用空间换时间。

但要注意即使是 Decoder-only 的模型，在预处理输入（Prefill）的时候也不需要利用 KV Cache（P/D 分离），本代码只作为示例：

```python
import torch
from torch import nn

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size) 
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_state, causal_mask=None, past_key_value=None, use_cache=False):
        batch_size = hidden_state.size(0)
        
        # 计算 Q、K、V，注意此时只有一个 Token
        query = self.q_linear(hidden_state)  # (batch_size, 1, hidden_size)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)
        
        # 分割多头，得到形状：(batch_size, num_heads, 1, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 若存在缓存，拼接当前 K、V
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)	  # (batch_size, num_heads, seq_len, head_dim)
            value = torch.cat([past_value, value], dim=2)
        
        # 保存新的缓存
        new_past_key_value = (key, value) if use_cache else None
        
        # 计算注意力分数，attention_scores 形状: (batch_size, num_heads, 1, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) \
        / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 应用因果掩码（若需要）
        if causal_mask is not None:
            attention_scores += causal_mask * -1e9
            
        # 计算注意力输出
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, value)
        
        # 合并多头并线性变换
        output = output.transpose(1, 2).view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.o_linear(output)
        
        return (output, new_past_key_value) if use_cache else output

def test_MHA_with_cache():
    batch_size = 2
    seq_len = 5
    hidden_size = 64
    num_heads = 4
    
    # 构造输入，模拟整个序列
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    # 刻意分步处理，使用 KV 缓存
    past_key_value = None
    outputs = []
    for i in range(seq_len):
        # 当前步骤的输入（单个 token）
        current_input = hidden_state[:, i:i+1, :]
        # 生成当前步骤的因果掩码（仅允许关注到当前位置及之前的）
        current_causal_mask = causal_mask[i:i+1, :i+1]  # (1, i+1)
        # 前向传播
        output_step, past_key_value = mha(
            current_input,
            causal_mask=current_causal_mask,
            past_key_value=past_key_value,
            use_cache=True
        )
        outputs.append(output_step)
    
    # 合并分布输出
    output = torch.cat(outputs, dim=1)
    
    print("Input shape:", hidden_state.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    test_MHA_with_cache()
```

## 四、多查询注意力（MQA)

围绕「如何减少 KV Cache 同时尽可能地保证效果」这个主题发展而来的产物。只有一组 key-value 对，由《Fast Transformer Decoding: One Write-Head is All You Need》在 2019 年提出。

与 MHA 不同的是，MQA 让所有的头之间共享同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量。使用 MQA 的模型包括 PaLM、StarCoder、Gemini 等。

```python
import torch
from torch import nn

class MultiQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 初始化 Q、K、V 投影矩阵，注意这里的 K V 比原来更小
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.head_dim)
        self.v_linear = nn.Linear(hidden_size, self.head_dim)
        
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_state, causal_mask=None, padding_mask=None):
        batch_size = hidden_state.size(0)
        
        query = self.q_linear(hidden_state)  # (batch_size, seq_len, hidden_size)
        key = self.k_linear(hidden_state)    # (batch_size, seq_len, head_dim)
        value = self.v_linear(hidden_state)  # (batch_size, seq_len, head_dim)
        
        # 分割头部，K V 矩阵也要加上一个维度
        query = self.split_head(query)  # (batch_size, num_heads, seq_len, head_dim)
        key = self.split_head(key, 1)   # (batch_size, 1, seq_len, head_dim)
        value = self.split_head(value, 1) # (batch_size, 1, seq_len, head_dim)
        
        # 计算注意力分数，自动广播，(batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if causal_mask is not None:
            attention_scores += causal_mask * -1e9  

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            attention_scores += padding_mask * -1e9
            
        attention_probs = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 对注意力输出进行拼接，(batch_size, seq_len, hidden_size)
        output = output.transpose(1, 2).view(batch_size, -1, self.head_dim * self.num_heads)
        
        output = self.o_linear(output)  # (batch_size, seq_len, hidden_size)
        
        return output

    def split_head(self, x, head_num=None):
        batch_size = x.size(0)  # 获取批量大小
        if head_num is None:
            head_num = self.num_heads  # 默认使用类中的 num_heads
        
        # 返回形状: (batch_size, head_num, seq_len, head_dim)
        return x.reshape(batch_size, -1, head_num, self.head_dim).transpose(1, 2)
```

## 五、分组查询注意力（GQA）

有人担心 MQA 对 KV Cache 的压缩太严重，于是提出了一个折中版本，出自 2023 年论文《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》。




```python
import torch
from torch import nn

class GroupQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, group_num):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.group_num = group_num  # 组的数量
        
        # 初始化 Q、K、V 投影矩阵，注意这里的 K V 做了折衷
        self.q_linear = nn.Linear(hidden_size, hidden_size)  # (hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.group_num * self.head_dim)  # (hidden_size, group_num * head_dim)
        self.v_linear = nn.Linear(hidden_size, self.group_num * self.head_dim)  # (hidden_size, group_num * head_dim)
        
        self.o_linear = nn.Linear(hidden_size, hidden_size)  # (hidden_size, hidden_size)
        
    def forward(self, hidden_state, causal_mask=None, padding_mask=None):
        batch_size = hidden_state.size(0)
        
        query = self.q_linear(hidden_state)  # (batch_size, seq_len, hidden_size)
        key = self.k_linear(hidden_state)    # (batch_size, seq_len, group_num * head_dim)
        value = self.v_linear(hidden_state)  # (batch_size, seq_len, group_num * head_dim)
        
        # 分割头部，将每个头的维度拆分出来
        query = self.split_head(query)  # (batch_size, num_heads, seq_len, head_dim)
        key = self.split_head(key, self.group_num)  # (batch_size, num_heads, seq_len, head_dim)
        value = self.split_head(value, self.group_num)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 计算注意力分数，自动广播，(batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if causal_mask is not None:
            attention_scores += causal_mask * -1e9  

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            attention_scores += padding_mask * -1e9
        
        attention_probs = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 对注意力输出进行拼接，形状: (batch_size, seq_len, hidden_size)
        output = output.transpose(1, 2).view(batch_size, -1, self.head_dim * self.num_heads)
        
        # 通过线性层将拼接后的输出变换为所需的输出维度
        output = self.o_linear(output)  # (batch_size, seq_len, hidden_size)
        
        return output

    def split_head(self, x, group_num=None):
        batch_size, seq_len = x.size()[:2]  # 获取批量大小和序列长度
        
        if group_num is None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # 将 hidden_size 分割为 group_num 和 head_dim
            x = x.view(batch_size, -1, group_num, self.head_dim).transpose(1, 2)
            # 再将其手动 expand 到相同大小
            x = x[:, :, None, :, :].expand(batch_size, group_num, self.num_heads // group_num, seq_len, self.head_dim).view(batch_size, self.num_heads, seq_len, self.head_dim)
            return x 	# 形状: (batch_size, num_heads, seq_len, head_dim)
```

