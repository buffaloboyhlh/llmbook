## 一、张量

### 1.1 构造Tensor

**dtype类型**：dtype=torch.long,torch.float,torch.double

常见的构造Tensor的函数：

| 函数                    | 功能 | 示例 |
|-----------------------|--|----|
| `torch.Tensor(sizes)` | 基础构造函数 | `torch.Tensor(4,3)` |
| `torch.tensor()`      | 从数据构造 Tensor | `torch.tensor([1, 2, 3])` |
| `torch.zeros()`       | 构造全 0 的 Tensor | `torch.zeros(2, 3)` |
| `torch.ones()`        | 构造全 1 的 Tensor | `torch.ones(3, 2)` |
| `torch.full()`        | 构造填充值为指定值的 Tensor | `torch.full((2, 2), 7)` |
| `torch.arange()`      | 构造从起始到终止的整数序列 Tensor | `torch.arange(0, 10, 2)` |
| `torch.linspace()`    | 构造线性间隔的 Tensor | `torch.linspace(0, 1, 5)` |
| `torch.logspace()`    | 构造对数间隔的 Tensor | `torch.logspace(1, 3, steps=3)` |
| `torch.eye()`         | 构造单位矩阵 | `torch.eye(3)` |
| `torch.rand()`        | 构造 0~1 区间的均匀分布随机数 | `torch.rand(2, 2)` |
| `torch.randn()`       | 构造标准正态分布的随机数 | `torch.randn(2, 2)` |
| `torch.randint()`     | 构造整数范围内的随机 Tensor | `torch.randint(0, 10, (2, 3))` |
| `torch.empty()`       | 构造未初始化的 Tensor | `torch.empty(2, 2)` |
| `torch.from_numpy()`  | 从 NumPy 数组构造 Tensor | `torch.from_numpy(np.array([1, 2, 3]))` |
| `torch.as_tensor()`   | 将数据转换为 Tensor，尽量避免复制 | `torch.as_tensor([1, 2, 3])` |
| `torch.clone()`       | 克隆一个 Tensor（复制） | `a.clone()` |
| `torch.empty_like()`  | 创建和给定 Tensor 相同形状但未初始化的 Tensor | `torch.empty_like(a)` |
| `torch.zeros_like()`  | 创建和给定 Tensor 相同形状的全 0 Tensor | `torch.zeros_like(a)` |
| `torch.ones_like()`   | 创建和给定 Tensor 相同形状的全 1 Tensor | `torch.ones_like(a)` |


### 1.2 张量的属性

```python
x.size() # 早期 PyTorch 版本使用  
x.shape # 推荐，更现代语法
```

### 1.3 常用方法

#### tensor.view(*shape)

改变一个 tensor 的大小或者形状，可以使用 torch.view：view() 返回的新tensor与源tensor共享内存(其实是同一个tensor)，也即更改其中的一个，另 外一个也会跟着改变。

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
```

!!! warning 
    
    tensor 必须是 contiguous 的，否则需先调用 .contiguous()
    ```python
    x = torch.randn(2, 3)
    y = x.transpose(0, 1)  # 非连续内存
    z = y.contiguous().view(-1)  # 先 contiguous
    ```

#### 转置transpose/维度交换permute

在PyTorch中，`transpose`和`permute`都是用于调整张量维度顺序的函数，但它们在功能和使用场景上有显著区别。

##### 1. 功能区别
- **`transpose(dim0, dim1)`**  
  交换张量的两个指定维度（`dim0`和`dim1`），其他维度保持不变。  
  **示例**：对于一个形状为 `(2, 3, 4)` 的三维张量，执行 `transpose(0, 2)` 后，维度变为 `(4, 3, 2)`。

- **`permute(*dims)`**  
  重新排列所有维度的顺序，参数是一个包含所有维度索引的元组。  
  **示例**：将形状为 `(2, 3, 4)` 的张量调整为 `(4, 2, 3)`，需使用 `permute(2, 0, 1)`。

##### 2. 参数与灵活性
- **`transpose`** 接受两个参数，表示要交换的维度索引，仅适用于两个维度的交换。
- **`permute`** 接受一个完整的维度顺序元组，可灵活调整任意多个维度的顺序，但必须指定所有维度。

##### 3. 示例代码
###### 使用 `transpose`
```python
import torch

# 二维张量（矩阵转置）
x = torch.tensor([[1, 2], [3, 4]])
print(x.transpose(0, 1))  # 输出：tensor([[1, 3], [2, 4]])

# 三维张量交换维度
x = torch.randn(2, 3, 4)
y = x.transpose(0, 2)     # 形状变为 (4, 3, 2)
```

###### 使用 `permute`
```python
# 三维张量重新排列维度
x = torch.randn(2, 3, 4)
z = x.permute(2, 0, 1)    # 形状变为 (4, 2, 3)

# 四维张量调整通道位置（NCHW → NHWC）
batch = torch.randn(4, 5, 6, 7)       # 形状 (4, 5, 6, 7)
batch_permuted = batch.permute(0, 2, 3, 1)  # 形状变为 (4, 6, 7, 5)
```

##### 4. 注意事项
- **视图（View）机制**：两者均返回原始张量的视图，共享存储空间，修改一个会影响另一个。
- **连续性（Contiguity）**：调整维度可能导致张量内存不连续，后续操作（如 `view()`）需先调用 `contiguous()`。
  ```python
  y = x.transpose(0, 2).contiguous()  # 确保连续
  ```

##### 5. 应用场景
- **`transpose`**：适用于仅需交换两个维度的简单操作（如矩阵转置）。
- **`permute`**：适用于复杂的多维度重排（如调整图像数据的通道顺序）。

##### 总结
| 特性               | `transpose`                   | `permute`                     |
|--------------------|-------------------------------|-------------------------------|
| **功能**           | 交换两个维度                  | 任意维度重排                  |
| **参数**           | `dim0`, `dim1`                | 完整维度顺序的元组            |
| **灵活性**         | 低（仅两维度）                | 高（多维度）                  |
| **典型应用场景**   | 矩阵转置、简单维度交换        | 复杂维度调整（如通道重排）    |


#### 扩展expand

```python
#返回当前张量在某个维度为1扩展为更大的张量
x = torch.Tensor([[1], [2], [3]])#shape=[3,1]
t=x.expand(3, 4)
print(t)
'''
tensor([[1., 1., 1., 1.],
[2., 2., 2., 2.],
[3., 3., 3., 3.]])
'''
```

#### 重复repeat

```python
#沿着特定的维度重复这个张量
x=torch.Tensor([[1,2,3]])
t=x.repeat(3, 2)
print(t)
'''
tensor([[1., 2., 3., 1., 2., 3.],
[1., 2., 3., 1., 2., 3.],
[1., 2., 3., 1., 2., 3.]])
'''
```

#### 拼接cat

```python
x = torch.randn(2,3,6)
y = torch.randn(2,4,6)
c=torch.cat((x,y),1)
#c=(2*7*6)
```

#### 堆叠stack

```python
"""
而stack则会增加新的维度。
如对两个1*2维的tensor在第0个维度上stack，则会变为2*1*2的tensor；在第1个维度上stack，则会变为1*2*2的tensor。
"""
a = torch.rand((1, 2))
b = torch.rand((1, 2))
c = torch.stack((a, b), 0)
```

#### 压缩和扩展维度：改变tensor中只有1个维度的tensor

```python
x = torch.Tensor(1,3)
y=torch.squeeze(x, 0) # squeeze(tensor, dim)：移除指定维度中 大小为 1 的维度。
y=torch.unsqueeze(y, 1) # unsqueeze(tensor, dim)：在指定位置插入 大小为 1 的新维度。
```

#### 矩阵乘法

做矩阵a*b以下操作一样。

如果a是一个n×m张量，b是一个 m×p 张量，将会输出一个 n×p 张量c。

```python
a = torch.rand(2,4) 
b = torch.rand(4,3) 
c = a.mm(b) 
c = torch.mm(a, b) 
c = torch.matmul(a, b) 
c = a @ b
```

## 二、pytorch自动求导机制及使用

### 2.1 pytorch自动求导机制

PyTorch 中autograd包为张量上的所有操作提供了自动求导机制。torch.Tensor是这个包的核心类。如果设置它的属性.requires_grad 为 True，那么它将会追踪对于该张量的所有操作。当完成计算后可以通过调用.backward()，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到.grad属性。

```python
x = torch.arange(0, 100, 0.01,dtype=torch.double,requires_grad=True)
y = sum(10 * x + 5 )
y.backward()
print(x.grad)
#tensor([10,10,....10],dtype=torch.float64)
```

## 三、GPU配置

### 3.1 GPU的设置

```python
# 方法一：使用os.environ，这种情况如果使用GPU不需要设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 方法二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### 3.2 数据/模型拷贝到GPU上/拷贝到CPU上

