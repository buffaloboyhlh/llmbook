# LangChain 教程

## 一、 基础概念和环境设置

LangChain 建立在几个核心概念之上：

+ 模型（Models）：与各种LLM（如OpenAI、Anthropic、本地模型等）进行交互的接口
+ 提示（Prompts）：构建和管理发送给模型的输入
+ 输出解析器（Output Parsers）：将模型输出转换为结构化格式
+ 链（Chains）：将多个组件连接成一个工作流
+ 记忆（Memory）：存储和检索对话历史
+ 工具（Tools）：使模型能够与外部系统交互
+ 代理（Agents）：让模型决定使用哪些工具以及如何使用

首先，让我们设置环境并安装 LangChain ：

```bash
pip install langchain

pip install langchain-openai

pip install langchain-community

pip install langgraph

pip install "langserve[all]"

pip install langchain-cli

pip install langsmith

pip install langchain-huggingface
```

使用示例：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://192.168.3.5:45299/v1",  # Ollama API 兼容 v1 接口
    api_key="ollama-not-needed",  # Ollama 本地通常不需要 key，可随便写
    model="deepseek-r1:1.5b",
)

response = llm.invoke("你是谁？")
print(response)
```

## 二、提示词工程

### 2.1 消息（Messages）

+ **作用**：提供不同类型的消息结构，用于构建聊天历史

+ **常用消息类型：**
    + SystemMessage：设置助手的行为指南
    + HumanMessage：表示用户的输入
    + AIMessage：表示AI的回复
+ **参数说明：**
    + content：消息的文本内容

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

llm = ChatOpenAI(
    model="deepseek-r1:1.5b",
    api_key="random-api-key",
    base_url="http://192.168.3.5:45299/v1"
)

messages = [
    SystemMessage(content="你是一位有用的AI助手。"),
    HumanMessage(content="我想了解人工智能的基础知识。"),
    AIMessage(content="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"),HumanMessage(content="什么事机器学习？")
]

response = llm.invoke(messages)
print(response.content)
```

### 2.2 提示模版

#### PromptTemplate 模块

+ **作用：** 创建和格式化**文本**提示模板
+ **常用方法：**
    + from_template()：从模板字符串创建提示模板
    + format()：格式化提示，将值替换到占位符
+ **参数说明：**
    + 模板字符串中使用 {**变量名**} 作为占位符

```python
from langchain import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "请给我提供关于{主题}的详细信息。"  # 定义模板文本，使用{变量名}作为占位符
)

prompt = prompt_template.format(主题="深度学习")
print(prompt) # 请给我提供关于机器学习的详细信息。

respone = llm.invoke(prompt)
print(respone.content)
```

#### ChatPromptTemplate 模块

+ **作用：** 创建和格式化**聊天**提示模板
+ **常用方法：**
    - from_messages()：从消息列表创建聊天提示模板
    - format_messages()：格式化聊天提示，生成消息列表
+ **参数说明：**
    - 接受 SystemMessage、HumanMessagePromptTemplate 等消息类型

```python
from langchain.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate

teaching_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""你是一位精通{主题}的教学专家。你的目标是以简单易懂、循序渐进的方式教授{概念}。请记住以下教学原则：
1. 从基础概念开始，逐步引入复杂内容
2. 使用具体的例子和比喻来解释抽象概念
3. 积极鼓励学习者，肯定他们的每一步进步
4. 提供实践机会和互动式学习
5. 适当检查理解程度，及时调整教学内容"""),
    HumanMessagePromptTemplate.from_template("请教我{概念}的基本原理。我是一个{学习者水平}的学习者，特别感兴趣的方面是{兴趣点}。如果可能，请包含一些简单的练习让我实践。")
])

messages = teaching_template.format_messages(
    主题="人工智能",
    概念="深度学习",
    学习者水平="初学者",
    兴趣点="计算机视觉应用"
)

print("Messages".center(80, "="),'\n',messages)

response = llm.invoke(messages)
print("\n","输出结果".center(80, "="),'\n')
print(response.content)
```

#### FewShotPromptTemplate 

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# 示例数据
examples = [
    {"name": "小明", "age": 18, "destination": "青春活力的城市，如上海、深圳"},
    {"name": "李奶奶", "age": 70, "destination": "舒适休闲的景区，如杭州西湖、成都青城山"}
]

# 示例模板
example_prompt = PromptTemplate(
    input_variables=["name", "age", "destination"],
    template="姓名: {name}\n年龄: {age}\n推荐目的地: {destination}"
)

# 创建FewShot模板
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="姓名: {name}\n年龄: {age}\n推荐目的地:",
    input_variables=["name", "age"]
)

# 格式化模板
formatted_prompt = few_shot_prompt.format(name="小陈", age=22)
print(formatted_prompt)
```

#### 自定义提示词模板

```python
from langchain.prompts import BasePromptTemplate

class CustomPromptTemplate(BasePromptTemplate):
    input_variables: list[str]
    
    def format(self, **kwargs) -> str:
        # 自定义格式化逻辑
        return f"自定义前缀: {', '.join([f'{k}={v}' for k, v in kwargs.items()])}"

# 使用自定义模板
prompt = CustomPromptTemplate(input_variables=["a", "b"])
print(prompt.format(a=1, b=2))
# 输出: 自定义前缀: a=1, b=2
```



### 三、输出解析器

#### 3.1 核心概念

##### 1. 输出解析器（OutputParser）

+ **作用：** 将 LLM 生成的文本转换为结构化数据。
+ **核心方法：**
    - parse(text: str)：将文本解析为目标格式。
    - get_format_instructions()：返回格式说明，指导 LLM 生成符合要求的文本。

##### 2. 常见解析器类型

+ **结构化解析器：** 如 StructuredOutputParser、ResponseSchema。
+ **JSON 解析器**：如 JsonOutputParser。
+ **自定义解析器**：继承 BaseOutputParser 实现自定义逻辑。

#### 3.2 StrOutputParser

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 创建模型
llm = ChatOpenAI(
    base_url="http://192.168.3.5:45299/v1",  # Ollama API 兼容 v1 接口
    api_key="ollama-not-needed",  # Ollama 本地通常不需要 key，可随便写
    model="deepseek-r1:1.5b",
)
# 提示词
prompt = PromptTemplate.from_template(
    "给我5个关于{主题}的有趣事实。"  # 定义模板文本
)

# 创建输出解析器
output_parser = StrOutputParser()  # 将模型输出解析为字符串

chain = prompt | llm | output_parser

# 调用链
response = chain.invoke({"主题": "量子物理学"})  # 传入参数字典
print(response)  # 打印链的输出
```

#### 3.3 JsonOutputParser

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

# 创建模型
llm = ChatOpenAI(
    base_url="http://192.168.3.5:45299/v1",  # Ollama API 兼容 v1 接口
    api_key="ollama-not-needed",  # Ollama 本地通常不需要 key，可随便写
    model="deepseek-r1:1.5b",
)
# 提示词
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="请把以下信息输出为 JSON 格式"),
    HumanMessage(content="名字：张三，年龄：25，爱好：读书和游泳")
])

# 创建输出解析器
output_parser = JsonOutputParser()

chain = prompt | llm | output_parser

# 调用链
response = chain.invoke({})  # 传入参数字典
print(response)  # 打印链的输出
```

**输出**

```text
{'name': '张三', 'age': 25, '爱好': ['读书', '游泳']}
```

#### 3.4 自定义输出解析器

自定义解析器需要实现以下核心部分：

1. **继承 BaseOutputParser**：所有解析器必须继承该基类。
2. **实现 parse 方法**：定义文本到目标格式的转换逻辑。
3. **实现 get_format_instructions 方法**：返回格式说明，指导 LLM 生成符合要求的文本（可选但推荐）。

**简单示例：解析逗号分隔的列表**
```python
from langchain_core.output_parsers import BaseOutputParser

class CommaSeparatedListParser(BaseOutputParser):
    """将逗号分隔的文本转换为列表"""
    
    def parse(self, text: str) -> list[str]:
        """解析文本为列表"""
        return [item.strip() for item in text.split(",")]
    
    def get_format_instructions(self) -> str:
        """返回格式说明"""
        return "请用逗号分隔多个项目（例如：苹果,香蕉,橙子）"

# 使用示例
parser = CommaSeparatedListParser()
result = parser.parse("足球,篮球,乒乓球")
print(result)  # 输出: ['足球', '篮球', '乒乓球']
```

**带错误处理的解析器**

当解析失败时，可以抛出 OutputParserException：

```python
from langchain_core.output_parsers import BaseOutputParser, OutputParserException

class NumberListParser(BaseOutputParser[list[int]]):
    """解析由空格分隔的数字列表"""
    
    def parse(self, text: str) -> list[int]:
        try:
            return [int(num.strip()) for num in text.split()]
        except ValueError as e:
            raise OutputParserException(f"无法将文本解析为数字列表: {text}") from e

# 使用示例
parser = NumberListParser()
try:
    result = parser.parse("1 2 3a 4")  # 触发异常
except OutputParserException as e:
    print(f"解析错误: {e}")
```



