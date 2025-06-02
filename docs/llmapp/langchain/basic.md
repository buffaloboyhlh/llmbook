# LangChain 教程

## 一、 简介

![简介](https://python.langchain.com/svg/langchain_stack_112024.svg)

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
from langchain.chat_models import init_chat_model

llm = init_chat_model(model="deepseek-r1:1.5b", model_provider="ollama", base_url="http://192.168.3.5:11434/")

response = llm.invoke("你是谁？")
print(response.content)
```

!!! success "输出结果"

    您好！我是由中国的深度求索（DeepSeek）公司开发的智能助手DeepSeek-R1。如您有任何任何问题，我会尽我所能为您提供帮助。

## 二、模型使用

### 2.1  使用本地模型

Ollama 是一个本地大语言模型（LLM）运行工具，它让你能在自己的电脑上快速运行大模型（像 LLaMA、Mistral、Gemma、Phi-3、Starling
等），并且提供类似于 ChatGPT 的交互体验。

#### 1、文本模型

```python
from langchain_ollama import OllamaLLM, ChatOllama

llm = OllamaLLM(
    model="deepseek-r1:1.5b",
    base_url="http://192.168.3.5:11434/"
)

llm.invoke("你是谁？")
```

!!! success "输出结果"

    '<think>\n\n</think>\n\n您好！我是由中国的深度求索（DeepSeek）公司开发的智能助手DeepSeek-R1。如您有任何任何问题，我会尽我所能为您提供帮助。'

#### 2. 聊天模型

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

chat = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://192.168.3.5:11434/"
)

messages = [
    SystemMessage(content="你是一名翻译助手，将用户的输入翻译成英语"),
    HumanMessage(content="小明喜欢去中国旅游，有什么推荐的景点?"),
]

response = chat.invoke(messages)
print(response.content)
```

!!! success "输出结果"

    He likes traveling in China. What are your recommendations for the places to visit?

### 2.2 使用第三方模型（Hugging Face为例）

使用Hugging Face平台上的模型

```bash
pip install langchain-huggingface
```

langchain_huggingface 分成两种，一种下载模型到本地，在本地运行模型，一种使用hugging face 提供的模型api接口

#### 使用本地模型

```python
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "top_k": 50,
        "temperature": 0.1,
    },
)
llm.invoke("Hugging Face is")
```

#### 使用在线模型

```python
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
llm.invoke("Hugging Face is")
```

### 2.3 自定义聊天模型

```text
BaseLanguageModel
├── BaseLLM
└── BaseChatModel
```

+ **BaseLanguageModel**：最通用（抽象语言模型）
+ **BaseLLM**：单轮 prompt → 文本
+ **BaseChatModel**：多轮对话消息

#### 自定义多功能 Echo LLM

下面是一个多功能 Echo LLM：

+ 同步调用：回显大写
+ 异步调用：回显小写
+ 批量调用：每个句子首字母大写
+ 流式调用：逐字母输出
+ 事件流式调用：逐单词输出事件

```python
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.runnables import Event
import asyncio


class MyMultiFunctionLLM(BaseLLM):
    def _call(self, prompt: str, stop=None):
        """
        同步推理：将输入转成大写
        """
        return prompt.upper()

    async def _acall(self, prompt: str, stop=None):
        """
        异步推理：将输入转成小写
        """
        return prompt.lower()

    def _batch(self, prompts, stop=None, **kwargs):
        """
        批量推理：每句话首字母大写
        """
        return [p.capitalize() for p in prompts]

    async def _abatch(self, prompts, stop=None, **kwargs):
        """
        异步批量推理：每句话倒序输出
        """
        return [p[::-1] for p in prompts]

    def _stream(self, prompt: str, stop=None, **kwargs):
        """
        流式推理：逐字母输出
        """
        for char in prompt:
            yield GenerationChunk(text=char)

    async def _astream(self, prompt: str, stop=None, **kwargs):
        """
        异步流式推理：逐字母输出（异步版）
        """
        for char in prompt:
            await asyncio.sleep(0.05)  # 模拟慢速生成
            yield GenerationChunk(text=char)

    async def _astream_events(self, prompt: str, stop=None, **kwargs):
        """
        事件流式推理：逐单词输出事件
        """
        words = prompt.split()
        for word in words:
            await asyncio.sleep(0.1)
            yield Event(type="word", data=word)
```

### 2.4 流失输出

#### 1. 同步流

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="deepseek-r1:1.5b",
    model_provider="ollama",
    base_url="http://192.168.3.5:11434/"
)

for chunk in llm.stream("给我写一首有关于爱情的歌曲"):
    print(chunk.content)
```

#### 2. 异步流

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="deepseek-r1:1.5b",
    model_provider="ollama",
    base_url="http://192.168.3.5:11434/"
)

async for chunk in llm.astream("给我写一首校园歌曲"):
    print(chunk.content, end="|", flush=True)
```

#### 3. 异步事件流

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="deepseek-r1:1.5b",
    model_provider="ollama",
    base_url="http://192.168.3.5:11434/"
)

async  for event in llm.astream_events("你可以为我做什么？"):
    print(event)
```

### 2.5 工具调用

```python
from langchain.chat_models import init_chat_model


def add(a: int, b: int) -> int:
    return a + b


def mul(a: int, b: int) -> int:
    return a * b


llm = init_chat_model(
    model="deepseek-r1:1.5b",
    model_provider="ollama",
    base_url="http://192.168.3.5:11434/"
)

llm_with_tools = llm.bind_tools(tools=[add, mul])

llm_with_tools.invoke("What is 3 * 12?")
```

!!! failure "输出结果"

    ResponseError: registry.ollama.ai/library/deepseek-r1:1.5b does not support tools (status code: 400)

## 三、提示（Prompts）

### 3.1 消息（messages）

`langchain_core.messages` 提供了四种常用的消息类型，分别是：

| 消息类型            | 用途                 | 说明                                          |
|-----------------|--------------------|---------------------------------------------|
| `SystemMessage` | 系统消息 / 上下文约束       | 通常用于给模型设置全局角色、行为说明，类似 OpenAI `system` role  |
| `HumanMessage`  | 用户输入               | 用户发送的消息，类似 OpenAI `user` role               |
| `AIMessage`     | AI 的回复             | 模型的回答，类似 OpenAI `assistant` role            |
| `ToolMessage`   | 工具返回的结果（供模型后续对话使用） | 当工具被调用后返回结果，作为工具的「后续上下文消息」传给模型（类似工具调用的中转消息） |

---

#### ToolMessage

**用途**：当模型调用工具（如搜索、数据库）后，工具返回的结果需要用 ToolMessage 传递给模型，让模型基于工具结果进行回答。

```python
from langchain_core.messages import ToolMessage

ToolMessage(
    tool_call_id="search_tool_123",
    content="搜索到的结果是：Python 的最新版本是 3.12"
)
```

!!! info

    + tool_call_id：唯一标识工具调用，告诉模型这是哪个工具的返回结果
    +  content：工具的返回内容

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

llm = init_chat_model(
    model="deepseek-r1:1.5b",
    model_provider="ollama",
    base_url="http://192.168.3.5:11434/"
)

messages = [
    SystemMessage(content="你是中国的旅游专家，善于给出旅游方案"),
    HumanMessage(content="给我一个北京到海南的旅游方案，要包含主要的旅游景点")
]

llm.invoke(messages).content
```

### 3.2 提示模版

`langchain.prompts` 中常用的核心类有：

| 类 / 方法                        | 作用                                     |
|-------------------------------|----------------------------------------|
| `PromptTemplate`              | 最常用的提示模板类，支持动态变量替换                     |
| `ChatPromptTemplate`          | 面向多轮对话消息的提示模板（生成对话格式）                  |
| `SystemMessagePromptTemplate` | 用于生成 System 消息（system role）            |
| `HumanMessagePromptTemplate`  | 用于生成 Human 消息（user role）               |
| `AIMessagePromptTemplate`     | 用于生成 AI 消息（assistant role）             |
| `MessagesPlaceholder`         | 在多轮对话中插入「历史上下文」占位符，常用于 memory / RAG 场景 |

#### PromptTemplate

最常用的提示模板，适用于单次 Prompt 或「补全模型」的调用。
它支持使用花括号 {} 来标识变量，并在调用时进行替换。

```python
from langchain.prompts import PromptTemplate

template = "请用中文回答以下问题：{question}"
prompt = PromptTemplate.from_template(template)

final_prompt = prompt.format(question="LangChain 是什么？")
print(final_prompt)
# 输出: 请用中文回答以下问题：LangChain 是什么？
```

#### ChatPromptTemplate

用于和「聊天模型（ChatModel）」配合，生成符合对话消息格式的 Prompt。
通常包含多个「消息模板（MessagePromptTemplate）」。

```python
from langchain.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    ("user", "你好，帮我用中文总结一下：{text}")
])

final_prompt = chat_prompt.format_messages(text="LangChain 是一个用于构建 LLM 应用的框架。")
# 返回一个消息列表，包含 system、user 角色消息
for msg in final_prompt:
    print(msg)
```

!!! success "输出结果"

    content='你是一个乐于助人的助手。' additional_kwargs={} response_metadata={}
    content='你好，帮我用中文总结一下：LangChain 是一个用于构建 LLM 应用的框架。' additional_kwargs={} response_metadata={}

#### SystemMessagePromptTemplate 、 HumanMessagePromptTemplate 、 AIMessagePromptTemplate

它们是「消息模板」的具体实现，专门用于生成对应角色的消息，可与 ChatPromptTemplate 组合使用。

```python
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

system_msg = SystemMessagePromptTemplate.from_template("你是一个专业的翻译家。")
human_msg = HumanMessagePromptTemplate.from_template("请翻译：{text}")

chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
final_prompt = chat_prompt.format_messages(text="Hello world!")

for msg in final_prompt:
    print(msg)
```

#### MessagesPlaceholder

在多轮对话（如有 Memory）或 RAG 中，保留历史上下文 的占位符。

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个记忆力很强的助理。"),
    MessagesPlaceholder(variable_name="chat_history"),  # 占位符
    ("user", "{input}")
])
```

然后在调用时，将多轮上下文传给 chat_history：

```python
final_prompt = chat_prompt.format_messages(
    chat_history=[
        HumanMessage(content="你好！"),
        AIMessage(content="你好，有什么可以帮你？")
    ],
    input="请帮我写一个 Python 脚本"
)
```

#### FewShotPromptTemplate

在提示工程中，Few-Shot 提示（少量示例提示） 意味着在提示中提供若干个示例，帮助大模型「模仿示例」更好地生成目标答案。
FewShotPromptTemplate 就是为此专门准备的模板类。

```python
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# 1️⃣ 定义示例
examples = [
    {"input": "2 + 2", "output": "4"},
    {"input": "3 + 5", "output": "8"}
]

# 2️⃣ 定义示例的模板
example_prompt = PromptTemplate.from_template(
    "输入: {input}\n输出: {output}"
)

# 3️⃣ 创建 FewShotPromptTemplate
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="下面是一些数学题的示例：",
    suffix="请回答: {input}",
    input_variables=["input"]
)

# 4️⃣ 渲染提示
final_prompt = prompt.format(input="7 + 9")
print(final_prompt)
```
!!! success "输出结果"
    
    下面是一些数学题的示例：

        输入: 2 + 2
        输出: 4

        输入: 3 + 5
        输出: 8

        请回答: 7 + 9
    

## 四、输出解析器

在使用大模型（LLM / Chat 模型）时，模型输出通常是「自然语言文本」。
然而，在真实场景中，我们往往希望模型返回结构化数据（如 JSON / 列表 / 具体字段等），便于后续处理。

### 4.1  langchain.output_parsers 核心内容概览

| 类 / 方法                             | 作用                                                         |
|-------------------------------------|------------------------------------------------------------|
| `StrOutputParser`                   | 直接将输出文本原样返回为字符串                                    |
| `CommaSeparatedListOutputParser`    | 解析模型输出为「逗号分隔的字符串列表」                               |
| `ListOutputParser`                  | 将模型输出的多行文本，解析为列表                                   |
| `JsonOutputParser`                  | 解析模型输出的 JSON 字符串，返回字典 / 列表                          |
| `PydanticOutputParser`              | 将模型输出解析为符合 Pydantic 模型的结构化数据，支持字段校验             |
| `OutputFixingParser`                | 当输出格式不符合预期时，自动请求 LLM 进行格式修正（基于 Retry）            |

### 4.2 示例

#### 1️⃣ StrOutputParser

最简单的解析器，**直接返回模型输出的原始文本**。
适合对模型原始回复没有结构化需求的场景。
    
```python
from langchain.output_parsers import StrOutputParser
parser = StrOutputParser()

result = parser.parse("Hello, this is your answer.")
print(result)
# 输出: "Hello, this is your answer."
```


#### 2️⃣ CommaSeparatedListOutputParser

将模型输出的「逗号分隔文本」解析成列表。
适合场景：让模型返回「A, B, C」格式的答案。

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
parser = CommaSeparatedListOutputParser()

output = "苹果, 香蕉, 橙子"
result = parser.parse(output)
print(result)
# 输出: ["苹果", "香蕉", "橙子"]
```

#### 3️⃣ ListOutputParser

将模型输出的「多行文本」解析成列表。
适合场景：让模型逐行列出内容（每行一个项目）。

```python
from langchain.output_parsers import ListOutputParser
parser = ListOutputParser()

output = "第一项\n第二项\n第三项"
result = parser.parse(output)
print(result)
# 输出: ["第一项", "第二项", "第三项"]
```

#### 4️⃣ JsonOutputParser

当你希望模型返回 JSON 格式时，使用它来解析 JSON 并返回字典 / 列表。

```python
from langchain.output_parsers import JsonOutputParser
parser = JsonOutputParser()

output = '{"name": "Tom", "age": 20}'
result = parser.parse(output)
print(result)
# 输出: {'name': 'Tom', 'age': 20}
```

####  5️⃣ PydanticOutputParser

⚡️ 高级解析器：
将模型输出直接解析为Pydantic 模型对象，支持严格字段校验。
适合场景：输出有固定字段（如 name, age）。

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

parser = PydanticOutputParser(pydantic_object=Person)

output = '{"name": "Alice", "age": 30}'
result = parser.parse(output)
print(result)
# 输出: Person(name='Alice', age=30)
```

#### 6️⃣ OutputFixingParser

如果你用 PydanticOutputParser 或 JsonOutputParser，有时候模型返回的格式并不完全正确（比如缺少字段）。
OutputFixingParser 会自动让模型重新格式化输出，直到符合预期。

```python
from langchain.output_parsers import OutputFixingParser

parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=Person), llm=llm)
# llm: ChatOpenAI 等大模型对象
```

## 五、记忆（Memory）

在 多轮对话 或者 Agent 场景中，大模型往往需要记住用户之前说过什么，才能更自然、连贯地回答。
LangChain 的 Memory，就是专门用来保存上下文对话记录，并在每次调用时动态补充上下文，让模型具备「记忆力」。


| 类 / 类型                          | 说明                                                        |
|---------------------------------|-----------------------------------------------------------|
| `ConversationBufferMemory`      | 最简单的「上下文缓冲」内存，按顺序保存所有消息                          |
| `ConversationSummaryMemory`     | 对历史对话做「总结」保存（避免上下文过长）                           |
| `ConversationBufferWindowMemory`| 只保存最近 N 轮对话，节省上下文长度                                  |
| `ConversationKGMemory`          | 使用「知识图谱」方式存储记忆，更结构化                                   |
| `VectorStoreRetrieverMemory`    | 将记忆嵌入向量数据库，检索相似上下文（更先进）                             |
    

```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 定义 LLM
llm = ChatOpenAI(model="gpt-4o")

# 定义记忆
memory = ConversationBufferMemory()

# 创建对话链，自动携带记忆
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

# 多轮对话
print(conversation.invoke({"input": "你好！"}))
print(conversation.invoke({"input": "你叫什么名字？"}))
print(conversation.invoke({"input": "今天天气怎么样？"}))
```
模型在每次生成时，会自动拼接上下文历史（Memory 中的消息），回答更自然！

----

#### 🔶 Memory 内部结构

所有 Memory 都是基于 `BaseMemory` 抽象类，核心方法有：

| 方法                              | 说明                                                   |
|---------------------------------|------------------------------------------------------|
| `load_memory_variables(inputs)` | 加载记忆变量，返回一个 dict，通常是历史对话文本（`history`）      |
| `save_context(inputs, outputs)` | 在模型调用后，保存「用户输入 & 模型输出」到 Memory            |
| `clear()`                        | 清空记忆                                                 |
    
    
## 六、工具（Tools）

### 工具的核心抽象：BaseTool

langchain_core.tools 中的核心基类是：

```python
from langchain_core.tools import BaseTool
```

🔹 BaseTool 定义了「工具」的标准接口：
+ name: 工具的名称（模型需要用它来调用）
+ description: 工具的描述（帮助模型理解它能做什么）
+ args_schema: 工具参数的 Schema（参数校验，通常用 Pydantic BaseModel）
+ invoke() / run(): 工具真正被调用时的逻辑

### 定义一个工具示例

下面是一个简单示例，展示如何基于 BaseTool 自定义一个工具：

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# 定义参数 Schema
class CalculatorArgs(BaseModel):
    a: int = Field(..., description="第一个数字")
    b: int = Field(..., description="第二个数字")

# 自定义工具
class AddTool(BaseTool):
    name = "add_numbers"
    description = "计算两个数字之和"
    args_schema = CalculatorArgs

    def _run(self, a: int, b: int) -> int:
        return a + b

    async def _arun(self, a: int, b: int) -> int:
        # 这里也可以写异步逻辑
        return a + b
```

### 内置工具：tool

除了手动继承 BaseTool，LangChain 也提供了快速注册工具的装饰器 @tool，让定义更简单！

```python
from langchain_core.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个数字之和"""
    return a + b
```


    
    
    
    
    

    
    

    
    

    
    
    
    
    
    
    

    
    
    
    

