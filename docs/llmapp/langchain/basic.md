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

Ollama 是一个本地大语言模型（LLM）运行工具，它让你能在自己的电脑上快速运行大模型（像 LLaMA、Mistral、Gemma、Phi-3、Starling 等），并且提供类似于 ChatGPT 的交互体验。

#### 1、文本模型

```python
from langchain_ollama import OllamaLLM,ChatOllama

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

response =  chat.invoke(messages)
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


    
    
    



