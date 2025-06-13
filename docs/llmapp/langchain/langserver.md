# LangServe 教程

## 1️⃣ 什么是 LangServer（LangServe）？

在 LangChain 中，LangServe（有时也叫 LangServer）是一个非常强大的模块：
它能把 LangChain Chain / Agent / Runnable 通过一个 API 服务（RESTful API / WebSocket） 的形式发布出来。

这样，你可以很方便地：

✅ 把大模型工作流（RAG、Agent、对话链等）部署成 HTTP 服务

✅ 与外部系统（比如前端、APP、其他后端）集成

✅ 快速实现「模型 API」功能

##  2️⃣ LangServe 的核心功能

| 功能                     | 说明                                                                          |
|------------------------|-----------------------------------------------------------------------------|
| 🌐 自动生成 RESTful API | 自动将 Runnable（Chain、Agent）暴露为 HTTP 接口                                      |
| ⚡️ 支持流式输出         | WebSocket/Server-Sent Events（SSE） 流式接口，适配长文本/多轮对话                     |
| 📄 自动生成 OpenAPI 文档 | 生成交互式的 API 文档（可直接测试接口）                                               |
| 🔧 快速部署             | 只需要几行代码，就能把模型工作流打包成服务                                               |


## 3️⃣ 安装 LangServe

```bash
pip install "langserve[all]"
```

##  4️⃣ 一个最简单的 LangServe 示例

下面以一个简单的对话链为例（ConversationChain），演示如何用 LangServe 把它发布成 API：

```python
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 定义模型和链
llm = ChatOpenAI(model="gpt-4o")
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)

# 创建 FastAPI 应用
app = FastAPI()

# 将 chain 发布为 API
add_routes(app, chain, path="/chat")

# 运行服务
# uvicorn main:app --reload --port 8000
```
!!! tip "运行后，就能访问到："

    +	http://localhost:8000/chat/invoke（一次性调用）
	+	http://localhost:8000/chat/stream（流式输出）
	+	以及自动生成的交互式文档（http://localhost:8000/docs）
    

## 5️⃣ LangServe 接口概览

当你使用 `add_routes` 发布一个 `Runnable`（如 Chain / Agent），LangServe 会自动生成这些端点：

| Endpoint                        | 说明                                  |
|---------------------------------|-------------------------------------|
| `/invoke`                       | 直接一次性调用，返回结果                       |
| `/batch`                        | 批量调用                               |
| `/stream`                       | 流式输出（SSE）                          |
| `/stream_events`                | 事件流输出（事件型流式输出，适合更复杂的可视化）     |
| `/metadata`                     | 返回 Chain / Agent 元数据（如输入/输出类型等）      |
    
## 6️⃣ LangServe 的强大特性

✅ 零配置：只要是 Runnable（包括 LLM、Agent、Chain），都能直接发布

✅ 自动化文档：Swagger UI / ReDoc，方便测试和对接

✅ 多种输入输出模式：兼容对话式、检索式、Agent 工具式场景

✅ 灵活部署：支持 Docker、Kubernetes 等多种方式


## 7️⃣ LangServe 高级用法

+ 发布多个 Chain / Agent：多次 add_routes
+ 发布自定义工具链（Agent + Memory + Tool）
+ 自定义输入输出 Schema：通过 pydantic 模型，严格约束输入输出
+ 结合向量数据库：打造真正的 RAG API
+ 安全/认证：结合 FastAPI 的中间件（如认证、限流等）