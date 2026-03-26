# LangChain RAG 智扫通机器人智能客服

## 项目概述

这是一个基于 LangChain + ReAct 模式的智能客服机器人项目，使用 RAG（检索增强生成）技术从知识库中检索相关信息，结合大语言模型生成回答。项目主要服务于扫地机器人产品的智能客服场景。

---

## 项目架构流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           用户交互层 (Streamlit UI)                          │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────────────┐  │
│  │  st.title() │    │  st.chat_input() │    │  st.chat_message()        │  │
│  │   标题展示   │    │    用户输入框     │    │   消息展示                 │  │
│  └─────────────┘    └────────┬─────────┘    └────────────────────────────┘  │
└────────────────────────────────│─────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ReactAgent 处理层                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    ReactAgent.execute_stream()                         │  │
│  │                         LangChain ReAct Agent                           │  │
│  │                    (create_agent + middleware)                          │  │
│  └────────────────────────────────┬────────────────────────────────────────┘  │
└──────────────────────────────────│────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
         ┌──────────────────┐           ┌─────────────────────┐
         │    思考阶段       │           │     行动阶段          │
         │  (Think/Reason)  │           │   (Act/Execute)      │
         │                  │           │                     │
         │  1. 分析用户问题   │           │  调用 Tool 执行任务   │
         │  2. 判断是否需要   │──────────▶│  (最多5次工具调用)    │
         │     调用工具      │           │                     │
         │  3. 决定使用哪个   │           │                     │
         │     工具组合      │           │                     │
         └──────────────────┘           └─────────────────────┘
                                                 │
                             ┌───────────────────┼───────────────────┐
                             ▼                   ▼                   ▼
                   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
                   │ rag_summarize│   │ get_weather  │   │fetch_external_data│
                   │   (RAG检索)   │   │ (天气查询)   │   │ (获取用户记录)     │
                   └───────┬──────┘   └──────────────┘   └──────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              RAG 服务层                                        │
│                                                                                │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐   │
│  │     VectorStoreService          │    │      RagSummarizeService        │   │
│  │         (向量存储)               │    │         (RAG摘要服务)             │   │
│  │                                 │    │                                 │   │
│  │  • ChromaDB 向量数据库          │    │  prompt + model + parser        │   │
│  │  • DashScope embeddings        │───▶│                                 │   │
│  │  • top-k=3 相似文档检索         │    │  • 格式化检索结果                 │   │
│  │  • chunk_size=200              │    │  • 调用 Qwen3-max 生成答案       │   │
│  │  • RecursiveCharacterTextSplitter│   │  • StrOutputParser 输出          │   │
│  └─────────────────────────────────┘    └─────────────────────────────────┘   │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              中间件层 (Middleware)                              │
│                                                                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  monitor_tool   │    │  log_before_    │    │  report_prompt_switch   │  │
│  │ (@wrap_tool_call│    │    model        │    │   (@dynamic_prompt)    │  │
│  │                 │    │ (@before_model) │    │                         │  │
│  │ • 记录工具调用   │    │                 │    │ 根据 context["report"]  │  │
│  │   请求和结果    │    │ • 记录模型执行   │    │ 切换不同 system prompt   │  │
│  │ • 设置report标志│    │   前的状态       │    │                         │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│                                                                                │
│           Normal Query ──▶ main_prompt.txt (ReAct 聊天指令)                     │
│           Report请求 ──▶ report_prompt.txt (报告生成指令)                      │
└───────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              模型层 (Model Factory)                            │
│                                                                                │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐   │
│  │        ChatTongyi               │    │       DashScopeEmbeddings        │   │
│  │       (Qwen3-max)              │    │      (text-embedding-v4)          │   │
│  │                                 │    │                                 │   │
│  │ • 通义千问大模型                │    │ • 文本向量化嵌入                  │   │
│  │ • 流式输出响应                 │    │ • ChromaDB 向量存储              │   │
│  │ • 支持 function calling        │    │                                 │   │
│  └─────────────────────────────────┘    └─────────────────────────────────┘   │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 完整请求流程时序图

```
用户          Streamlit          ReactAgent       LangChain        Tools          RAG/Model
 │                │                  │               Agent           │               │
 │  输入问题       │                  │               │              │               │
 │───────────────▶│                  │               │              │               │
 │                │  execute_stream()│               │              │               │
 │                │─────────────────▶│               │              │               │
 │                │                  │  stream()     │              │               │
 │                │                  │──────────────▶│              │               │
 │                │                  │               │              │               │
 │                │                  │◀──────────────│              │               │
 │                │                  │  [思考]        │              │               │
 │                │                  │               │              │               │
 │                │                  │  [决定调用工具]│              │               │
 │                │                  │──────────────▶│  rag_summarize│              │
 │                │                  │               │─────────────▶│               │
 │                │                  │               │              │               │
 │                │                  │               │              │ ChromaDB检索   │
 │                │                  │               │              │◀──────────────│
 │                │                  │               │              │  返回Top-3文档 │
 │                │                  │◀──────────────│──────────────│               │
 │                │                  │               │              │               │
 │                │                  │  调用LLM生成   │              │               │
 │                │                  │──────────────────────────────────────────▶│
 │                │                  │               │              │     Qwen3-max │
 │                │                  │◀──────────────────────────────────────────│
 │                │                  │  流式返回结果   │              │               │
 │◀───────────────│◀─────────────────│───────────────│──────────────│──────────────│
 │   流式显示      │   write_stream() │               │              │               │
 │   思考中...     │                  │               │              │               │
```

---

## 工具调用流程

### 7个 Tool 一览表

| Tool名称 | 功能说明 |
|----------|----------|
| `rag_summarize` | 从向量数据库检索相关文档，生成RAG摘要 |
| `get_weather` | 获取城市天气（Mock数据） |
| `get_user_location` | 获取用户位置（随机返回：深圳/合肥/杭州） |
| `get_user_id` | 获取用户ID（随机返回：1001-1010） |
| `get_current_month` | 获取当前月份（随机返回：2025-01 至 2025-12） |
| `fetch_external_data` | 从 records.csv 获取用户指定月份的用电记录 |
| `fill_context_for_report` | 触发中间件，将 report=True 切换为报告生成模式 |

---

## 项目文件结构

```
Agent/
├── app.py                          # Streamlit Web入口
├── agent/
│   ├── react_agent.py              # ReactAgent核心类
│   └── tools/
│       ├── agent_tools.py          # 7个Tool定义
│       └── middleware.py           # 3个中间件
├── RAG/
│   ├── rag_service.py              # RAG摘要服务
│   └── vector_store.py             # ChromaDB向量存储
├── model/
│   └── factory.py                  # 模型工厂(ChatTongyi/DashScope)
├── config/
│   ├── agent.yml                   # Agent配置
│   ├── chroma.yml                  # ChromaDB配置
│   ├── prompts.yml                 # Prompt路径配置
│   └── rag.yml                     # RAG配置
├── prompts/
│   ├── main_prompt.txt             # 主System Prompt (ReAct)
│   ├── rag_summarize.txt           # RAG Prompt
│   └── report_prompt.txt           # 报告生成Prompt
├── utils/
│   ├── config_handler.py           # YAML配置加载
│   ├── file_handler.py             # 文件处理(MD5/PDF/TXT)
│   ├── logger_handler.py           # 日志
│   ├── path_tool.py                # 路径工具
│   └── prompt_loader.py            # Prompt加载
├── data/
│   ├── external/records.csv        # 外部数据(120条用户记录)
│   ├── 扫地机器人100问.pdf          # 知识库PDF
│   ├── 扫地机器人100问2.txt         # 知识库TXT
│   ├── 扫拖一体机器人100问.txt      # 知识库TXT
│   ├── 故障排除.txt                 # 故障排除指南
│   ├── 维护保养.txt                 # 维护保养指南
│   └── 选购指南.txt                 # 选购指南
├── chroma_db/                       # ChromaDB持久化存储
└── logs/                           # 日志目录
```

---

## 核心技术栈

| 层级 | 技术 |
|------|------|
| **UI** | Streamlit |
| **Agent框架** | LangChain + LangGraph (ReAct模式) |
| **LLM** | 阿里云 DashScope (Qwen3-max) |
| **向量数据库** | ChromaDB |
| **Embedding** | DashScope text-embedding-v4 |
| **配置管理** | YAML |

---

## 详细模块说明

### 1. ReactAgent

`ReactAgent` 是项目的核心类，负责：
- 初始化 LangChain ReAct Agent
- 加载系统提示词
- 注册工具和中间件
- 提供流式执行接口 `execute_stream()`

### 2. RAG 服务

**VectorStoreService**:
- 使用 ChromaDB 作为向量数据库
- DashScope text-embedding-v4 进行文本向量化
- top-k=3 检索最相关的文档
- chunk_size=200 字符进行文档分割

**RagSummarizeService**:
- 构建 prompt + model + parser 链
- 结合检索结果和原始问题生成答案
- 返回基于知识库的摘要回答

### 3. 中间件系统

| 中间件 | 类型 | 功能 |
|--------|------|------|
| `monitor_tool` | @wrap_tool_call | 记录工具调用，设置report标志 |
| `log_before_model` | @before_model | 记录模型执行前的状态 |
| `report_prompt_switch` | @dynamic_prompt | 根据context切换prompt模板 |

### 4. 配置说明

**config/rag.yml**:
- `chat_model_name`: `qwen3-max-2026-01-23`
- `embedding_model_name`: `text-embedding-v4`

**config/chroma.yml**:
- `collection_name`: `agent`
- `persist_directory`: `chroma_db`
- `k`: `3`
- `chunk_size`: `200`
- `chunk_overlap`: `20`

**config/agent.yml**:
- `external_data_path`: `data/external/records.csv`

---

## 环境依赖

```
langchain
langchain-core
langchain-community
langchain_chroma
langgraph
dashscope
streamlit
PyPDFLoader
yaml
```
