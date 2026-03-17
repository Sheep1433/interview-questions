# LangChain / LangGraph / DeepAgents 面试突击手册

> 生成日期: 2026-03-17

---

## 一、整体架构关系

```
┌─────────────────────────────────────────────────────────────┐
│                        LangChain                            │
│  (上层抽象库: LCEL, Components, Agents)                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                        LangGraph                            │
│  (底层编排引擎: 状态图, 节点, 边, 检查点)                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       DeepAgents                            │
│  (基于 LangGraph 的生产级 Agent 框架: Subagents, Skills)    │
└─────────────────────────────────────────────────────────────┘
```

**一句话概括**:
- **LangChain**: 高层抽象库，简化 LLM 应用开发
- **LangGraph**: 底层编排引擎，基于状态图构建复杂 Agent
- **DeepAgents**: 基于 LangGraph 的生产级框架，提供 Skills、Subagents 等企业级功能

---

## 二、LangChain 核心概念

### 2.1 什么是 LangChain?

LangChain 是一个用于构建 LLM 应用的 Python/JS 框架，核心目标是**简化 Prompt 工程**和**链式调用**。

### 2.2 核心组件

| 组件 | 作用 | 面试关键词 |
|------|------|------------|
| **Models** | 封装各种 LLM (OpenAI, Anthropic, Claude等) | model agnostic |
| **Prompts** | Prompt 模板管理 | PromptTemplate, FewShotPrompt |
| **Chains** | 链式调用多个组件 | LLMChain, SequentialChain |
| **Agents** | 动态决定调用哪些工具 | AgentExecutor, Tool |
| **Memory** | 对话上下文管理 | ConversationBufferMemory |
| **Indexes** | 文档检索 | VectorStore, Retriever |

### 2.3 LCEL (LangChain Expression Language)

```python
# LangChain 的函数式接口
chain = prompt | model | output_parser
```

**核心优势**:
- 统一调用接口 (Runnable)
- 支持流式输出 (stream)
- 支持异步 (async)
- 支持批处理 (batch)

### 2.4 面试常见问题

**Q: LangChain 和 LangGraph 的区别?**
> LangChain 提供高层抽象，LangGraph 提供低层状态机编排。LangChain 的高级 Agents 内部基于 LangGraph 实现。

**Q: LangChain 的 Agent 是如何工作的?**
> Agent 使用 ReAct 模式或 Tool Calling，通过 LLM 决定是否调用工具，然后执行工具并把结果喂回给 LGM，直到任务完成。

---

## 三、LangGraph 核心概念

### 3.1 什么是 LangGraph?

LangGraph 是 LangChain 的底层编排引擎，用**状态图**来构建多轮交互的 Agent。

### 3.2 核心组件

| 组件 | 作用 | 示例 |
|------|------|------|
| **State** | 整个图共享的状态字典 | `{"messages": [], "count": 0}` |
| **Node** | 图中的节点 (函数) | `def node(state): return {"x": 1}` |
| **Edge** | 节点之间的边 | `graph.add_edge("start", "end")` |
| **Reducer** | 状态合并逻辑 | `add_messages` |
| **Checkpointer** | 状态持久化 | `MemorySaver` |

### 3.3 状态图示例

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    count: int

def node_a(state):
    return {"messages": ["hello"]}

def node_b(state):
    return {"messages": ["world"]}

# 构建图
graph = StateGraph(AgentState)
graph.add_node("a", node_a)
graph.add_node("b", node_b)
graph.add_edge("__start__", "a")
graph.add_edge("a", "b")
graph.add_edge("b", END)

app = graph.compile()
```

### 3.4 核心特性

1. **循环支持**: LangGraph 支持节点之间的循环，这是 LangChain Chains 做不来的
2. **状态持久化**: 通过 Checkpointer 保存/恢复状态
3. **人机交互**: 支持 Human-in-the-loop，在关键节点暂停等待用户确认

### 3.5 面试常见问题

**Q: LangGraph 相比 LangChain Chains 的优势?**
> - 支持循环 (Loops)，适合多轮对话
> - 细粒度状态管理
> - 支持条件分支 (Conditional Edges)
> - 内置持久化

**Q: 什么是 Reducer?**
> Reducer 定义如何合并状态更新。LangGraph 默认的 `add_messages` 会把新消息追加到列表，而不是覆盖。

---

## 四、DeepAgents 核心概念

### 4.1 什么是 DeepAgents?

DeepAgents 是基于 LangGraph 构建的**生产级 Agent 框架**，提供 Subagents、Skills、Memory 等企业级功能。

### 4.2 核心组件

| 组件 | 作用 | 面试关键词 |
|------|------|------------|
| **create_agent()** | 创建主 Agent | 工厂函数 |
| **Subagents** | 子 Agent Spawn | 任务分解 |
| **Skills** | 渐进式技能加载 | Progressive Disclosure |
| **Backends** | 文件/执行环境抽象 | Sandbox, Filesystem |
| **Middleware** | 中间件扩展 | Skills, Memory, Summarization |
| **ACP** | Agent Client Protocol | 标准化通信协议 |

### 4.3 Subagents 子代理

```python
# DeepAgents 的子代理机制
agent = create_agent(...)
agent.spawn(
    label="researcher",
    task="帮我研究 AI 的最新进展",
    skills=["/skills/research/"]
)
```

**特点**:
- 独立的状态和执行上下文
- 可以有独立的 Skills 配置
- 支持流式输出
- 支持任务委派

### 4.4 Skills 渐进式披露

```python
# Skills 配置示例
middleware = SkillsMiddleware(
    backend=my_backend,
    sources=[
        "/skills/base/",      # 基础技能
        "/skills/user/",     # 用户技能
        "/skills/project/",  # 项目技能（优先级最高）
    ]
)
```

**渐进式披露流程**:
```
1. Agent 启动时 → SkillsMiddleware 加载 metadata (name, description, path)
2. 注入 System Prompt → 告诉 Agent 有哪些 Skills 可用
3. Agent 识别需要使用 Skill → 调用 read_file 读取完整 SKILL.md
4. 按 Skill 指令执行任务
```

**分层加载**: 后加载的 source 覆盖前面的 (last one wins)

### 4.5 Backends 抽象

DeepAgents 通过 Backend 抽象支持多种执行环境:

| Backend | 作用 |
|---------|------|
| **FilesystemBackend** | 本地文件系统 |
| **StateBackend** | 内存/临时状态 |
| **SandboxBackend** | 隔离沙箱执行 |
| **CompositeBackend** | 组合多个 Backend |

### 4.6 ACP (Agent Client Protocol)

ACP 是 DeepAgents 的标准化通信协议，用于:
- Agent 注册与发现
- 工具调用
- 流式响应
- 任务提交

### 4.7 面试常见问题

**Q: DeepAgents 和 LangGraph 的关系?**
> DeepAgents 底层基于 LangGraph 构建，使用 LangGraph 的状态图机制。DeepAgents 是 LangGraph 的上层应用框架。

**Q: DeepAgents 的 Skills 机制是如何实现的?**
> - 第一步: SkillsMiddleware 在 Agent 启动时批量加载 SKILL.md 的 YAML frontmatter (name, description, path)
> - 第二步: 将 Skills 列表注入 System Prompt，告诉 Agent 在何时需要使用 Skill
> - 第三步: Agent 自行通过 read_file 工具读取完整的 SKILL.md 内容
> - 这是经典的 "渐进式披露" 模式，避免一次性加载所有内容造成 Context 膨胀

**Q: DeepAgents 如何支持多轮对话和状态管理?**
> 通过 LangGraph 的 Checkpointer 实现状态持久化，支持断点续训。

**Q: DeepAgents 的 Subagent 和 LangChain 的 Agent 有什么区别?**
> DeepAgents Subagent 是独立的执行单元，有自己的状态、Skills、Backend；LangChain Agent 通常是单个图的执行逻辑。

---

## 五、三者对比总结

| 维度 | LangChain | LangGraph | DeepAgents |
|------|-----------|-----------|------------|
| **定位** | 高层抽象库 | 底层编排引擎 | 生产级框架 |
| **核心抽象** | Chain | 状态图 | Agent + Skills |
| **循环支持** | ❌ | ✅ | ✅ |
| **状态管理** | Memory | State + Reducer | State + Checkpointer |
| **适用场景** | 简单 LLM 应用 | 复杂多轮 Agent | 企业级 Agent |
| **学习曲线** | 低 | 中 | 中高 |

---

## 六、面试高频问题汇总

### 6.1 基础概念

1. **LangChain 的核心组件有哪些?**
2. **LangChain 和 LangGraph 的区别是什么?**
3. **LangGraph 的状态管理机制是什么?**
4. **什么是 Reducer?**
5. **DeepAgents 相比 LangGraph 增加了哪些功能?**

### 6.2 技术细节

1. **LangChain 的 LCEL 是什么?有什么优势?**
2. **LangGraph 如何实现状态的持久化和恢复?**
3. **DeepAgents 的 Skills 渐进式披露机制是如何工作的?**
4. **DeepAgents 的 Subagent 和主 Agent 是什么关系?**
5. **ACP 协议的作用是什么?**

### 6.3 实战问题

1. **如何用 LangGraph 实现一个多轮对话 Agent?**
2. **DeepAgents 如何实现 Skills 的分层加载?**
3. **如果需要在一个 Agent 中调用多个子任务，你会选择 LangChain 还是 LangGraph?**
4. **DeepAgents 的 Backend 抽象是为了解决什么问题?**

---

## 七、快速代码示例

### 7.1 LangChain 简单示例

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(model="gpt-4")
prompt = PromptTemplate.from_template("用一句话描述 {topic}")
chain = prompt | llm  # LCEL 写法

result = chain.invoke({"topic": "人工智能"})
```

### 7.2 LangGraph 状态图

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(dict)
graph.add_node("first", first_node)
graph.add_node("second", second_node)
graph.add_edge("__start__", "first")
graph.add_edge("first", "second")
graph.add_edge("second", END)

app = graph.compile()
```

### 7.3 DeepAgents 创建 Agent

```python
from deepagents import create_agent
from deepagents.backends import FilesystemBackend

agent = create_agent(
    backend=FilesystemBackend(root_dir="/workspace"),
    skills=["/skills/base/", "/skills/user/"],
    model="claude-sonnet-4-20250514"
)

result = agent.invoke("帮我写一个 hello world 程序")
```

---

## 八、关键术语速查

| 术语 | 含义 |
|------|------|
| LCEL | LangChain Expression Language，函数式接口 |
| Reducer | 状态合并函数 |
| Checkpointer | 状态持久化组件 |
| Skills | 可插拔的 Agent 技能包 |
| Progressive Disclosure | 渐进式披露，按需加载 |
| ACP | Agent Client Protocol，标准化通信协议 |
| Backend | 文件/执行环境抽象层 |
| Middleware | 中间件，扩展 Agent 功能 |
| Subagent | 子代理，独立执行单元 |

---

*面试突击手册 - 祝面试顺利!*

---

## 九、深度知识点补充

### 9.1 LangChain 内部机制

#### 9.1.1 LCEL 底层原理

LCEL 的 `|` 操作符实际调用了 `Runnable.__or__`，将组件串成管道:

```python
# 底层实现简化版
class Runnable:
    def __or__(self, other):
        return RunnableSequence(self, other)
    
    def invoke(self, input, config=None):
        # 依次调用每个组件
        for step in self.steps:
            input = step.invoke(input, config)
        return input
```

**关键点**:
- 所有组件都实现 `Runnable` 接口
- 支持 `.stream()`, `.batch()`, `.ainvoke()` 统一方法
- `config` 参数传递配置（如 callbacks, recursion_limit）

#### 9.1.2 LangChain 的 Streaming 机制

```python
# 流式输出的三种方式
# 方式1: token 级别流式
for token in chain.stream({"topic": "AI"}):
    print(token, end="")

# 方式2: 消息级别流式
async for msg in chain.astream_messages({"topic": "AI"}):
    print(msg)

# 方式3: 自定义回调
from langchain.callbacks import StreamingStdOutCallbackHandler
callbacks = [StreamingStdOutCallbackHandler()]
chain.invoke({"topic": "AI"}, config={"callbacks": callbacks})
```

#### 9.1.3 Tool Calling 的实现

```python
# LangChain Agent 的 Tool Calling 流程
# 1. LLM 输出包含 tool_calls 的 JSON
# 2. LangChain 解析并绑定到 Tool 对象
# 3. 执行 Tool 并将结果作为消息追加到状态
# 4. 再次调用 LLM 处理 Tool 结果

from langchain.agents import create_openai_functions_agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

---

### 9.2 LangGraph 深度机制

#### 9.2.1 Checkpointing 机制

LangGraph 的状态持久化通过 Checkpointer 实现:

```python
from langgraph.checkpoint.memory import MemorySaver

# 编译时添加 checkpointer
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpoint)

# 指定 thread_id 保存状态
config = {"configurable": {"thread_id": "user-123"}}
app.invoke(input, config=config)

# 恢复状态
app.get_state(config)  # 获取当前状态
app.get_state_history(config)  # 获取历史状态
```

**底层原理**:
- 每次节点执行后，对 State 进行快照
- 使用 `Reducer` 合并更新
- 支持条件分支的状态追踪

#### 9.2.2 条件边与动态路由

```python
from langgraph.graph import END

def should_continue(state):
    if state["count"] > 5:
        return "end"
    return "continue"

graph = StateGraph(AgentState)
graph.add_node("process", process_node)
graph.add_conditional_edges(
    "process",
    should_continue,
    {
        "continue": "process",  # 循环
        "end": END
    }
)
```

#### 9.2.3 错误处理与重试

```python
from langgraph.prebuilt import ToolNode

# 使用 try-except 包装节点
def safe_node(state):
    try:
        # 可能失败的逻辑
        return {"result": "success"}
    except Exception as e:
        # 返回错误状态
        return {"error": str(e)}

# 或使用 LangGraph 的内置错误处理
graph.add_node("process", process_node)
# 当节点失败时自动重试
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["process"]  # 节点执行前中断（人机交互）
)
```

#### 9.2.4 Human-in-the-Loop 实现

```python
# 关键节点暂停，等待人类确认
from langgraph.constants import INTERRUPT

def ask_human(state):
    return {"status": "waiting_human"}

def process_after_human(state):
    # 人类确认后继续执行
    return {"result": "processed"}

graph = StateGraph(AgentState)
graph.add_node("ask_human", ask_human)
graph.add_node("process", process_after_human)
graph.add_edge("ask_human", "process")

# 在外部等待人类输入
app = graph.compile(interrupt_before=["process"])

# 暂停，等待人类
app.invoke(input, config=config)
# ... 人类确认后 ...
app.invoke(None, config=config)  # 继续执行
```

---

### 9.3 DeepAgents 深度机制

#### 9.3.1 Middleware 架构

DeepAgents 的 Middleware 基于 LangChain 的 Agent Middleware 协议:

```python
# Middleware 本质上是请求/响应的拦截器
class SkillsMiddleware(AgentMiddleware):
    def modify_request(self, request):
        # 在请求发送到 LLM 前修改
        return modified_request
    
    def wrap_model_call(self, request, handler):
        # 包装模型调用
        return handler(modified_request)
    
    def before_agent(self, state):
        # Agent 执行前回调
        return state_update
    
    def after_agent(self, state, response):
        # Agent 执行后回调
        return modified_response
```

**内置 Middleware**:
- `SkillsMiddleware`: 技能加载
- `MemoryMiddleware`: 长期记忆
- `SummarizationMiddleware`: 消息压缩
- `FilesystemMiddleware`: 文件系统工具
- `SubagentsMiddleware`: 子代理管理

#### 9.3.2 Backend 协议与实现

```python
# Backend 协议定义
class BackendProtocol:
    def ls_info(self, path: str) -> list[FileInfo]
    def download_files(self, paths: list[str]) -> list[FileContent]
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[str]
    def execute(self, command: str) -> ExecutionResult

# StateBackend 实现（内存存储）
class StateBackend(BackendProtocol):
    def __init__(self, runtime):
        self.runtime = runtime
        self._state = {}
    
    def ls_info(self, path):
        return list(self._state.get(path, {}).values())
    
    def download_files(self, paths):
        return [FileResponse(content=self._state.get(p)) for p in paths]
```

#### 9.3.3 ACP 协议详解

ACP (Agent Client Protocol) 是 DeepAgents 的标准化通信协议:

```python
# ACP 消息格式
{
    "jsonrpc": "2.0",
    "id": "req-123",
    "method": "tasks/run",  # 方法名
    "params": {
        "agentId": "agent-001",
        "input": {"message": "hello"},
        "stream": true
    }
}

# 响应
{
    "jsonrpc": "2.0",
    "id": "req-123",
    "result": {
        "output": "Hi there!",
        "metrics": {"tokens": 100}
    }
}

# 流式响应（Server Push）
{
    "jsonrpc": "2.0",
    "method": "output",
    "params": {"delta": "Hi"}
}
```

**核心方法**:
- `tasks/run`: 运行任务
- `tasks/cancel`: 取消任务
- `agents/list`: 列出可用 Agent
- `agents/describe`: 获取 Agent 信息

#### 9.3.4 Skills 的安全机制

```python
# SkillsMiddleware 中的安全限制
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024  # 10MB 限制
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024

# 验证逻辑
def _validate_skill_name(name: str, directory_name: str):
    # 名称必须小写字母+连字符
    # 必须与目录名匹配
    # 防止路径遍历攻击
```

---

### 9.4 高级设计模式

#### 9.4.1 Agent 任务规划模式

```python
# ReAct 模式（推理+行动）
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    llm,
    tools,
    state_modifier="你是一个助手..."
)

# Plan-and-Execute 模式
# 先规划，再执行
def planner(state):
    # 调用 LLM 生成执行计划
    plan = llm.invoke(f"为以下任务生成步骤: {state['task']}")
    return {"plan": plan}

def executor(state):
    # 按计划执行
    for step in state["plan"]["steps"]:
        result = execute_step(step)
    return {"result": result}

graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
```

#### 9.4.2 路由与分发模式

```python
# 消息路由到不同的处理节点
def route_message(state):
    last_message = state["messages"][-1]
    content = last_message.content.lower()
    
    if "search" in content:
        return "search"
    elif "code" in content:
        return "coder"
    elif "analyze" in content:
        return "analyzer"
    else:
        return "general"

graph.add_conditional_edges("router", route_message)
```

#### 9.4.3 记忆分层模式

```python
# 短期记忆（当前对话）
def short_term_memory(state):
    return {"recent_messages": state["messages"][-10:]}

# 长期记忆（向量存储）
def long_term_memory(state):
    # 检索相关历史
    docs = vectorstore.similarity_search(state["messages"][-1])
    return {"context": docs}

# 元记忆（任务摘要）
def meta_memory(state):
    # 压缩历史为摘要
    summary = summarize(state["messages"])
    return {"task_summary": summary}
```

---

### 9.5 性能优化技巧

#### 9.5.1 Token 优化

```python
# 1. 消息压缩
from langchain.globals import set_verbose, set_debug

# 2. 限制上下文长度
from langchain.schema import HumanMessage, AIMessage

# 3. 使用摘要而非完整历史
from langchain.chains import ConversationSummaryMemory

# 4. 流式响应减少等待感知
async for token in chain.astream(input):
    print(token, end="", flush=True)
```

#### 9.5.2 并发执行

```python
# 多个工具并行执行
from langgraph.prebuilt import ToolNode

# 在同一节点中调用多个工具
def parallel_tools(state):
    results = ToolNode(tools).batch([...])
    return {"results": results}
```

#### 9.5.3 缓存策略

```python
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()

# 相同 prompt 不会重复调用 LLM
```

---

### 9.6 常见面试追问及应答

#### Q: LangGraph 如何保证状态一致性?
> 通过 Reducer 函数保证。默认的 `add_messages` 会追加而非覆盖。如果需要自定义合并逻辑，可以编写自定义 Reducer。需要分布式一致性时，可使用外部存储（PostgreSQL、Redis）配合 Checkpointer。

#### Q: DeepAgents 的 Skills 加载失败如何处理?
> SkillsMiddleware 会捕获异常并跳过失败的 Skill，同时记录 Warning 日志。Agent 仍可正常运行，只是缺少该 Skill 功能。

#### Q: 如何设计一个支持百万级用户的 Agent 系统?
> 1. 使用 Stateless Design（无状态设计）
> 2. 外部化状态到 Redis/PostgreSQL
> 3. 使用消息队列解耦
> 4. 实现 Agent Pool 复用实例
> 5. 考虑降级策略（高峰期关闭某些 Middleware）

#### Q: LangChain vs LangGraph 选型建议?
> - 简单脚本/原型 → LangChain
> - 需要多轮对话/条件分支 → LangGraph
> - 企业级生产环境 → DeepAgents
> - 自研 Agent 框架 → LangGraph 底层

#### Q: 讲讲你对 Agent 架构的理解?
> Agent = LLM + Tools + Memory + Planning
> - LLM: 理解意图、生成响应
> - Tools: 扩展能力（搜索、代码执行等）
> - Memory: 保持上下文（短期+长期）
> - Planning: 任务分解（ReAct、CoT 等模式）

---

## 十、代码实战高频模式

### 模式1: 带重试的 Tool 调用

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def call_with_retry(tool, input):
    return tool.invoke(input)
```

### 模式2: 结构化输出验证

```python
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser

class Response(BaseModel):
    answer: str
    confidence: float

parser = PydanticOutputParser(pydantic_object=Response)
chain = prompt | model | parser
```

### 模式3: 动态工具选择

```python
def select_tools(task_description: str):
    # 基于任务描述动态选择工具
    if "file" in task_description:
        return [read_file, write_file]
    elif "web" in task_description:
        return [web_search, web_fetch]
    return [general_chat]
```

### 模式4: 异步批处理

```python
import asyncio
from langchain.batch import ainvoke

async def process_batch(queries):
    tasks = [chain.ainvoke(q) for q in queries]
    return await asyncio.gather(*tasks)
```

---

*补充完成 - 祝面试顺利!*
