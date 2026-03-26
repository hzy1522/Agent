"""
LangGraph 版本的 ReactAgent

迁移说明:
1. 用 StateGraph 手动构建 ReAct agent
2. 显式节点函数控制流程
3. 中间件逻辑融入节点中
"""

import os
import random
from typing import Literal
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from model.factory import chat_model
from utils.prompt_loader import load_system_prompts, load_report_prompts
from utils.logger_handler import logger
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from RAG.rag_service import RagSummarizeService


# ============ 工具定义 (从 agent_tools.py 迁移) ============

rag_service = RagSummarizeService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]

external_data = {}


def _load_external_data():
    """加载外部数据"""
    global external_data
    if external_data:
        return external_data

    external_data_path = get_abs_path(agent_conf["external_data_path"])
    if not os.path.exists(external_data_path):
        raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

    with open(external_data_path, "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            arr = line.strip().split(",")

            user_id = arr[0].replace('"', "")
            feature = arr[1].replace('"', "")
            efficiency = arr[2].replace('"', "")
            consumables = arr[3].replace('"', "")
            comparison = arr[4].replace('"', "")
            time = arr[5].replace('"', "")

            if user_id not in external_data:
                external_data[user_id] = {}

            external_data[user_id][time] = {
                "特征": feature,
                "效率": efficiency,
                "耗材": consumables,
                "对比": comparison,
            }

    return external_data


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag_service.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}天气为晴转多云，温度 25°C，南风1级，AQI=21，最近6小时降水概率为0%。"


@tool(description="获取用户所在城市的名称，以字符串的形式返回")
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户的id，以字符串的形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，以字符串的形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


@tool(description="从外部系统中获取用户指定月份的使用记录，以纯字符串的形式返回，如果没有记录则返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    data = _load_external_data()
    try:
        return data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report() -> str:
    return "fill_context_for_report已经调用"


# 工具列表
tools = [
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report
]

# 创建 ToolNode
tool_node = ToolNode(tools)


# ============ LangGraph State 定义 ============

class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: list  # 消息历史
    report: bool     # 是否报告生成模式


# ============ 节点函数 ============

def model_node(state: AgentState) -> dict:
    """
    LLM 节点: 调用模型生成回复
    """
    # 日志记录
    logger.info(f"[model_node] 模型执行，{len(state['messages'])}条消息")

    # 根据 report 状态选择 prompt
    if state.get("report", False):
        system_prompt = load_report_prompts()
        logger.info("[model_node] 使用报告生成模式")
    else:
        system_prompt = load_system_prompts()
        logger.info("[model_node] 使用普通聊天模式")

    # 检查 messages 中是否已包含 AI 消息（有则不加 system prompt）
    has_ai_message = any(isinstance(msg, AIMessage) for msg in state["messages"])

    # 只有第一轮才加 system prompt，后续轮次不加（避免破坏消息顺序）
    if has_ai_message:
        messages_with_prompt = state["messages"]
    else:
        messages_with_prompt = [HumanMessage(content=system_prompt)] + state["messages"]

    logger.info(f"[model_node] 实际消息数: {len(messages_with_prompt)}")

    # 调试: 打印消息类型和内容摘要
    for i, msg in enumerate(messages_with_prompt):
        msg_type = type(msg).__name__
        role = getattr(msg, 'role', 'N/A')
        content = getattr(msg, 'content', '')[:50] if hasattr(msg, 'content') else ''
        tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
        logger.debug(f"[model_node] msg[{i}] type={msg_type}, role={role}, content={content}..., tool_calls={bool(tool_calls)}")

    # 调用模型
    response = chat_model.bind_tools(tools).invoke(messages_with_prompt)

    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """
    工具节点: 执行模型调用的工具
    """
    last_message = state["messages"][-1]

    # 检查是否有工具调用
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    # 记录工具调用
    for tool_call in last_message.tool_calls:
        logger.info(f"[tools_node] 执行工具：{tool_call['name']}, 参数：{tool_call['args']}")

        if tool_call["name"] == "fill_context_for_report":
            logger.info("[tools_node] 检测到 fill_context_for_report")

    # 执行工具并获取结果
    result = tool_node.invoke(state)

    # 检查 fill_context_for_report 是否被调用，设置 report 标志
    # 注意: 需要在 result 中检查，因为 tool_call 的 name 可能在 ToolMessage 的 additional_kwargs 中
    messages = result["messages"]
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if msg.name == "fill_context_for_report":
                logger.info("[tools_node] fill_context_for_report 被调用，设置 report=True")
                result["report"] = True

    logger.info(f"[tools_node] 工具执行完成")
    return result


def should_continue(state: AgentState) -> Literal["tools", END]:
    """
    条件边: 决定是否继续调用工具
    """
    last_message = state["messages"][-1]

    # 如果最后一条消息有 tool_calls，说明模型要求调用工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # 检查工具调用数量，防止无限循环
        tool_call_count = sum(
            1 for msg in state["messages"]
            if hasattr(msg, "tool_calls") and msg.tool_calls
        )
        if tool_call_count >= 5:
            logger.warning("[should_continue] 工具调用次数超过5次，强制结束")
            return END
        return "tools"

    return END


# ============ 构建 Graph ============

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("model", model_node)
workflow.add_node("tools", tools_node)

# 设置入口点
workflow.set_entry_point("model")

# 添加条件边: model -> (tools | END)
workflow.add_conditional_edges(
    "model",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# 添加边: tools -> model (循环)
workflow.add_edge("tools", "model")

# 编译 graph
graph = workflow.compile()


# ============ ReactAgent 类 ============

class LangGraphReactAgent:
    """使用 LangGraph 的 ReactAgent"""

    def __init__(self):
        self.graph = graph

    def execute_stream(self, query: str):
        """
        流式执行 agent
        """
        input_state = {
            "messages": [HumanMessage(content=query)],
            "report": False
        }

        # 使用 stream 进行同步流式执行
        for event in self.graph.stream(input_state, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                # 只 yield 最终回复内容，不 yield 工具调用
                if hasattr(last_msg, "content") and last_msg.content:
                    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
                        yield last_msg.content.strip() + "\n"


if __name__ == "__main__":
    agent = LangGraphReactAgent()
    for chunk in agent.execute_stream("扫地机器人维护保养有哪些建议？"):
        print(chunk, end="", flush=True)
