from langchain.agents import AgentState
from langgraph.runtime import Runtime
from langgraph.types import Command
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from typing import Callable
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts, load_report_prompts

@wrap_tool_call
def monitor_tool(               # 工具执行的监控
        request: ToolCallRequest,        # 请求的数据封装
        handler: Callable[[ToolCallRequest],ToolMessage | Command]         # 模型调用的函数
) -> ToolMessage | Command:
    logger.info(f"[tool monitor]执行工具：工具调用信息：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：工具调用信息：{request.tool_call['args']}")
    try:
        result =  handler(request)
        logger.info(f"[tool monitor]执行{request.tool_call['name']}工具成功!")

        if request.tool_call["name"] == "fill_context_report":
            request.runtime.context["report"] = True

        return result

    except Exception as e:
        logger.error(f"[tool monitor]执行工具失败：工具调用信息：{request.tool_call['name']}")
        logger.error(f"[tool monitor]执行工具失败：工具调用信息：{request.tool_call['args']}")
        logger.error(f"[tool monitor]执行工具失败：错误信息：{str(e)}")
        raise e

@before_model
def log_before_model(          # 模型执行前的监控
        state: AgentState,      # 整个agent智能体中的状态记录
        runtime: Runtime        # 记录整个执行过程中的上下文信息
):
    logger.info(f"[log_before_model]模型即将开始执行，并附带{len(state['messages'])}条消息")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None

@dynamic_prompt             #每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     #动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:           #是报告生成场景，返回报告生成场景的提示词
        return load_report_prompts()
    return load_system_prompts()


