# ai_sdlc_agentic_langgraph_tool.py
# --------------------------------
# Dynamic tool-calling agent with LangGraph + LangChain + @tool
#
# Install:
#   pip install -U langgraph langchain langchain-openai
#
# Run:
#   export OPENAI_API_KEY="your_key"
#   python ai_sdlc_agentic_langgraph.py
#   python ai_sdlc_agentic_langgraph.py "Build a healthcare appointment platform"

import os
import sys
from typing import Annotated, TypedDict, List

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ---------------------------
# 1) Tools
# ---------------------------
def _llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


def _ask(system_prompt: str, user_prompt: str) -> str:
    model = _llm()
    resp = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return resp.content if hasattr(resp, "content") else str(resp)


@tool
def generate_requirements(project_idea: str) -> str:
    """Generate requirements for a software project idea."""
    return _ask(
        "You are a senior product manager.",
        f"""Project idea: {project_idea}
Create: problem, personas, functional + non-functional reqs, assumptions, out-of-scope, acceptance criteria."""
    )


@tool
def generate_design(project_idea: str, requirements: str) -> str:
    """Generate architecture/design from requirements."""
    return _ask(
        "You are a principal architect.",
        f"""Project idea: {project_idea}
Requirements: {requirements}
Create architecture, components, data model, API samples, stack, risks/mitigations."""
    )


@tool
def generate_implementation(requirements: str, design: str) -> str:
    """Generate implementation plan and starter code guidance."""
    return _ask(
        "You are a lead engineer.",
        f"""Requirements: {requirements}
Design: {design}
Create sprint plan, folder structure, module responsibilities, snippets, CI/CD checklist."""
    )


@tool
def generate_testing(requirements: str, implementation: str) -> str:
    """Generate testing strategy."""
    return _ask(
        "You are a QA lead.",
        f"""Requirements: {requirements}
Implementation: {implementation}
Create unit/integration/e2e strategy, test cases, edge cases, performance plan, quality gates."""
    )


@tool
def generate_security(project_idea: str, design: str) -> str:
    """Generate security review and controls."""
    return _ask(
        "You are an application security engineer.",
        f"""Project idea: {project_idea}
Design: {design}
Create threat model, top risks, AuthN/AuthZ guidance, secrets/data protection, security checklist."""
    )


@tool
def compile_final_report(
    project_idea: str,
    requirements: str,
    design: str,
    implementation: str,
    testing: str,
    security: str,
) -> str:
    """Compile final SDLC report from all sections."""
    return _ask(
        "You are an engineering manager.",
        f"""Project: {project_idea}
Requirements: {requirements}
Design: {design}
Implementation: {implementation}
Testing: {testing}
Security: {security}
Create executive summary, roadmap, risks, go-live checklist."""
    )


TOOLS = [
    generate_requirements,
    generate_design,
    generate_implementation,
    generate_testing,
    generate_security,
    compile_final_report,
]


# ---------------------------
# 2) Graph State
# ---------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ---------------------------
# 3) Agent Node
# ---------------------------
SYSTEM_PROMPT = """
You are an AI SDLC orchestrator.
Goal: produce complete SDLC artifacts and final report for the user's project.

Rules:
- Use tools to generate each phase in order:
  1) generate_requirements
  2) generate_design
  3) generate_implementation
  4) generate_testing
  5) generate_security
  6) compile_final_report
- Reuse outputs from previous tool calls as inputs.
- After final report is ready, provide a concise completion message.
"""

llm_with_tools = _llm().bind_tools(TOOLS)


def agent_node(state: AgentState):
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


# ---------------------------
# 4) Router
# ---------------------------
def should_continue(state: AgentState):
    last = state["messages"][-1]
    # If model requested tool calls, go to tools; else finish
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


# ---------------------------
# 5) Build Graph
# ---------------------------
def build_app():
    graph = StateGraph(AgentState)
    tool_node = ToolNode(TOOLS)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ---------------------------
# 6) Main
# ---------------------------
def main():
    project_idea = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Build a SaaS project management platform with RBAC, notifications, and analytics."
    )

    app = build_app()

    user_msg = HumanMessage(
        content=f"""
Create a full SDLC package for this project idea:

{project_idea}

You must call tools to produce each phase and then compile the final report.
Return the final report in markdown.
"""
    )

    result = app.invoke({"messages": [user_msg]})
    final_messages = result["messages"]

    # Print final assistant message
    print("\n=== FINAL OUTPUT ===\n")
    print(final_messages[-1].content)

    with open("ai_sdlc_agentic_output.md", "w", encoding="utf-8") as f:
        f.write("# AI SDLC Agentic Output\n\n")
        for m in final_messages:
            role = m.__class__.__name__
            content = getattr(m, "content", "")
            if content:
                f.write(f"## {role}\n\n{content}\n\n")

    print("\nSaved: ai_sdlc_agentic_output.md")


if __name__ == "__main__":
    main()
