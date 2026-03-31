# ai_sdlc_agentic_human_gate.py
# -------------------------------------------------------
# Agentic SDLC with:
# - LangGraph
# - LangChain tools (@tool)
# - Dynamic tool calling
# - Human approval interrupts/resume
#
# Install:
#   pip install -U langgraph langchain langchain-openai
#
# Run:
#   export OPENAI_API_KEY="your_key"
#   python ai_sdlc_agentic_human_gate.py
#   python ai_sdlc_agentic_human_gate.py "Build a fintech budgeting app"

import os
import sys
import json
from typing import Annotated, TypedDict, List, Dict, Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ---------------------------
# 1) LLM
# ---------------------------
def get_llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


def ask(system_prompt: str, user_prompt: str) -> str:
    llm = get_llm()
    resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return resp.content if hasattr(resp, "content") else str(resp)


# ---------------------------
# 2) Tools
# ---------------------------
@tool
def generate_requirements(project_idea: str) -> str:
    """Generate software requirements from project idea."""
    return ask(
        "You are a senior product manager. Return markdown.",
        f"""Project idea: {project_idea}
Create: problem statement, personas, FRs, NFRs, assumptions, out-of-scope, acceptance criteria."""
    )


@tool
def generate_design(project_idea: str, requirements: str) -> str:
    """Generate software architecture/design from requirements."""
    return ask(
        "You are a principal software architect. Return markdown.",
        f"""Project idea: {project_idea}
Requirements: {requirements}
Create: architecture, components, data model, API samples, stack, risks/mitigations."""
    )


@tool
def generate_implementation(requirements: str, design: str) -> str:
    """Generate implementation plan from requirements and design."""
    return ask(
        "You are a lead software engineer. Return markdown.",
        f"""Requirements: {requirements}
Design: {design}
Create: sprint plan, folder structure, modules, starter snippets, CI/CD, deployment checklist."""
    )


@tool
def generate_testing(requirements: str, implementation: str) -> str:
    """Generate testing strategy and test cases."""
    return ask(
        "You are a QA lead. Return markdown.",
        f"""Requirements: {requirements}
Implementation: {implementation}
Create: unit/integration/e2e plan, feature tests, edge cases, performance plan, quality gates."""
    )


@tool
def generate_security(project_idea: str, design: str) -> str:
    """Generate security review and controls."""
    return ask(
        "You are an application security engineer. Return markdown.",
        f"""Project idea: {project_idea}
Design: {design}
Create: threat model, top risks, auth recommendations, secrets management, security checklist."""
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
    """Compile final SDLC report."""
    return ask(
        "You are an engineering manager. Return markdown.",
        f"""Project: {project_idea}
Requirements: {requirements}
Design: {design}
Implementation: {implementation}
Testing: {testing}
Security: {security}
Create: executive summary, roadmap, risks/mitigations, go-live checklist."""
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
# 3) State
# ---------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    approved_outputs: Dict[str, str]  # tool_name -> approved text


SYSTEM_PROMPT = """
You are an AI SDLC orchestrator.

Process (must follow):
1) generate_requirements
2) generate_design
3) generate_implementation
4) generate_testing
5) generate_security
6) compile_final_report

Use approved outputs from previous steps as tool inputs.
After each tool call, wait for human approval (handled by graph).
"""


# ---------------------------
# 4) Nodes
# ---------------------------
llm_with_tools = get_llm().bind_tools(TOOLS)


def agent_node(state: AgentState):
    msgs = state["messages"]
    response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + msgs)
    return {"messages": [response]}


def route_after_agent(state: AgentState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


tool_node = ToolNode(TOOLS)


def human_approval_node(state: AgentState):
    """
    Inspects last ToolMessage and asks for human approval.
    interrupt(...) pauses graph and expects resume via Command(resume=...).
    """
    messages = state["messages"]
    last_msg = messages[-1]

    # Only gate ToolMessage; otherwise pass
    if not isinstance(last_msg, ToolMessage):
        return {}

    tool_name = last_msg.name
    tool_output = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    # Pause and request human input
    decision = interrupt(
        {
            "type": "approval_required",
            "tool_name": tool_name,
            "tool_output_preview": tool_output[:4000],
            "instructions": "Respond JSON: "
                            '{"action":"approve"} OR '
                            '{"action":"reject","feedback":"..."} OR '
                            '{"action":"edit","content":"..."}'
        }
    )

    # decision is whatever passed in Command(resume=...)
    if isinstance(decision, str):
        try:
            decision = json.loads(decision)
        except Exception:
            decision = {"action": "approve"}

    action = (decision or {}).get("action", "approve").lower()
    approved_outputs = dict(state.get("approved_outputs", {}))

    if action == "approve":
        approved_outputs[tool_name] = tool_output
        return {"approved_outputs": approved_outputs}

    if action == "edit":
        edited = (decision or {}).get("content", "").strip()
        approved_outputs[tool_name] = edited if edited else tool_output

        # Replace last ToolMessage with edited content so agent sees edited version
        edited_tool_msg = ToolMessage(
            content=approved_outputs[tool_name],
            name=tool_name,
            tool_call_id=last_msg.tool_call_id,
        )
        return {"approved_outputs": approved_outputs, "messages": [edited_tool_msg]}

    # reject -> add human feedback as HumanMessage and let agent retry
    feedback = (decision or {}).get("feedback", "Please improve quality and clarity.")
    return {
        "messages": [
            HumanMessage(
                content=f"Tool output from `{tool_name}` is rejected. Regenerate with this feedback: {feedback}"
            )
        ]
    }


def route_after_human_approval(state: AgentState):
    """
    If last message is human rejection instruction -> go back to agent.
    Else continue to agent (normal flow).
    """
    return "agent"


# ---------------------------
# 5) Build graph
# ---------------------------
def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("human_approval", human_approval_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "human_approval")
    graph.add_conditional_edges("human_approval", route_after_human_approval, {"agent": "agent"})

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ---------------------------
# 6) CLI runner with resume loop
# ---------------------------
def main():
    idea = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Build a SaaS project management platform with RBAC, notifications, and analytics."
    )

    app = build_app()
    config = {"configurable": {"thread_id": "sdlc-thread-1"}}

    # Start run
    state = {
        "messages": [
            HumanMessage(
                content=f"Create complete SDLC artifacts for this project idea: {idea}. "
                        f"Use tools in required sequence and compile final report."
            )
        ],
        "approved_outputs": {},
    }

    result = app.invoke(state, config=config)

    # Handle interrupts until done
    while "__interrupt__" in result:
        intr = result["__interrupt__"][0].value
        print("\n" + "=" * 80)
        print(f"APPROVAL NEEDED: {intr.get('tool_name')}")
        print("=" * 80)
        print(intr.get("tool_output_preview", ""))

        choice = input("\nAction [approve/edit/reject]: ").strip().lower()

        if choice == "edit":
            print("Enter edited content. End with line :::end")
            lines = []
            while True:
                line = input()
                if line.strip() == ":::end":
                    break
                lines.append(line)
            payload = {"action": "edit", "content": "\n".join(lines)}
        elif choice == "reject":
            fb = input("Feedback for regeneration: ").strip()
            payload = {"action": "reject", "feedback": fb or "Improve quality and completeness."}
        else:
            payload = {"action": "approve"}

        result = app.invoke(Command(resume=payload), config=config)

    # Print final
    messages = result["messages"]
    final_text = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            final_text = m.content
            break

    print("\n" + "=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)
    print(final_text or "No final AI output found.")

    # Save artifact
    with open("ai_sdlc_agentic_human_gate_output.md", "w", encoding="utf-8") as f:
        f.write("# AI SDLC Agentic Output (Human-Gated)\n\n")
        for msg in messages:
            role = msg.__class__.__name__
            content = getattr(msg, "content", "")
            if content:
                f.write(f"## {role}\n\n{content}\n\n")

    print("\nSaved: ai_sdlc_agentic_human_gate_output.md")


if __name__ == "__main__":
    main()
