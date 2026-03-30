# ai_sdlc_final_hardened.py
# -------------------------------------------------------
# Final hardened AI-SDLC:
# - LangGraph + LangChain + @tool
# - Human approval interrupts after each stage
# - Strict stage order enforcement
# - Structured canonical state
#
# Install:
#   pip install -U langgraph langchain langchain-openai
#
# Run:
#   export OPENAI_API_KEY="your_key"
#   python ai_sdlc_final_hardened.py
#   python ai_sdlc_final_hardened.py "Build a fintech expense app"

import os
import sys
from typing import TypedDict, Dict, Any, Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


STEPS = ["requirements", "design", "implementation", "testing", "security", "final_report"]


# ---------------------------
# State
# ---------------------------
class SDLCState(TypedDict, total=False):
    project_idea: str
    current_step: str
    last_output: str
    approved: Dict[str, str]
    feedback: Dict[str, str]
    done: bool
    error: Optional[str]


# ---------------------------
# LLM helper
# ---------------------------
def get_llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


def ask(system_prompt: str, user_prompt: str) -> str:
    llm = get_llm()
    resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return resp.content if hasattr(resp, "content") else str(resp)


# ---------------------------
# Tools
# ---------------------------
@tool
def t_requirements(project_idea: str, feedback: str = "") -> str:
    """Generate requirements."""
    return ask(
        "You are a senior product manager. Return concise markdown.",
        f"Project: {project_idea}\nFeedback: {feedback}\nGenerate requirements, NFRs, assumptions, scope.",
    )


@tool
def t_design(project_idea: str, requirements: str, feedback: str = "") -> str:
    """Generate design."""
    return ask(
        "You are a principal architect. Return concise markdown.",
        f"Project: {project_idea}\nRequirements:\n{requirements}\nFeedback: {feedback}\nGenerate architecture, APIs, data model.",
    )


@tool
def t_implementation(requirements: str, design: str, feedback: str = "") -> str:
    """Generate implementation plan."""
    return ask(
        "You are a lead engineer. Return concise markdown.",
        f"Requirements:\n{requirements}\nDesign:\n{design}\nFeedback: {feedback}\nGenerate plan, modules, CI/CD.",
    )


@tool
def t_testing(requirements: str, implementation: str, feedback: str = "") -> str:
    """Generate testing strategy."""
    return ask(
        "You are a QA lead. Return concise markdown.",
        f"Requirements:\n{requirements}\nImplementation:\n{implementation}\nFeedback: {feedback}\nGenerate test strategy.",
    )


@tool
def t_security(project_idea: str, design: str, feedback: str = "") -> str:
    """Generate security review."""
    return ask(
        "You are an appsec engineer. Return concise markdown.",
        f"Project: {project_idea}\nDesign:\n{design}\nFeedback: {feedback}\nGenerate threat model + controls.",
    )


@tool
def t_final_report(
    project_idea: str,
    requirements: str,
    design: str,
    implementation: str,
    testing: str,
    security: str,
) -> str:
    """Compile final report."""
    return ask(
        "You are an engineering manager. Return final stakeholder-ready markdown.",
        f"""Project: {project_idea}
Requirements: {requirements}
Design: {design}
Implementation: {implementation}
Testing: {testing}
Security: {security}
Create executive summary, roadmap, risks, go-live checklist.""",
    )


# ---------------------------
# Stage executor
# ---------------------------
def execute_step(state: SDLCState) -> Dict[str, Any]:
    step = state["current_step"]
    approved = state.get("approved", {})
    feedback_map = state.get("feedback", {})
    fb = feedback_map.get(step, "")

    try:
        if step == "requirements":
            out = t_requirements.invoke({"project_idea": state["project_idea"], "feedback": fb})
        elif step == "design":
            out = t_design.invoke({
                "project_idea": state["project_idea"],
                "requirements": approved["requirements"],
                "feedback": fb
            })
        elif step == "implementation":
            out = t_implementation.invoke({
                "requirements": approved["requirements"],
                "design": approved["design"],
                "feedback": fb
            })
        elif step == "testing":
            out = t_testing.invoke({
                "requirements": approved["requirements"],
                "implementation": approved["implementation"],
                "feedback": fb
            })
        elif step == "security":
            out = t_security.invoke({
                "project_idea": state["project_idea"],
                "design": approved["design"],
                "feedback": fb
            })
        elif step == "final_report":
            # Hard guard: only compile if all prior stages approved
            for k in ["requirements", "design", "implementation", "testing", "security"]:
                if k not in approved:
                    raise ValueError(f"Missing approved stage: {k}")
            out = t_final_report.invoke({
                "project_idea": state["project_idea"],
                "requirements": approved["requirements"],
                "design": approved["design"],
                "implementation": approved["implementation"],
                "testing": approved["testing"],
                "security": approved["security"],
            })
        else:
            raise ValueError(f"Unknown step: {step}")

        return {"last_output": out, "error": None}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------
# Human gate
# ---------------------------
def approval_gate(state: SDLCState) -> Dict[str, Any]:
    if state.get("error"):
        # Let human decide how to proceed if error occurred
        decision = interrupt({
            "type": "error",
            "step": state["current_step"],
            "error": state["error"],
            "expected": {"action": "retry|abort"}
        })
        action = (decision or {}).get("action", "retry")
        if action == "abort":
            return {"done": True}
        return {"error": None}  # retry same step

    step = state["current_step"]
    output = state.get("last_output", "")

    decision = interrupt({
        "type": "approval",
        "step": step,
        "preview": output[:5000],
        "expected": {
            "approve": {"action": "approve"},
            "reject": {"action": "reject", "feedback": "improvement feedback"},
            "edit": {"action": "edit", "content": "replacement markdown"},
        }
    })

    action = (decision or {}).get("action", "approve").lower()
    approved = dict(state.get("approved", {}))
    feedback = dict(state.get("feedback", {}))

    if action == "approve":
        approved[step] = output
        feedback.pop(step, None)
        return {"approved": approved, "feedback": feedback}

    if action == "edit":
        approved[step] = (decision or {}).get("content", "").strip() or output
        feedback.pop(step, None)
        return {"approved": approved, "feedback": feedback}

    # reject
    feedback[step] = (decision or {}).get("feedback", "Improve quality and completeness.")
    return {"feedback": feedback}


def advance_step(state: SDLCState) -> Dict[str, Any]:
    step = state["current_step"]
    approved = state.get("approved", {})

    # If current step not approved, stay on same step
    if step not in approved:
        return {}

    idx = STEPS.index(step)
    if idx == len(STEPS) - 1:
        return {"done": True}
    return {"current_step": STEPS[idx + 1]}


def route_done(state: SDLCState):
    return END if state.get("done") else "execute_step"


# ---------------------------
# Build graph
# ---------------------------
def build_app():
    g = StateGraph(SDLCState)
    g.add_node("execute_step", execute_step)
    g.add_node("approval_gate", approval_gate)
    g.add_node("advance_step", advance_step)

    g.set_entry_point("execute_step")
    g.add_edge("execute_step", "approval_gate")
    g.add_edge("approval_gate", "advance_step")
    g.add_conditional_edges("advance_step", route_done, {"execute_step": "execute_step", END: END})

    return g.compile(checkpointer=MemorySaver())


# ---------------------------
# CLI
# ---------------------------
def main():
    idea = sys.argv[1] if len(sys.argv) > 1 else "Build a SaaS project management app with RBAC and analytics."
    app = build_app()
    config = {"configurable": {"thread_id": "final-sdlc-thread"}}

    result = app.invoke(
        {
            "project_idea": idea,
            "current_step": "requirements",
            "approved": {},
            "feedback": {},
            "done": False,
        },
        config=config,
    )

    while "__interrupt__" in result:
        i = result["__interrupt__"][0].value
        print("\n" + "=" * 80)
        print(f"INTERRUPT :: {i.get('type')} :: step={i.get('step')}")
        print("=" * 80)
        if i.get("type") == "error":
            print(i.get("error"))
            action = input("Action [retry/abort]: ").strip().lower() or "retry"
            payload = {"action": action}
        else:
            print(i.get("preview", ""))
            action = input("Action [approve/reject/edit]: ").strip().lower() or "approve"
            if action == "reject":
                fb = input("Feedback: ").strip()
                payload = {"action": "reject", "feedback": fb}
            elif action == "edit":
                print("Paste edited text. End with :::end")
                lines = []
                while True:
                    line = input()
                    if line.strip() == ":::end":
                        break
                    lines.append(line)
                payload = {"action": "edit", "content": "\n".join(lines)}
            else:
                payload = {"action": "approve"}

        result = app.invoke(Command(resume=payload), config=config)

    approved = result.get("approved", {})
    final_report = approved.get("final_report", "No final report generated.")

    print("\n\n===== FINAL REPORT =====\n")
    print(final_report)

    with open("ai_sdlc_final_output.md", "w", encoding="utf-8") as f:
        f.write("# Final AI-SDLC Output\n\n")
        for s in STEPS:
            f.write(f"## {s.title().replace('_', ' ')}\n\n{approved.get(s, '')}\n\n")

    print("\nSaved: ai_sdlc_final_output.md")


if __name__ == "__main__":
    main()
