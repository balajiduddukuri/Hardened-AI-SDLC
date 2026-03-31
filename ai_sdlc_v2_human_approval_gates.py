#!/usr/bin/env python3
"""
AI-SDLC workflow with LangGraph + OpenAI + Human Approval Gates

Usage:
    export OPENAI_API_KEY="your_key_here"
    python ai_sdlc_langgraph.py
    python ai_sdlc_langgraph.py "Build a Flask todo app with JWT auth and PostgreSQL"
"""

import os
import sys
from typing import TypedDict, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ---------------------------
# 1) State
# ---------------------------
class SDLState(TypedDict, total=False):
    project_idea: str
    requirements: str
    design: str
    implementation: str
    testing: str
    security: str
    final_report: str


# ---------------------------
# 2) Prompts
# ---------------------------
PROMPTS = {
    "requirements": {
        "system": """You are a Senior Product Manager and Business Analyst.
Create clear, actionable requirements in markdown.
Include MUST-HAVE vs NICE-TO-HAVE, assumptions, out-of-scope.""",
        "user_template": """
Project Idea:
{project_idea}

Generate:
1) Problem Statement
2) Product Goals
3) User Personas
4) Functional Requirements (FR-1...)
5) Non-Functional Requirements (NFR-1...)
6) MUST-HAVE vs NICE-TO-HAVE
7) Assumptions
8) Out-of-Scope
9) Acceptance Criteria
Feedback to apply (if any):
{feedback}
""",
    },
    "design": {
        "system": """You are a Principal Software Architect.
Produce practical architecture with trade-offs, risks, and mitigations.""",
        "user_template": """
Project Idea:
{project_idea}

Requirements:
{requirements}

Generate:
1) Architecture Overview
2) Components/Services
3) Data Model
4) API design samples
5) Tech Stack recommendation
6) Scalability + Reliability strategy
7) Risks + Mitigations
Feedback to apply (if any):
{feedback}
""",
    },
    "implementation": {
        "system": """You are a Lead Software Engineer.
Create an execution-ready implementation plan and starter snippets.""",
        "user_template": """
Requirements:
{requirements}

Design:
{design}

Generate:
1) Sprint plan
2) Folder structure
3) Module responsibilities
4) Starter snippets
5) DB migration plan
6) CI/CD plan
7) Deployment checklist
Feedback to apply (if any):
{feedback}
""",
    },
    "testing": {
        "system": """You are a Senior QA Lead.
Create a risk-based test strategy that is CI-friendly.""",
        "user_template": """
Requirements:
{requirements}

Implementation:
{implementation}

Generate:
1) Test strategy
2) Unit/Integration/E2E plan
3) Feature-wise test cases
4) Edge/negative cases
5) Performance tests
6) CI quality gates
Feedback to apply (if any):
{feedback}
""",
    },
    "security": {
        "system": """You are an Application Security Engineer.
Provide practical threat model and prioritized controls.""",
        "user_template": """
Project Idea:
{project_idea}

Design:
{design}

Generate:
1) Threat model summary
2) Top risks ranked
3) AuthN/AuthZ recommendations
4) Data protection guidance
5) Secrets management
6) Security testing checklist
Feedback to apply (if any):
{feedback}
""",
    },
    "final_report": {
        "system": """You are an Engineering Manager.
Create a final stakeholder-ready SDLC report in markdown.""",
        "user_template": """
[Project Idea]
{project_idea}

[Requirements]
{requirements}

[Design]
{design}

[Implementation]
{implementation}

[Testing]
{testing}

[Security]
{security}

Output:
1) Executive Summary
2) SDLC phase details
3) Delivery roadmap
4) Risks and mitigations
5) Go-live checklist
6) Post-launch plan
""",
    },
}


# ---------------------------
# 3) LLM Helpers
# ---------------------------
def get_llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


def run_prompt(stage: str, **kwargs) -> str:
    llm = get_llm()
    system_prompt = PROMPTS[stage]["system"]
    user_prompt = PROMPTS[stage]["user_template"].format(**kwargs)
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return response.content if hasattr(response, "content") else str(response)


# ---------------------------
# 4) Human Approval Gate
# ---------------------------
def human_gate(stage_name: str, content: str, regen_fn):
    """
    regen_fn(feedback: str) -> str
    """
    while True:
        print("\n" + "=" * 80)
        print(f"STAGE: {stage_name.upper()}")
        print("=" * 80)
        print(content[:8000])  # avoid flooding terminal too much
        print("\nOptions: [approve] continue | [edit] manual edit | [reject] regenerate")
        choice = input("Your choice: ").strip().lower()

        if choice in {"approve", "a"}:
            return content
        elif choice in {"edit", "e"}:
            print("\nEnter your replacement text. Finish with a single line: :::end")
            lines = []
            while True:
                line = input()
                if line.strip() == ":::end":
                    break
                lines.append(line)
            edited = "\n".join(lines).strip()
            if edited:
                return edited
            print("Empty edit received. Keeping previous content.")
            return content
        elif choice in {"reject", "r"}:
            feedback = input("Enter feedback for regeneration: ").strip()
            content = regen_fn(feedback=feedback if feedback else "Improve clarity and structure.")
            print("\nRegenerated. Review again.")
        else:
            print("Invalid choice. Type approve/edit/reject.")


# ---------------------------
# 5) Nodes with approval
# ---------------------------
def requirements_node(state: SDLState) -> Dict[str, Any]:
    def regen_fn(feedback: str):
        return run_prompt(
            "requirements",
            project_idea=state["project_idea"],
            feedback=feedback,
        )

    draft = regen_fn(feedback="None")
    approved = human_gate("requirements", draft, regen_fn)
    return {"requirements": approved}


def design_node(state: SDLState) -> Dict[str, Any]:
    def regen_fn(feedback: str):
        return run_prompt(
            "design",
            project_idea=state["project_idea"],
            requirements=state["requirements"],
            feedback=feedback,
        )

    draft = regen_fn(feedback="None")
    approved = human_gate("design", draft, regen_fn)
    return {"design": approved}


def implementation_node(state: SDLState) -> Dict[str, Any]:
    def regen_fn(feedback: str):
        return run_prompt(
            "implementation",
            requirements=state["requirements"],
            design=state["design"],
            feedback=feedback,
        )

    draft = regen_fn(feedback="None")
    approved = human_gate("implementation", draft, regen_fn)
    return {"implementation": approved}


def testing_node(state: SDLState) -> Dict[str, Any]:
    def regen_fn(feedback: str):
        return run_prompt(
            "testing",
            requirements=state["requirements"],
            implementation=state["implementation"],
            feedback=feedback,
        )

    draft = regen_fn(feedback="None")
    approved = human_gate("testing", draft, regen_fn)
    return {"testing": approved}


def security_node(state: SDLState) -> Dict[str, Any]:
    def regen_fn(feedback: str):
        return run_prompt(
            "security",
            project_idea=state["project_idea"],
            design=state["design"],
            feedback=feedback,
        )

    draft = regen_fn(feedback="None")
    approved = human_gate("security", draft, regen_fn)
    return {"security": approved}


def final_report_node(state: SDLState) -> Dict[str, Any]:
    report = run_prompt(
        "final_report",
        project_idea=state["project_idea"],
        requirements=state.get("requirements", ""),
        design=state.get("design", ""),
        implementation=state.get("implementation", ""),
        testing=state.get("testing", ""),
        security=state.get("security", ""),
    )
    return {"final_report": report}


# ---------------------------
# 6) Graph
# ---------------------------
def build_graph():
    g = StateGraph(SDLState)

    g.add_node("requirements", requirements_node)
    g.add_node("design", design_node)
    g.add_node("implementation", implementation_node)
    g.add_node("testing", testing_node)
    g.add_node("security", security_node)
    g.add_node("final_report", final_report_node)

    g.set_entry_point("requirements")
    g.add_edge("requirements", "design")
    g.add_edge("design", "implementation")
    g.add_edge("implementation", "testing")
    g.add_edge("testing", "security")
    g.add_edge("security", "final_report")
    g.add_edge("final_report", END)

    return g.compile()


# ---------------------------
# 7) Main
# ---------------------------
def main():
    project_idea = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Build a SaaS project management platform with role-based access, notifications, and analytics dashboard."
    )

    app = build_graph()
    result = app.invoke({"project_idea": project_idea})

    print("\n" + "=" * 80)
    print("AI-SDLC FINAL REPORT")
    print("=" * 80 + "\n")
    print(result.get("final_report", "No final report generated."))

    with open("ai_sdlc_full_output.md", "w", encoding="utf-8") as f:
        f.write("# AI-SDLC Full Output\n\n")
        f.write(f"## Project Idea\n{project_idea}\n\n")
        f.write(f"## Requirements\n{result.get('requirements','')}\n\n")
        f.write(f"## Design\n{result.get('design','')}\n\n")
        f.write(f"## Implementation\n{result.get('implementation','')}\n\n")
        f.write(f"## Testing\n{result.get('testing','')}\n\n")
        f.write(f"## Security\n{result.get('security','')}\n\n")
        f.write(f"## Final Report\n{result.get('final_report','')}\n")

    print("\nSaved full output to ai_sdlc_full_output.md")


if __name__ == "__main__":
    main()
