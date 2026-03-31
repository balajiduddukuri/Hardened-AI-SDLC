#!/usr/bin/env python3
"""
AI-SDLC workflow with LangGraph + OpenAI.

Steps:
1) Requirements
2) Design
3) Implementation
4) Testing
5) Security
6) Final report

Usage:
    export OPENAI_API_KEY="your_key_here"
    python ai_sdlc_langgraph.py

Optional:
    python ai_sdlc_langgraph.py "Build a Flask todo app with JWT auth and PostgreSQL"
"""

import os
import sys
from typing import TypedDict, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ---------------------------
# 1) Define Graph State
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
# 2) LLM Helper
# ---------------------------
def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Please export it before running."
        )
    return ChatOpenAI(
        model="gpt-4o-mini",   # change if desired
        temperature=0.2
    )


def ask_llm(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> str:
    resp = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return resp.content if hasattr(resp, "content") else str(resp)


# ---------------------------
# 3) Node Functions
# ---------------------------
def requirements_node(state: SDLState) -> Dict[str, Any]:
    llm = get_llm()
    idea = state["project_idea"]

    system_prompt = (
        "You are a senior product manager. Produce concise and clear software requirements."
    )
    user_prompt = f"""
Project Idea:
{idea}

Create:
- Problem statement
- User personas
- Functional requirements
- Non-functional requirements
- Assumptions
- Out-of-scope items
"""
    requirements = ask_llm(llm, system_prompt, user_prompt)
    return {"requirements": requirements}


def design_node(state: SDLState) -> Dict[str, Any]:
    llm = get_llm()
    idea = state["project_idea"]
    requirements = state["requirements"]

    system_prompt = (
        "You are a software architect. Provide practical architecture/design."
    )
    user_prompt = f"""
Project Idea:
{idea}

Requirements:
{requirements}

Create:
- High-level architecture
- Key components/services
- Data model suggestions
- API design (sample endpoints)
- Technology stack recommendation
- Risks and mitigations
"""
    design = ask_llm(llm, system_prompt, user_prompt)
    return {"design": design}


def implementation_node(state: SDLState) -> Dict[str, Any]:
    llm = get_llm()
    requirements = state["requirements"]
    design = state["design"]

    system_prompt = (
        "You are a lead software engineer. Provide implementation plan and starter code."
    )
    user_prompt = f"""
Requirements:
{requirements}

Design:
{design}

Create:
- Sprint-wise implementation plan
- Folder structure
- Core module responsibilities
- Minimal starter code snippets
- Deployment checklist
"""
    implementation = ask_llm(llm, system_prompt, user_prompt)
    return {"implementation": implementation}


def testing_node(state: SDLState) -> Dict[str, Any]:
    llm = get_llm()
    requirements = state["requirements"]
    implementation = state["implementation"]

    system_prompt = "You are a QA lead. Create a robust test strategy."
    user_prompt = f"""
Requirements:
{requirements}

Implementation Plan:
{implementation}

Create:
- Test strategy (unit/integration/e2e)
- Test cases by feature
- Edge cases
- Performance test ideas
- CI test gating policy
"""
    testing = ask_llm(llm, system_prompt, user_prompt)
    return {"testing": testing}


def security_node(state: SDLState) -> Dict[str, Any]:
    llm = get_llm()
    idea = state["project_idea"]
    design = state["design"]

    system_prompt = "You are an application security engineer."
    user_prompt = f"""
Project Idea:
{idea}

Design:
{design}

Create:
- Threat model summary
- Top security risks
- Secure coding controls
- AuthN/AuthZ recommendations
- Secrets management guidance
- Security testing checklist
"""
    security = ask_llm(llm, system_prompt, user_prompt)
    return {"security": security}


def final_report_node(state: SDLState) -> Dict[str, Any]:
    llm = get_llm()

    system_prompt = "You are an engineering manager preparing final SDLC documentation."
    user_prompt = f"""
Combine the following sections into one coherent SDLC report:

[Requirements]
{state.get("requirements", "")}

[Design]
{state.get("design", "")}

[Implementation]
{state.get("implementation", "")}

[Testing]
{state.get("testing", "")}

[Security]
{state.get("security", "")}

Output:
- Executive summary
- Detailed SDLC sections
- Delivery roadmap
- Go-live readiness checklist
"""
    final_report = ask_llm(llm, system_prompt, user_prompt)
    return {"final_report": final_report}


# ---------------------------
# 4) Build Graph
# ---------------------------
def build_graph():
    workflow = StateGraph(SDLState)

    workflow.add_node("requirements", requirements_node)
    workflow.add_node("design", design_node)
    workflow.add_node("implementation", implementation_node)
    workflow.add_node("testing", testing_node)
    workflow.add_node("security", security_node)
    workflow.add_node("final_report", final_report_node)

    workflow.set_entry_point("requirements")
    workflow.add_edge("requirements", "design")
    workflow.add_edge("design", "implementation")
    workflow.add_edge("implementation", "testing")
    workflow.add_edge("testing", "security")
    workflow.add_edge("security", "final_report")
    workflow.add_edge("final_report", END)

    return workflow.compile()


# ---------------------------
# 5) Main
# ---------------------------
def main():
    project_idea = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Build a SaaS project management platform with role-based access, notifications, and analytics dashboard."
    )

    app = build_graph()

    initial_state: SDLState = {
        "project_idea": project_idea
    }

    result = app.invoke(initial_state)

    print("\n" + "=" * 80)
    print("AI-SDLC FINAL REPORT")
    print("=" * 80 + "\n")
    print(result.get("final_report", "No report generated."))

    # Optional: save outputs to file
    with open("ai_sdlc_output.md", "w", encoding="utf-8") as f:
        f.write("# AI-SDLC Report\n\n")
        f.write(result.get("final_report", ""))

    print("\nSaved final report to ai_sdlc_output.md")


if __name__ == "__main__":
    main()
