#!/usr/bin/env python3
"""
AI-SDLC workflow with LangGraph + OpenAI
Includes explicit SYSTEM and USER prompts at each stage.

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
# 2) Prompts (System + User)
# ---------------------------
PROMPTS = {
    "requirements": {
        "system": """
You are a Senior Product Manager and Business Analyst.
Your job is to transform a project idea into high-quality software requirements.
Rules:
- Be precise, actionable, and concise.
- Use markdown headings and bullet points.
- Clearly separate MUST-HAVE vs NICE-TO-HAVE requirements.
- Avoid vague statements.
- Include assumptions and out-of-scope.
Output must be implementation-friendly for engineering teams.
""".strip(),
        "user_template": """
Project Idea:
{project_idea}

Generate:
1) Problem Statement
2) Product Goals (business + user)
3) User Personas
4) Functional Requirements (FR-1, FR-2, ...)
5) Non-Functional Requirements (NFR-1, NFR-2, ...)
6) MUST-HAVE vs NICE-TO-HAVE
7) Assumptions
8) Out-of-Scope
9) Acceptance Criteria (high-level)
""".strip(),
    },
    "design": {
        "system": """
You are a Principal Software Architect.
Design a practical, production-minded architecture from given requirements.
Rules:
- Prioritize simplicity, scalability, and security.
- Use markdown headings.
- Include trade-offs and justify technology choices.
- Mention failure points and mitigations.
Output should guide implementation directly.
""".strip(),
        "user_template": """
Project Idea:
{project_idea}

Requirements:
{requirements}

Generate:
1) Architecture Overview
2) Core Components/Services
3) Data Model (entities + relationships)
4) API Design (sample REST endpoints with method/path/purpose)
5) Tech Stack Recommendation (frontend/backend/db/cache/queue/infra)
6) Scalability Strategy
7) Reliability Strategy (timeouts, retries, circuit breaker ideas)
8) Risks + Mitigations
9) Open Questions
""".strip(),
    },
    "implementation": {
        "system": """
You are a Lead Software Engineer.
Create an execution-ready implementation plan and starter technical artifacts.
Rules:
- Be concrete and step-by-step.
- Favor maintainable project structure and coding standards.
- Include CI/CD and deployment considerations.
- Provide minimal but meaningful code snippets.
""".strip(),
        "user_template": """
Requirements:
{requirements}

Design:
{design}

Generate:
1) Sprint Plan (Sprint 1, Sprint 2, ...)
2) Repository/Folder Structure
3) Module Responsibilities
4) Sample Interfaces/Contracts
5) Starter Code Snippets (key flows only)
6) Database Migration Plan
7) CI/CD Plan
8) Deployment Checklist
9) Definition of Done (engineering)
""".strip(),
    },
    "testing": {
        "system": """
You are a Senior QA Lead.
Create a complete and risk-based testing strategy.
Rules:
- Cover functional, integration, e2e, and non-functional testing.
- Map tests to requirements when possible.
- Include edge cases and failure scenarios.
- Make it CI-friendly and measurable.
""".strip(),
        "user_template": """
Requirements:
{requirements}

Implementation Plan:
{implementation}

Generate:
1) Test Strategy Overview
2) Test Pyramid (unit/integration/e2e split)
3) Feature-wise Test Scenarios
4) Edge Cases and Negative Tests
5) Performance/Load Test Plan
6) Security Test Inputs for QA
7) Test Data Strategy
8) CI Quality Gates (coverage, flakiness, pass thresholds)
9) Exit Criteria for Release
""".strip(),
    },
    "security": {
        "system": """
You are an Application Security Engineer.
Perform a practical security review and produce engineering-ready controls.
Rules:
- Use threat-model thinking (assets, actors, entry points).
- Prioritize top risks by severity and likelihood.
- Recommend concrete controls and verification steps.
- Keep guidance aligned with modern web/API security practices.
""".strip(),
        "user_template": """
Project Idea:
{project_idea}

Design:
{design}

Generate:
1) Threat Model Summary
2) Top Risks (ranked)
3) Authentication & Authorization Recommendations
4) Data Protection (encryption at rest/in transit, PII handling)
5) Secrets Management Guidance
6) Secure Coding Checklist
7) Logging/Monitoring/Alerting for Security
8) Security Testing Checklist (SAST/DAST/dependency/container)
9) Incident Response Readiness Basics
""".strip(),
    },
    "final_report": {
        "system": """
You are an Engineering Manager preparing final SDLC documentation for stakeholders.
Rules:
- Produce an executive-ready but technical report.
- Ensure consistency across all sections.
- Highlight delivery roadmap, risks, and go-live readiness.
- Use markdown with clear sections and action items.
""".strip(),
        "user_template": """
Create a single final SDLC report using the sections below.

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

Output format:
1) Executive Summary
2) SDLC Phases (Requirements, Design, Implementation, Testing, Security)
3) Delivery Roadmap (Milestones + timeline suggestion)
4) Key Risks and Mitigations
5) Go-Live Readiness Checklist
6) Post-Launch Improvement Plan
""".strip(),
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
# 4) Nodes
# ---------------------------
def requirements_node(state: SDLState) -> Dict[str, Any]:
    text = run_prompt("requirements", project_idea=state["project_idea"])
    return {"requirements": text}


def design_node(state: SDLState) -> Dict[str, Any]:
    text = run_prompt(
        "design",
        project_idea=state["project_idea"],
        requirements=state["requirements"],
    )
    return {"design": text}


def implementation_node(state: SDLState) -> Dict[str, Any]:
    text = run_prompt(
        "implementation",
        requirements=state["requirements"],
        design=state["design"],
    )
    return {"implementation": text}


def testing_node(state: SDLState) -> Dict[str, Any]:
    text = run_prompt(
        "testing",
        requirements=state["requirements"],
        implementation=state["implementation"],
    )
    return {"testing": text}


def security_node(state: SDLState) -> Dict[str, Any]:
    text = run_prompt(
        "security",
        project_idea=state["project_idea"],
        design=state["design"],
    )
    return {"security": text}


def final_report_node(state: SDLState) -> Dict[str, Any]:
    text = run_prompt(
        "final_report",
        project_idea=state["project_idea"],
        requirements=state.get("requirements", ""),
        design=state.get("design", ""),
        implementation=state.get("implementation", ""),
        testing=state.get("testing", ""),
        security=state.get("security", ""),
    )
    return {"final_report": text}


# ---------------------------
# 5) Graph
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
# 6) Main
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

    # Save all phase outputs
    with open("ai_sdlc_full_output.md", "w", encoding="utf-8") as f:
        f.write("# AI-SDLC Full Output\n\n")
        f.write("## Project Idea\n")
        f.write(project_idea + "\n\n")
        f.write("## Requirements\n")
        f.write(result.get("requirements", "") + "\n\n")
        f.write("## Design\n")
        f.write(result.get("design", "") + "\n\n")
        f.write("## Implementation\n")
        f.write(result.get("implementation", "") + "\n\n")
        f.write("## Testing\n")
        f.write(result.get("testing", "") + "\n\n")
        f.write("## Security\n")
        f.write(result.get("security", "") + "\n\n")
        f.write("## Final Report\n")
        f.write(result.get("final_report", "") + "\n")

    print("\nSaved full output to ai_sdlc_full_output.md")


if __name__ == "__main__":
    main()
