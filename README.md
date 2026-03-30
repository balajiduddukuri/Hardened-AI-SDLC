```markdown
# AI-SDLC Agentic Workflow (LangGraph + LangChain + OpenAI)

An end-to-end **AI-powered SDLC orchestrator** built with:

- **LangGraph** (workflow/state machine + interrupts)
- **LangChain** (`@tool` abstractions)
- **OpenAI** (LLM generation)
- **Human-in-the-loop approval gates** (approve/reject/edit + resume)

This project generates software delivery artifacts in sequence:

1. Requirements  
2. Design/Architecture  
3. Implementation Plan  
4. Testing Strategy  
5. Security Review  
6. Final SDLC Report

---

## ✨ Key Features

- **Agentic orchestration** of SDLC phases
- **`@tool`-based modular generation**
- **Human approval interrupt after each phase**
- **Reject with feedback** to regenerate a phase
- **Edit mode** to manually override AI output
- **Checkpointing + resume** with `thread_id`
- **Structured state** for deterministic flow
- **Final report only after all prior phases approved**

---

## 🧱 Architecture Overview

The workflow is modeled as a graph with strict phase progression:

```text
execute_step -> approval_gate -> advance_step -> (loop or END)
```

### Nodes

- `execute_step`: runs current SDLC tool (`requirements`, `design`, etc.)
- `approval_gate`: interrupts execution and asks for human decision
- `advance_step`: moves to next step only when approved

### Human Interrupt Contract

At each phase, human can respond with JSON-like payload:

- Approve:
  ```json
  {"action":"approve"}
  ```

- Reject + feedback:
  ```json
  {"action":"reject","feedback":"Add clearer API examples"}
  ```

- Edit:
  ```json
  {"action":"edit","content":"<replacement markdown>"}
  ```

---

## 📁 Suggested Project Structure

```text
.
├── ai_sdlc_final_hardened.py
├── README.md
└── ai_sdlc_final_output.md   # generated after run
```

---

## ✅ Prerequisites

- Python 3.10+
- OpenAI API key

---

## 📦 Installation

```bash
pip install -U langgraph langchain langchain-openai
```

---

## 🔐 Environment Setup

Set your OpenAI key:

### macOS/Linux
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### Windows PowerShell
```powershell
setx OPENAI_API_KEY "your_openai_api_key"
```

---

## ▶️ Run the Project

```bash
python ai_sdlc_final_hardened.py
```

Or provide your own product idea:

```bash
python ai_sdlc_final_hardened.py "Build a healthcare appointment platform with RBAC and analytics"
```

---

## 🧑‍⚖️ Human Approval Flow (CLI)

For each generated phase, CLI will pause and ask:

```text
Action [approve/reject/edit]:
```

### Approve
Moves to next SDLC stage.

### Reject
Provide feedback; same stage is regenerated with your input.

### Edit
Paste fully edited replacement output (manual override).

---

## 🧠 SDLC Stage Details

### 1) Requirements
Generates:
- problem statement
- functional/non-functional requirements
- assumptions
- scope boundaries

### 2) Design
Generates:
- architecture
- components/services
- data model
- API surface
- risks/mitigations

### 3) Implementation
Generates:
- sprint breakdown
- module plan
- CI/CD checklist
- deployment readiness points

### 4) Testing
Generates:
- unit/integration/e2e strategy
- edge/negative scenarios
- quality gates

### 5) Security
Generates:
- threat model
- major risks
- auth/data/secrets controls
- security testing checklist

### 6) Final Report
Compiles all approved sections into stakeholder-ready SDLC output.

---

## 🗂 Output

Final approved artifacts are saved to:

- `ai_sdlc_final_output.md`

This includes all six SDLC sections.

---

## ⚙️ Configuration Notes

In code, you can tune:

- model name (`gpt-4o-mini` → `gpt-4.1`, etc.)
- temperature
- interrupt preview length
- thread id for checkpoint sessions

---

## 🔁 Resume Behavior

The graph uses `MemorySaver` checkpointer and `thread_id` config.  
If interrupted, execution resumes from the exact paused point when you send `Command(resume=...)`.

> For persistent resume across process restarts, replace in-memory saver with a persistent checkpointer (e.g., Redis/DB-backed).

---

## 🛡 Guardrails Implemented

- strict stage sequence
- no final report before all prior approvals
- explicit human gate per stage
- deterministic canonical state (`approved`, `feedback`, `current_step`)
- basic error interrupt (`retry` / `abort`)

---

## 🧪 Example Run (Short)

Input:
```text
Build a SaaS project management app with RBAC and analytics
```

Flow:
- Requirements generated → approved
- Design generated → rejected with feedback → regenerated → approved
- Implementation → approved
- Testing → edited manually
- Security → approved
- Final report → approved and exported

---

## 🚀 Roadmap / Improvements

- [ ] Split into modules (`tools.py`, `graph.py`, `runner.py`, `prompts.py`)
- [ ] Add `.env` support (`python-dotenv`)
- [ ] Add logging + tracing
- [ ] Add token/cost tracking
- [ ] Add unit/integration tests
- [ ] Add web UI for approvals
- [ ] Add persistent checkpointer backend
- [ ] Add model fallback/retry policy

---

## 🧯 Troubleshooting

### `OPENAI_API_KEY is not set`
Set environment variable correctly in current shell session.

### Graph pauses and “hangs”
It is likely waiting at a human interrupt. Check CLI prompt for action input.

### Unexpected output quality
Use `reject` with explicit feedback or `edit` to enforce exact content.

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Test locally
5. Open PR with clear description

---

## 📜 License

Choose your preferred license (MIT recommended for open-source starter projects).

---

## 🙌 Acknowledgements

- LangGraph team for interrupt/resume primitives
- LangChain for tool abstractions
- OpenAI for LLM capabilities
```

If you want, I can also generate:
1. a **short README** version for GitHub top-fold, and  
2. a **docs/ARCHITECTURE.md** with sequence diagrams.
