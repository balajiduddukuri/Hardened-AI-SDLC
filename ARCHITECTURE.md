## AI-SDLC Agentic Workflow Architecture

This document describes the architecture of the **human-gated AI SDLC orchestrator** built with:

- **LangGraph** for workflow/state transitions
- **LangChain tools (`@tool`)** for modular SDLC generation
- **OpenAI LLM** for content creation
- **Interrupt/Resume** for human approval after each stage

---

## 1. Goals

### Primary Goals
- Generate high-quality SDLC artifacts quickly
- Keep humans in control of every stage
- Enforce deterministic stage order
- Support pause/resume safely

### Non-Goals
- Autonomous deployment without human review
- Fully unsupervised code generation pipeline
- Multi-tenant production backend (in current CLI implementation)

---

## 2. System Context

```text
+-------------------+        +------------------+        +----------------------+
| Human Reviewer    | <----> | CLI Runner       | <----> | LangGraph App        |
| (approve/reject)  |        | (invoke/resume)  |        | (state machine)      |
+-------------------+        +------------------+        +----------+-----------+
                                                                     |
                                                                     v
                                                          +----------------------+
                                                          | LangChain Tools      |
                                                          | requirements/design/ |
                                                          | implementation/...   |
                                                          +----------+-----------+
                                                                     |
                                                                     v
                                                          +----------------------+
                                                          | OpenAI Chat Model    |
                                                          +----------------------+
```

---

## 3. High-Level Flow

The SDLC process is strictly sequenced:

1. `requirements`
2. `design`
3. `implementation`
4. `testing`
5. `security`
6. `final_report`

After each generated output:
- Workflow **interrupts**
- Human can `approve`, `reject`, or `edit`
- Graph resumes from checkpoint and continues

---

## 4. LangGraph Topology

```text
              +----------------+
              | execute_step   |
              +--------+-------+
                       |
                       v
              +----------------+
              | approval_gate  | --(interrupt)-> human input
              +--------+-------+
                       |
                       v
              +----------------+
              | advance_step   |
              +---+--------+---+
                  |        |
             done |        | not done
                  v        v
                 END   execute_step
```

### Node Responsibilities

#### `execute_step`
- Reads `current_step` from state
- Calls corresponding tool
- Writes output to `last_output`
- If failure occurs, writes `error`

#### `approval_gate`
- If `error`, asks human to `retry` or `abort`
- If normal output, interrupts for:
  - `approve`
  - `reject` (with feedback)
  - `edit` (manual override)
- Updates canonical state (`approved`, `feedback`)

#### `advance_step`
- Moves to next phase only if current step approved
- Marks `done=True` at final step

---

## 5. State Model

Canonical state (TypedDict):

```python
class SDLCState(TypedDict, total=False):
    project_idea: str
    current_step: str
    last_output: str
    approved: Dict[str, str]
    feedback: Dict[str, str]
    done: bool
    error: Optional[str]
```

### Field Semantics

- `project_idea`: user-provided product concept
- `current_step`: active SDLC phase
- `last_output`: latest generated draft
- `approved`: authoritative accepted content by phase
- `feedback`: rejection guidance to improve next attempt
- `done`: termination flag
- `error`: exception text from tool execution

---

## 6. Tooling Layer (`@tool`)

Each SDLC phase is implemented as a dedicated tool:

- `t_requirements(...)`
- `t_design(...)`
- `t_implementation(...)`
- `t_testing(...)`
- `t_security(...)`
- `t_final_report(...)`

### Why tools?
- Clear modular boundaries
- Easy to test independently
- Prompt ownership per stage
- Replaceable implementation (swap LLM/provider later)

---

## 7. Human-in-the-Loop Interrupt Contract

`approval_gate` calls `interrupt(payload)` with metadata:

```json
{
  "type": "approval",
  "step": "design",
  "preview": "...",
  "expected": {
    "approve": {"action":"approve"},
    "reject": {"action":"reject","feedback":"..."},
    "edit": {"action":"edit","content":"..."}
  }
}
```

Resume payload via `Command(resume=...)`.

### Decision Outcomes

- **approve**
  - `approved[current_step] = last_output`
  - clear step feedback
- **reject**
  - store feedback
  - same step reruns with feedback
- **edit**
  - user content becomes approved output directly

---

## 8. Sequence Diagram (Approval Path)

```text
User          Runner            Graph:execute_step         Graph:approval_gate
 |              |                      |                           |
 | start idea   | invoke(initial)      |                           |
 |------------->|--------------------->| generate stage output     |
 |              |                      |------------+              |
 |              |                      |            v              |
 |              |                      |        last_output        |
 |              |                      |-------------------------->|
 |              |                      |                           | interrupt(payload)
 |              |<-----------------------------------------------  |
 | review       |                                               wait
 | approve      | invoke(Command(resume={"action":"approve"}))     |
 |------------->|-------------------------------------------------->|
 |              |                                                   | store approved
 |              |<--------------------------------------------------|
 |              | continue loop                                     |
```

---

## 9. Sequence Diagram (Reject + Regenerate)

```text
User -> Runner: reject + feedback
Runner -> Graph: Command(resume={"action":"reject","feedback":"add API details"})
Graph approval_gate: save feedback for step
Graph advance_step: step not approved -> stay same step
Graph execute_step: rerun same tool with feedback
Graph approval_gate: interrupt again with new output
```

---

## 10. Determinism & Guardrails

### Enforced controls
- Strict step order (`STEPS` array)
- No skipping phases
- Final report blocked until all prior phases approved
- Human gate on every stage
- Retry/abort behavior for errors

### Practical impact
- Predictable run behavior
- Better auditability
- Reduced accidental hallucinated flow transitions

---

## 11. Checkpointing and Resume

Current implementation uses:

- `MemorySaver` (in-memory checkpoint backend)
- `thread_id` in `configurable` scope

### Limitation
- In-memory checkpoints are not durable across process restarts.

### Production recommendation
- Replace with persistent checkpointer (Redis / Postgres / custom store)
- Include run metadata (user, timestamp, project id, policy version)

---

## 12. Error Handling Model

Error path:
1. Tool raises exception
2. `execute_step` stores `error`
3. `approval_gate` interrupts with error payload
4. Human chooses:
   - `retry` (clear error and rerun)
   - `abort` (terminate run)

---

## 13. Security Considerations

- Keep `OPENAI_API_KEY` in env vars / secrets manager
- Avoid logging sensitive input/output by default
- Add PII scrubbing before persistence/logging
- Protect approval endpoints if exposed via web API
- Add RBAC around who can approve/reject/edit

---

## 14. Scalability Considerations

Current CLI version is single-session/single-runner oriented.

To scale:
- Move runner to API service
- Use persistent checkpoint + job queue
- Store artifacts in object store / DB
- Add observability (traces, latency, token usage)
- Add retries and model fallbacks

---

## 15. Extensibility

Easy extension points:
- Add `compliance` stage (SOC2, HIPAA)
- Add `cost_estimation` stage
- Add multi-model routing per stage
- Add policy engine for automatic rejection criteria
- Add UI workflow for reviewers

---

## 16. Trade-offs

### Chosen
- Deterministic staged orchestration + human gates

### Trade-off
- Slower than fully autonomous agents
- But much better control, trust, and reviewability

---

## 17. Production Hardening Checklist

- [ ] Persistent checkpointer
- [ ] Structured logs + tracing
- [ ] Prompt/version registry
- [ ] Tool-level unit tests
- [ ] Integration tests for interrupt/resume
- [ ] Access control for approvals
- [ ] PII/content policy filters
- [ ] Token/cost budget enforcement
- [ ] SLOs + alerting

---

## 18. Summary

This architecture combines:
- **agentic generation power**
- **state-machine reliability**
- **human governance**

Result: a practical, controllable AI SDLC pipeline suitable for incremental production adoption.
```
