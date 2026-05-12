# Nutrition AI Assistant — Revised Design Spec

## Status

Revision of `docs/architecture/PLAN.md`. This document captures design decisions made during brainstorming review. The original plan remains the source of truth for architecture, schemas, and project structure — this spec records what changed and why.

## Decisions

### 1. LLM: Ollama + Qwen 2.5 7B (unchanged, with mitigations)

Free-only is a hard constraint. The 7B model stays, but every LLM interaction is designed with failure in mind.

**Mitigations added:**

- **Structured output validation**: Every LLM response is parsed through Pydantic models (`ExtractedFoodItem`, `MealExtraction`, intent enum). If validation fails, retry once with the error message appended to the prompt. After 2 failures, return a friendly fallback message.
- **Few-shot examples in every prompt**: Each tool-calling prompt includes 2-3 complete input/output examples in Portuguese demonstrating correct tool usage. This is the single biggest lever for 7B reliability.
- **Circuit breaker from day one**: Max 5 tool call iterations per user message. Moved from Week 3 polish to Week 1 core. Non-negotiable for 7B models that can loop.
- **30-second timeout per LLM call**: If Ollama hangs, fail gracefully with "Estou com problemas tecnicos, tente novamente em alguns minutos."

### 2. Data source: TACO only for MVP

Removed USDA and Open Food Facts from the initial build.

**What this means:**

- 597 Brazilian foods, already in Portuguese
- No cross-source schema normalization
- `data_source` field stays in the `food_items` DB schema for future expansion
- Drop `name_en` from embedding documents — not needed for a Portuguese-only pipeline
- Simpler `embedding_text`: `name_pt + synonyms + category + preparation_method`
- 597 foods is small enough to re-seed ChromaDB in seconds during iteration

**Adding USDA later** requires only a new ingestion script and re-seeding ChromaDB. The architecture supports it without changes.

### 3. Intent classifier: kept (for learning value)

The separate intent classification node stays in the LangGraph graph despite adding latency. The learning value of the multi-node pattern outweighs the efficiency cost.

**Constraint**: The classifier uses constrained structured output (enum of intents: `log_food`, `query_history`, `edit_meal`, `general`), not free-form reasoning. Few-shot examples are critical here.

### 4. Error handling flows (new)

These are defined behaviors for each failure mode — no new components, just specified responses.

| Failure | Behavior |
|---|---|
| ChromaDB similarity < 0.75 | LLM asks user to rephrase. Example: "Nao encontrei esse alimento. Pode descrever de outra forma?" |
| ChromaDB similarity 0.75-0.90 | LLM picks from top 3 candidates (already in plan) |
| Invalid tool call (bad params/JSON) | Pydantic catches it, retry once with error appended to prompt. Second failure → "Desculpa, tive um problema. Pode repetir?" |
| Tool execution fails (DB error) | Log error, respond to user, roll back partial state. No half-logged meals. |
| Ollama unreachable | Health check on startup. Mid-conversation failure → "Estou com problemas tecnicos, tente novamente em alguns minutos." |
| Non-food message ("oi", "obrigado") | Fallback node handles with friendly response, no tool calls (already in plan) |

### 5. Timeline adjustments

Same 4-week structure, reordered to address reliability early.

**Week 1: Foundation + CLI Chatbot**

- Days 1-2: Project scaffolding (unchanged)
- Days 3-4: TACO-only ingestion (simpler, faster). Use saved time to write few-shot examples for each tool.
- Days 5-7: LangGraph + CLI. Circuit breaker and Pydantic validation built in from day one.

**Week 2: RAG Pipeline + Conversation Memory + API**

- Days 8-9: RAG refinement — TACO only, no cross-source work. Add rate limiting (moved from Week 3).
- Days 10-11: Conversation memory + corrections (unchanged)
- Days 12-14: FastAPI server (unchanged)

**Week 3: WhatsApp Integration**

- Days 15-16: Meta Cloud API setup (unchanged)
- Days 17-18: WhatsApp message flow (unchanged)
- Days 19-21: Testing and edge cases. Error handling is already defined (Section 4), so this is testing, not designing on the fly.

**Week 4 (Optional): Observability + Deployment** (unchanged)

## What stays unchanged from PLAN.md

- Architecture diagram and layer structure
- Database schema (all tables, indexes, fields)
- Project file structure
- RAG pipeline design (ingestion and retrieval flow)
- Agent tools (5 tools, same signatures)
- Orchestration flow (the 10-step example)
- Structured output schemas
- Conversation memory strategy
- System prompt
- Verification plan
- Security considerations
- Dependencies list
- Future improvements list
