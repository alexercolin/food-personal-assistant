# Personal Nutrition AI Assistant — Implementation Plan

## Context

Build a WhatsApp-based personal nutrition assistant as an educational project to learn AI application development. The user is a frontend engineer (intermediate Python) who wants to understand RAG pipelines, LLM orchestration, agent workflows, tool calling, and AI observability through hands-on building.

**Constraints**: Portuguese (pt-BR) interaction, free-only budget (Ollama + free tiers), fast timeline (2-4 weeks).

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    INGRESS LAYER                              │
│  WhatsApp (Meta Cloud API) ──► Webhook Endpoint               │
│  CLI Interface (dev/testing) ──► Direct Function Call          │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    API LAYER (FastAPI)                         │
│  POST /webhook/whatsapp    — incoming messages                │
│  GET  /webhook/whatsapp    — Meta verification                │
│  POST /api/chat            — direct API (CLI/testing)         │
│  GET  /api/meals/{user_id} — meal history                     │
│  GET  /api/stats/{user_id} — nutrition summaries              │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              ORCHESTRATION LAYER (LangGraph)                   │
│                                                               │
│  ┌──────────────┐   ┌───────────────┐   ┌────────────────┐  │
│  │ Intent        │──►│ Food Logging  │──►│ Nutrition      │  │
│  │ Classifier    │   │ Agent         │   │ Lookup Tool    │  │
│  └──────┬───────┘   └───────────────┘   └────────────────┘  │
│         │           ┌───────────────┐   ┌────────────────┐  │
│         ├──────────►│ Query Agent   │──►│ SQL Query Tool │  │
│         │           └───────────────┘   └────────────────┘  │
│         │           ┌───────────────┐                        │
│         └──────────►│ Fallback      │                        │
│                     └───────────────┘                        │
│                                                               │
│  Conversation Memory: last N messages per user (from DB)      │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    DATA ACCESS LAYER                           │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │ SQLite          │  │ TACO/USDA       │  │ ChromaDB     │  │
│  │ (Users, Meals,  │  │ Nutrition Data  │  │ (Food fuzzy  │  │
│  │  Conversations) │  │ (seed JSON)     │  │  matching)   │  │
│  └────────────────┘  └─────────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### RAG vs Tool Calling — When to Use Which

| Scenario | Pattern | Why |
|---|---|---|
| "Comi frango grelhado" — identify the food | **RAG** (vector search) | Fuzzy semantic matching for food names, synonyms, regional terms |
| "Quantas calorias comi hoje?" | **Tool calling** (SQL query) | Structured aggregation over relational data |
| "Apaga o almoco de hoje" | **Tool calling** (DB mutation) | CRUD operation on meals table |
| "Comi um Big Mac" | **RAG** (vector search) | Search branded foods database |

**Principle**: RAG resolves food identity (what IS this food nutritionally). Tool calling operates on user data (CRUD, aggregations). They are complementary.

---

## Tech Stack

### LLM: Ollama with Qwen 2.5 7B

**Why Qwen 2.5 7B** over Llama 3.1 8B:
- Superior multilingual support — trained on significantly more Portuguese data
- Better structured output and tool calling compared to Llama at the 7B scale
- Runs well on Apple Silicon (M-series Macs) with ~5GB RAM
- Active development with frequent improvements

**Alternatives considered**:
- Llama 3.1 8B: Good but slightly weaker Portuguese performance at the 7B tier
- Gemma 2 9B: Good multilingual but heavier (9B) and slightly worse tool calling
- Cloud APIs (Claude, GPT): Better quality but not free — can upgrade later if needed

### Orchestration: LangChain Core + LangGraph

**Why this combination**:
- **LangChain Core** provides the chat model abstraction layer — swap Ollama for Claude/OpenAI with one config change. This is the only part of LangChain we use.
- **LangGraph** provides the stateful agent graph — intent classification branching to food-logging or query-answering nodes. This is the current industry direction for agent design.
- **Why not plain SDK**: You'd rebuild state management, tool dispatch, and conversation threading from scratch. Educational but slow for a 2-4 week timeline.
- **Why not LangChain chains**: Being superseded by LangGraph. Chains are too rigid for conditional branching.

### Vector Database: ChromaDB

**Why**: Embeds in your Python process, zero infrastructure, SQLite backend, handles 10K food items easily. Perfect for a free, local project.

**Alternatives**: Qdrant (overkill for this scale), Pinecone (paid), pgvector (requires PostgreSQL) — all add complexity without benefit here.

### Relational Database: SQLite + SQLAlchemy

**Why SQLite**: Zero setup, file-based, free. For a single-user personal assistant, SQLite handles the workload perfectly. SQLAlchemy ORM means migrating to PostgreSQL later requires changing one connection string.

### Embeddings: paraphrase-multilingual-MiniLM-L12-v2

**Why this model**: 384 dimensions, supports 50+ languages including Portuguese, runs locally on CPU, free. Critical for matching "frango grelhado" to "chicken breast, grilled" and "mandioca" to "cassava".

**Alternative**: `intfloat/multilingual-e5-small` — slightly better Portuguese performance but larger. Either works.

### Nutritional Data: TACO + USDA

**Primary: TACO (Tabela Brasileira de Composicao de Alimentos)**
- Brazilian government database — 597 foods with Brazilian cooking methods
- Food names already in Portuguese
- Covers regional foods: mandioca, acai, feijao preto, carne seca, etc.
- Free, public domain

**Secondary: USDA FoodData Central (SR Legacy)**
- 8,000+ foods with detailed nutrient profiles
- Broader international coverage for foods not in TACO
- Free API + downloadable CSV dumps

**Supplementary: Open Food Facts**
- 3M+ branded/packaged products including Brazilian products
- Covers items like "Nestle Ninho", "Guarana Antarctica", etc.

### WhatsApp: Meta Cloud API

**Why**: Free for 1,000 conversations/month (more than enough for personal use). Official, reliable, no per-message cost.

**Dev setup**: ngrok to expose local FastAPI server to Meta's webhooks. Meta provides a sandbox test phone number.

### Observability: Langfuse (free tier)

**Why**: Open-source, 50K observations/month free, works with any LLM framework, built-in prompt management and evaluation. The `@observe` decorator is trivial to add.

---

## Project Structure

```
food-personal-assistant/
├── pyproject.toml
├── .env.example
├── .env                           # NEVER committed
├── .gitignore
├── README.md
├── Makefile                       # make dev, make test, make seed
│
├── data/
│   ├── taco_foods.json            # TACO database (Portuguese foods)
│   ├── usda_foods.json            # USDA common foods
│   └── prompts/
│       ├── system_prompt.txt
│       ├── food_extraction.txt
│       └── few_shot_examples.json
│
├── scripts/
│   ├── seed_food_db.py            # Ingest TACO + USDA into ChromaDB
│   └── test_whatsapp.py           # Manual webhook testing
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI app factory, lifespan
│   │   ├── dependencies.py        # DI: DB sessions, services
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── webhook.py         # WhatsApp webhook endpoints
│   │       ├── chat.py            # POST /api/chat (CLI/API)
│   │       └── meals.py           # GET meals, stats
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py               # LangGraph graph definition
│   │   ├── state.py               # LangGraph state schema
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── intent_classifier.py
│   │   │   ├── food_logger.py
│   │   │   ├── query_handler.py
│   │   │   └── fallback.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── nutrition_lookup.py
│   │   │   ├── food_search.py     # ChromaDB vector search
│   │   │   ├── meal_logger.py
│   │   │   └── stats_query.py
│   │   └── prompts.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings
│   │   └── llm.py                 # LLM provider factory
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── engine.py              # SQLAlchemy engine/session
│   │   ├── models.py              # ORM models
│   │   └── repositories/
│   │       ├── __init__.py
│   │       ├── meal_repo.py
│   │       └── conversation_repo.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── nutrition_service.py
│   │   ├── meal_service.py
│   │   └── whatsapp_service.py
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── store.py               # ChromaDB client
│   │   └── embeddings.py          # Embedding model wrapper
│   │
│   └── cli/
│       └── chat.py                # Interactive terminal chat
│
├── tests/
│   ├── conftest.py
│   ├── test_agents/
│   ├── test_api/
│   └── test_services/
│
└── notebooks/
    ├── 01_food_extraction.ipynb
    ├── 02_embeddings_test.ipynb
    └── 03_langgraph_prototype.ipynb
```

---

## Database Schema

```sql
CREATE TABLE users (
    id              TEXT PRIMARY KEY,         -- UUID
    phone_number    TEXT UNIQUE NOT NULL,     -- E.164: +5511999999999
    display_name    TEXT,
    timezone        TEXT DEFAULT 'America/Sao_Paulo',
    daily_calorie_goal  INTEGER,
    daily_protein_goal  REAL,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE meals (
    id              TEXT PRIMARY KEY,         -- UUID
    user_id         TEXT NOT NULL REFERENCES users(id),
    meal_type       TEXT,                     -- cafe_da_manha, almoco, jantar, lanche
    description     TEXT NOT NULL,            -- Raw user input
    logged_at       TEXT DEFAULT (datetime('now')),
    total_calories  REAL DEFAULT 0,
    total_protein   REAL DEFAULT 0,
    total_carbs     REAL DEFAULT 0,
    total_fat       REAL DEFAULT 0
);

CREATE INDEX idx_meals_user_logged ON meals(user_id, logged_at DESC);

CREATE TABLE food_items (
    id              TEXT PRIMARY KEY,
    meal_id         TEXT NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
    food_name       TEXT NOT NULL,            -- Canonical: "banana, crua"
    original_text   TEXT,                     -- User typed: "2 bananas"
    quantity        REAL NOT NULL,
    unit            TEXT NOT NULL,            -- "g", "ml", "unidade", "fatia", "xicara"
    serving_grams   REAL,
    calories        REAL NOT NULL,
    protein         REAL NOT NULL,
    carbs           REAL NOT NULL,
    fat             REAL NOT NULL,
    fiber           REAL,
    data_source     TEXT,                     -- "taco", "usda", "openfoodfacts"
    confidence      REAL,                     -- 0.0-1.0
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE conversations (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    role            TEXT NOT NULL,            -- 'user' or 'assistant'
    content         TEXT NOT NULL,
    metadata        TEXT,                     -- JSON string for tool calls, intent, etc.
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_conversations_user ON conversations(user_id, created_at DESC);

CREATE TABLE user_corrections (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    original_text   TEXT NOT NULL,
    corrected_to    TEXT NOT NULL,
    correction_type TEXT NOT NULL,            -- 'food_identity', 'quantity', 'preparation'
    created_at      TEXT DEFAULT (datetime('now'))
);
```

**Key design decisions**:
- **Denormalized totals on `meals`**: "Quantas calorias comi hoje?" hits meals table directly, no join needed
- **`original_text` on food_items**: Debug why "arroz" mapped to 150 calories
- **`confidence` score**: Flag low-confidence matches for user confirmation
- **`user_corrections`**: Learn from user corrections over time

---

## RAG Pipeline Design

### Ingestion

```
1. Download TACO CSV + USDA SR Legacy CSV
2. Normalize each food into:
   {
     "source": "taco",
     "id": "taco_001",
     "name_pt": "Arroz, integral, cozido",
     "name_en": "Brown rice, cooked",
     "synonyms": ["arroz integral", "arroz marrom"],
     "category": "Cereais",
     "nutrients_per_100g": {
       "calories": 124, "protein": 2.6, "carbs": 25.8,
       "fat": 1.0, "fiber": 2.7
     },
     "common_portions": [
       {"description": "1 colher de sopa", "grams": 25},
       {"description": "1 xicara", "grams": 160},
       {"description": "1 escumadeira", "grams": 90}
     ],
     "embedding_text": "arroz integral cozido brown rice cereal grao whole grain"
   }
3. Generate embedding_text: name_pt + name_en + synonyms + category
4. Embed with paraphrase-multilingual-MiniLM-L12-v2 (384 dims)
5. Upsert into ChromaDB with full metadata
```

### Retrieval

When user says "frango grelhado":

```
1. Embed query "frango grelhado" (same model)
2. Cosine similarity search in ChromaDB (top 5 results)
3. If top result similarity > 0.90 → use directly
4. If 0.75 < similarity < 0.90 → return top 3 to LLM for disambiguation
5. If similarity < 0.75 → ask user for clarification
```

---

## Agent Tools

The LLM has access to 5 tools:

| Tool | Purpose | When Used |
|---|---|---|
| `search_food_database` | Vector search in ChromaDB for food matching | "Comi frango grelhado" |
| `log_meal` | Save parsed meal to SQLite | After nutrition data is resolved |
| `query_meal_history` | Query meal history with date ranges and aggregations | "Quantas calorias hoje?" |
| `update_meal` | Correct a previously logged meal | "Na verdade foram 200g" |
| `delete_meal` | Remove a meal entry | "Apaga o almoco" |

### Orchestration Flow (Example)

User sends: "Almocei 150g de frango grelhado com arroz"

```
Step 1: WhatsApp webhook receives POST → respond 200 immediately
Step 2: Load last 10 conversation messages for context
Step 3: LangGraph intent classifier → "log_food"
Step 4: Food logger node calls LLM with tools
Step 5: LLM calls search_food_database("frango grelhado") + search_food_database("arroz branco cozido")
Step 6: Execute both searches in parallel → return nutritional data
Step 7: LLM calculates portions and calls log_meal({
          meal_type: "almoco",
          items: [
            {food_name: "Frango, peito, grelhado", quantity_grams: 150, calories: 247.5, ...},
            {food_name: "Arroz, branco, cozido", quantity_grams: 200, calories: 260, ...}
          ]
        })
Step 8: Save to SQLite → return meal_id
Step 9: LLM generates response:
        "Registrado seu almoco!
         Frango grelhado (150g): 248 cal | 47g prot | 0g carb | 5g gord
         Arroz branco (200g): 260 cal | 5g prot | 56g carb | 1g gord
         Total: 508 cal | 52g prot | 56g carb | 6g gord
         Nota: estimei o arroz em 200g (~1 xicara). Quer ajustar?"
Step 10: Send response via WhatsApp API
Step 11: Save conversation turn to DB
```

---

## Structured Output Schemas

```python
class ExtractedFoodItem(BaseModel):
    food_name: str           # Normalized name
    original_text: str       # What the user typed
    quantity: float | None
    unit: str | None         # "g", "ml", "unidade", "fatia"
    quantity_grams: float | None
    quantity_confidence: Literal["exact", "estimated", "unknown"]
    preparation_method: str | None  # grelhado, frito, cozido, cru

class MealExtraction(BaseModel):
    items: list[ExtractedFoodItem]
    meal_type: str | None    # cafe_da_manha, almoco, jantar, lanche
    is_correction: bool = False
    ambiguities: list[str] = []
```

---

## Conversation Memory Strategy

**Short-term**: Pass last 10 messages from `conversations` table into the LLM context.

**Long-term**: Store user preferences, dietary goals, and food corrections in the database. Inject a summary into the system prompt.

**What to remember**:
- User dietary goals (set via conversation)
- Food corrections ("quando digo 'coxa' quero dizer coxa de frango")
- Dietary restrictions

**Implementation**: Custom SQL queries, not LangChain memory modules. Loading 10 messages from SQLite is simpler and more educational than LangChain's abstraction.

---

## System Prompt (Portuguese)

```
Voce e um assistente pessoal de nutricao que ajuda o usuario a registrar suas
refeicoes via WhatsApp. Voce e amigavel, conciso e preciso.

## Capacidades
- Registrar refeicoes a partir de descricoes em linguagem natural
- Consultar informacoes nutricionais de qualquer alimento
- Responder perguntas sobre historico alimentar (totais diarios, tendencias semanais)
- Ajudar o usuario a entender seu consumo em relacao as suas metas

## Regras
1. SEMPRE busque no banco de alimentos antes de registrar. Nunca invente valores.
2. Quando nao houver quantidade, estime uma porcao razoavel e AVISE o usuario.
3. Quando houver ambiguidade, escolha a opcao mais comum e mencione alternativas.
4. Formate para WhatsApp: use quebras de linha, mantenha respostas curtas.
5. Infira o tipo de refeicao pelo horario se nao for dito:
   - Antes das 10:00 → cafe da manha
   - 10:00-14:00 → almoco
   - 14:00-17:00 → lanche
   - 17:00-21:00 → jantar
   - Apos 21:00 → lanche
6. Sempre mostre o detalhamento por item E o total apos registrar.
7. Para correcoes, encontre a refeicao mais recente e atualize.
8. Nunca de conselhos medicos ou dieteticos. Voce registra, nao prescreve.

## Perfil do Usuario
Nome: {user_name}
Meta diaria de calorias: {calorie_goal} kcal
Meta diaria de proteina: {protein_goal}g
Restricoes: {restrictions}
Fuso: {timezone}
Correcoes conhecidas: {corrections_summary}

## Contexto
Data/hora atual: {current_datetime}
Totais de hoje ate agora: {today_totals}
```

---

## Implementation Roadmap

### Week 1: Foundation + CLI Chatbot

**Days 1-2: Project Scaffolding**
- Initialize project: `pyproject.toml` with uv, virtual environment
- Set up Ollama + pull Qwen 2.5 7B model
- Create SQLite database with SQLAlchemy models
- Create `.env`, `.gitignore`, basic config
- Initialize git repository

**Days 3-4: Food Parsing + Nutrition Data**
- Download and process TACO database into `data/taco_foods.json`
- Download USDA SR Legacy common foods into `data/usda_foods.json`
- Build ChromaDB ingestion script (`scripts/seed_food_db.py`)
- Embed foods with multilingual model, upsert to ChromaDB
- Test vector search with Portuguese food queries

**Days 5-7: CLI Chatbot**
- Implement LLM provider factory (`src/core/llm.py`) — Ollama integration
- Define tools (search_food_database, log_meal, query_meal_history)
- Build basic LangGraph graph with intent classification
- Implement food logging flow: parse → search → calculate → store
- Implement query flow: "quantas calorias hoje?" → SQL query
- Build interactive CLI chat loop (`src/cli/chat.py`)
- Test end-to-end: message in → tool calls → DB write → response

**AI concepts learned**: structured output, tool calling, prompt engineering, embeddings, vector similarity search, LangGraph state machines

### Week 2: RAG Pipeline + Conversation Memory

**Days 8-9: RAG Refinement**
- Add Portuguese synonym mappings for common foods
- Improve embedding text quality (add preparation methods, regional names)
- Implement confidence thresholds with user clarification flow
- Handle compound foods ("arroz com feijao" → two items)
- Build evaluation dataset: 30+ food queries with expected results
- Test and iterate on retrieval quality (target Hit Rate@5 > 90%)

**Days 10-11: Conversation Memory**
- Implement conversation history storage (save/load from SQLite)
- Inject last 10 messages into LLM context
- Handle corrections: "na verdade foram 200g, nao 150g"
- Handle meal editing and deletion
- Store user corrections for future learning

**Days 12-14: FastAPI Server**
- Build FastAPI application with routes
- POST /api/chat — direct chat endpoint
- GET /api/meals — meal history
- GET /api/stats — daily/weekly nutrition summaries
- Add request/response validation with Pydantic
- Test API with httpie/curl

**AI concepts learned**: RAG pipeline (ingest → embed → retrieve → augment), conversation memory, few-shot prompting, evaluation metrics

### Week 3: WhatsApp Integration

**Days 15-16: Meta Cloud API Setup**
- Create Meta Business account + WhatsApp Business app
- Set up test phone number (Meta provides sandbox)
- Configure webhook URL with ngrok
- Implement webhook verification (GET endpoint)
- Implement message receiving (POST endpoint)

**Days 17-18: WhatsApp Message Flow**
- Wire webhook to agent graph
- Implement async message processing (respond 200 immediately, process in background)
- Implement WhatsApp message sending (reply to user)
- Handle WhatsApp-specific formatting
- Test full flow: send WhatsApp message → get nutritional response

**Days 19-21: Polish + Edge Cases**
- Handle unknown foods gracefully (ask for clarification)
- Handle non-food messages (friendly redirect)
- Handle multiple foods in one message
- Add rate limiting per user
- Add basic error handling and logging
- Implement circuit breaker for runaway tool loops (max 5 iterations)

**AI concepts learned**: webhook architecture, async processing, prompt injection defense, production error handling

### Week 4 (Optional): Observability + Deployment

**Days 22-23: Langfuse Integration**
- Set up Langfuse free cloud account
- Add `@observe` decorators to agent functions
- Track: latency, token usage, tool calls, costs
- Build food extraction eval suite (50+ test cases)
- Run evals and iterate on prompts

**Days 24-25: Deployment**
- Create Dockerfile + docker-compose.yml
- Deploy to Railway.app free tier (or Render.com)
- Configure production webhook URL (replace ngrok)
- Set up health check endpoint
- Test WhatsApp with deployed backend

**Days 26-28: Advanced Features (Stretch Goals)**
- User goal setting: "minha meta e 2000 calorias por dia"
- Daily summary: "como foi minha alimentacao hoje?"
- Weekly trends: "qual foi minha refeicao mais calorica da semana?"
- User preference learning from corrections

**AI concepts learned**: LLM observability, tracing, evaluation pipelines, containerization, production deployment

---

## Verification Plan

### After Week 1 (CLI works)
- Run `python -m src.cli.chat`
- Type "Comi 2 bananas" → verify it logs to DB with correct calories (~210 kcal)
- Type "Almocei 150g de frango com arroz" → verify two food items logged
- Type "Quantas calorias comi hoje?" → verify correct daily total
- Check SQLite DB directly to confirm data integrity

### After Week 2 (API works)
- `curl -X POST localhost:8000/api/chat -d '{"message": "Comi acai"}'` → verify fuzzy match works
- Type "na verdade foram 200g" after logging → verify meal gets updated
- Query history endpoint → verify aggregations are correct

### After Week 3 (WhatsApp works)
- Send real WhatsApp message to bot number → get nutritional response
- Test compound food: "almocei arroz, feijao e bife" → 3 items logged
- Test correction: "na verdade era frango, nao bife" → verify update
- Test query: "quantas calorias ja comi hoje?" → correct total

---

## Key Dependencies (all free)

```toml
[project]
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "langchain-core>=0.3.0",
    "langchain-community>=0.3.0",
    "langgraph>=0.2.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "httpx>=0.27.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "jupyter>=1.0.0",
]
observability = [
    "langfuse>=2.0.0",
]
```

---

## Security Considerations

- **Prompt injection**: WhatsApp is public-facing. Use chat model roles (system vs user) to separate instructions from user input. Validate LLM outputs before acting.
- **Webhook verification**: Verify `X-Hub-Signature-256` on every incoming Meta webhook request.
- **Secrets**: `.env` in `.gitignore` from the first commit. Never hardcode API keys.
- **Rate limiting**: 30 messages/hour per phone number via FastAPI middleware.
- **Data privacy**: Food intake is health-related data. Add a "apagar meus dados" command.

---

## Future Improvements (After v1)

1. **Cloud LLM upgrade**: Swap Ollama for Claude Haiku when budget allows — better Portuguese, faster, more reliable tool calling
2. **PostgreSQL migration**: Change SQLAlchemy connection string when moving to multi-user
3. **Photo recognition**: User sends a photo of food → use a vision model to identify items
4. **Barcode scanning**: Send photo of product barcode → Open Food Facts lookup
5. **Proactive summaries**: Daily WhatsApp message at 9pm with the day's nutrition summary
6. **Goal tracking**: Progress toward calorie/macro targets with visual charts
7. **Multi-language**: Support English alongside Portuguese based on message detection
