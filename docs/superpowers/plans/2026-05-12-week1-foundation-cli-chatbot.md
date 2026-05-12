# Week 1: Foundation + CLI Chatbot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working CLI chatbot that can log meals from Portuguese natural language input, look up nutrition data from the TACO database via ChromaDB vector search, store meals in SQLite, and answer questions about meal history.

**Architecture:** FastAPI app shell (for later use) with a LangGraph agent graph containing an intent classifier node that routes to food_logger, query_handler, or fallback nodes. The food_logger uses ChromaDB RAG for food identification and SQLite for storage. An interactive CLI chat loop drives the agent for testing. Every LLM call goes through Pydantic validation with retry logic and a circuit breaker.

**Tech Stack:** Python 3.13, uv (package manager), Ollama + Qwen 2.5 7B, LangChain Core, LangGraph, ChromaDB, sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2), SQLAlchemy + SQLite, Pydantic, FastAPI (shell only), pytest

---

## Prerequisites

Before starting Task 1, ensure these are installed:

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Ollama
brew install ollama

# Start Ollama and pull the model
ollama serve &
ollama pull qwen2.5:7b
```

Verify both work:
```bash
uv --version        # expect: uv 0.x.x
ollama run qwen2.5:7b "Diga oi"  # expect: a Portuguese greeting
```

---

## File Structure

```
food-personal-assistant/
├── pyproject.toml                     # Project config, dependencies, scripts
├── .env.example                       # Template for environment variables
├── .env                               # Actual env vars (gitignored)
├── .gitignore
├── Makefile                           # make dev, make test, make seed
├── data/
│   ├── taco_foods.json                # TACO database (597 Brazilian foods)
│   └── prompts/
│       ├── system_prompt.txt          # Main system prompt (Portuguese)
│       ├── intent_classifier.txt      # Intent classification prompt
│       ├── food_extraction.txt        # Food parsing prompt
│       └── few_shot_examples.json     # Tool calling examples for 7B
├── scripts/
│   └── seed_food_db.py                # Ingest TACO into ChromaDB
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                  # Pydantic Settings (env vars)
│   │   └── llm.py                     # Ollama/LangChain chat model factory
│   ├── db/
│   │   ├── __init__.py
│   │   ├── engine.py                  # SQLAlchemy engine + session factory
│   │   └── models.py                  # ORM models (users, meals, food_items, conversations)
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── embeddings.py              # Sentence-transformers embedding wrapper
│   │   └── store.py                   # ChromaDB client (search + upsert)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py                   # LangGraph state schema
│   │   ├── graph.py                   # LangGraph graph definition + compilation
│   │   ├── schemas.py                 # Pydantic models for LLM structured output
│   │   ├── safety.py                  # Circuit breaker, retry logic, timeout
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── intent_classifier.py   # Classifies user intent → route
│   │       ├── food_logger.py         # Extracts food, searches ChromaDB, logs meal
│   │       ├── query_handler.py       # Answers questions about meal history
│   │       └── fallback.py            # Handles non-food messages
│   ├── services/
│   │   ├── __init__.py
│   │   ├── nutrition_service.py       # ChromaDB food search + nutrition lookup
│   │   └── meal_service.py            # Meal CRUD operations
│   └── cli/
│       └── chat.py                    # Interactive terminal chat loop
└── tests/
    ├── conftest.py                    # Shared fixtures (test DB, mock LLM)
    ├── test_db/
    │   └── test_models.py
    ├── test_vectorstore/
    │   └── test_store.py
    ├── test_agents/
    │   ├── test_schemas.py
    │   ├── test_safety.py
    │   ├── test_intent_classifier.py
    │   ├── test_food_logger.py
    │   └── test_query_handler.py
    └── test_services/
        ├── test_nutrition_service.py
        └── test_meal_service.py
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `.env`
- Create: `Makefile`
- Create: `src/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "food-personal-assistant"
version = "0.1.0"
description = "WhatsApp nutrition assistant powered by local LLM"
requires-python = ">=3.13"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "langchain-core>=0.3.0",
    "langchain-ollama>=0.3.0",
    "langgraph>=0.4.0",
    "chromadb>=1.0.0",
    "sentence-transformers>=3.0.0",
    "httpx>=0.27.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
]

[project.scripts]
chat = "src.cli.chat:main"
seed = "scripts.seed_food_db:main"

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# Virtual environment
.venv/

# Environment variables
.env

# Database
*.db
*.sqlite
*.sqlite3

# ChromaDB
chroma_data/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
```

- [ ] **Step 3: Create .env.example and .env**

`.env.example`:
```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# Database
DATABASE_URL=sqlite:///./food_assistant.db

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_data
CHROMA_COLLECTION_NAME=foods

# Embedding model
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# App
LOG_LEVEL=INFO
MAX_TOOL_ITERATIONS=5
LLM_TIMEOUT_SECONDS=30
```

`.env` gets the same content (it's the local copy).

- [ ] **Step 4: Create Makefile**

```makefile
.PHONY: setup dev test seed lint format

setup:
	uv sync --all-extras

dev:
	uv run uvicorn src.api.app:app --reload --port 8000

test:
	uv run pytest -v

seed:
	uv run python -m scripts.seed_food_db

lint:
	uv run ruff check .

format:
	uv run ruff format .

chat:
	uv run python -m src.cli.chat
```

- [ ] **Step 5: Create src/__init__.py**

```python
```

Empty file — just marks `src` as a package.

- [ ] **Step 6: Install dependencies**

Run: `cd /Users/alexercolinoliveira/Desktop/food-personal-assistant && uv sync --all-extras`
Expected: All dependencies installed, `.venv` created.

- [ ] **Step 7: Verify installation**

Run: `uv run python -c "import langchain_core; import langgraph; import chromadb; import sqlalchemy; print('All imports OK')"`
Expected: `All imports OK`

```bash
git add pyproject.toml .gitignore .env.example Makefile src/__init__.py uv.lock
git commit -m "chore: scaffold project with uv, dependencies, and makefile"
```

---

## Task 2: Configuration Module

**Files:**
- Create: `src/core/__init__.py`
- Create: `src/core/config.py`

- [ ] **Step 1: Create src/core/__init__.py**

```python
```

- [ ] **Step 2: Write src/core/config.py**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    database_url: str = "sqlite:///./food_assistant.db"

    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "foods"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    log_level: str = "INFO"
    max_tool_iterations: int = 5
    llm_timeout_seconds: int = 30


settings = Settings()
```

- [ ] **Step 3: Verify config loads**

Run: `uv run python -c "from src.core.config import settings; print(settings.ollama_model)"`
Expected: `qwen2.5:7b`

```bash
git add src/core/__init__.py src/core/config.py
git commit -m "feat: add pydantic settings config module"
```

---

## Task 3: Database Models and Engine

**Files:**
- Create: `src/db/__init__.py`
- Create: `src/db/engine.py`
- Create: `src/db/models.py`
- Create: `tests/conftest.py`
- Create: `tests/test_db/__init__.py` (empty)
- Create: `tests/test_db/test_models.py`

- [ ] **Step 1: Create src/db/__init__.py**

```python
```

- [ ] **Step 2: Write src/db/engine.py**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(bind=engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
```

- [ ] **Step 3: Write src/db/models.py**

```python
import uuid
from datetime import datetime, timezone

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def generate_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid)
    phone_number: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String)
    timezone: Mapped[str] = mapped_column(String, default="America/Sao_Paulo")
    daily_calorie_goal: Mapped[int | None] = mapped_column(Integer)
    daily_protein_goal: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[str] = mapped_column(String, default=utcnow)
    updated_at: Mapped[str] = mapped_column(String, default=utcnow, onupdate=utcnow)

    meals: Mapped[list["Meal"]] = relationship(back_populates="user")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")


class Meal(Base):
    __tablename__ = "meals"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    meal_type: Mapped[str | None] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    logged_at: Mapped[str] = mapped_column(String, default=utcnow)
    total_calories: Mapped[float] = mapped_column(Float, default=0)
    total_protein: Mapped[float] = mapped_column(Float, default=0)
    total_carbs: Mapped[float] = mapped_column(Float, default=0)
    total_fat: Mapped[float] = mapped_column(Float, default=0)

    user: Mapped["User"] = relationship(back_populates="meals")
    food_items: Mapped[list["FoodItem"]] = relationship(
        back_populates="meal", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("idx_meals_user_logged", "user_id", logged_at.desc()),)


class FoodItem(Base):
    __tablename__ = "food_items"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid)
    meal_id: Mapped[str] = mapped_column(ForeignKey("meals.id", ondelete="CASCADE"), nullable=False)
    food_name: Mapped[str] = mapped_column(String, nullable=False)
    original_text: Mapped[str | None] = mapped_column(String)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String, nullable=False)
    serving_grams: Mapped[float | None] = mapped_column(Float)
    calories: Mapped[float] = mapped_column(Float, nullable=False)
    protein: Mapped[float] = mapped_column(Float, nullable=False)
    carbs: Mapped[float] = mapped_column(Float, nullable=False)
    fat: Mapped[float] = mapped_column(Float, nullable=False)
    fiber: Mapped[float | None] = mapped_column(Float)
    data_source: Mapped[str | None] = mapped_column(String)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[str] = mapped_column(String, default=utcnow)

    meal: Mapped["Meal"] = relationship(back_populates="food_items")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[str] = mapped_column(String, default=utcnow)

    user: Mapped["User"] = relationship(back_populates="conversations")

    __table_args__ = (Index("idx_conversations_user", "user_id", created_at.desc()),)


class UserCorrection(Base):
    __tablename__ = "user_corrections"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    original_text: Mapped[str] = mapped_column(String, nullable=False)
    corrected_to: Mapped[str] = mapped_column(String, nullable=False)
    correction_type: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, default=utcnow)
```

- [ ] **Step 4: Write tests/conftest.py**

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()
```

- [ ] **Step 5: Write the failing test**

Create `tests/test_db/__init__.py` (empty) and `tests/test_db/test_models.py`:

```python
from src.db.models import FoodItem, Meal, User


def test_create_user(db_session):
    user = User(phone_number="+5511999999999", display_name="Test User")
    db_session.add(user)
    db_session.commit()

    saved = db_session.query(User).first()
    assert saved.phone_number == "+5511999999999"
    assert saved.display_name == "Test User"
    assert saved.timezone == "America/Sao_Paulo"
    assert saved.id is not None


def test_create_meal_with_food_items(db_session):
    user = User(phone_number="+5511999999999")
    db_session.add(user)
    db_session.commit()

    meal = Meal(
        user_id=user.id,
        meal_type="almoco",
        description="frango grelhado com arroz",
        total_calories=508,
        total_protein=52,
        total_carbs=56,
        total_fat=6,
    )
    db_session.add(meal)
    db_session.commit()

    item = FoodItem(
        meal_id=meal.id,
        food_name="Frango, peito, grelhado",
        original_text="frango grelhado",
        quantity=150,
        unit="g",
        serving_grams=150,
        calories=247.5,
        protein=47,
        carbs=0,
        fat=5,
        data_source="taco",
        confidence=0.95,
    )
    db_session.add(item)
    db_session.commit()

    saved_meal = db_session.query(Meal).first()
    assert saved_meal.total_calories == 508
    assert len(saved_meal.food_items) == 1
    assert saved_meal.food_items[0].food_name == "Frango, peito, grelhado"


def test_cascade_delete_food_items(db_session):
    user = User(phone_number="+5511999999999")
    db_session.add(user)
    db_session.commit()

    meal = Meal(user_id=user.id, description="teste", total_calories=0)
    db_session.add(meal)
    db_session.commit()

    item = FoodItem(
        meal_id=meal.id,
        food_name="Banana",
        quantity=1,
        unit="unidade",
        calories=89,
        protein=1.1,
        carbs=22.8,
        fat=0.3,
    )
    db_session.add(item)
    db_session.commit()

    db_session.delete(meal)
    db_session.commit()

    assert db_session.query(FoodItem).count() == 0
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_db/test_models.py -v`
Expected: 3 tests PASS

```bash
git add src/db/ tests/conftest.py tests/test_db/
git commit -m "feat: add SQLAlchemy models for users, meals, food items, conversations"
```

---

## Task 4: TACO Data Preparation

**Files:**
- Create: `data/taco_foods.json`
- Create: `scripts/__init__.py`
- Create: `scripts/seed_food_db.py`

The TACO database is publicly available. We need to download it and normalize it into our JSON format. The TACO data is available as a structured dataset from the Brazilian government.

- [ ] **Step 1: Research TACO data availability**

Search for the TACO database in a downloadable structured format (CSV or JSON). The official source is the UNICAMP NEPA website. There are also GitHub repositories with pre-processed versions.

Possible sources:
- https://github.com/topics/taco-database — look for JSON/CSV exports
- IBGE/TACO published PDFs that have been digitized by the community

You need a JSON file with this structure per food item:

```json
{
  "id": "taco_001",
  "name_pt": "Arroz, integral, cozido",
  "category": "Cereais e derivados",
  "nutrients_per_100g": {
    "calories": 124.0,
    "protein": 2.6,
    "carbs": 25.8,
    "fat": 1.0,
    "fiber": 2.7
  },
  "common_portions": [
    {"description": "1 colher de sopa", "grams": 25},
    {"description": "1 xicara", "grams": 160}
  ],
  "synonyms": ["arroz integral"]
}
```

If a clean JSON isn't available, create a representative subset manually (30-50 of the most common Brazilian foods) to unblock development. The full database can be ingested later.

- [ ] **Step 2: Create data/taco_foods.json**

Write the TACO data file. If using a pre-existing JSON source, download and transform it. If creating manually, include at minimum these common foods:

```json
[
  {
    "id": "taco_001",
    "name_pt": "Arroz, tipo 1, cozido",
    "category": "Cereais e derivados",
    "nutrients_per_100g": {
      "calories": 128.0,
      "protein": 2.5,
      "carbs": 28.1,
      "fat": 0.2,
      "fiber": 1.6
    },
    "common_portions": [
      {"description": "1 colher de sopa", "grams": 25},
      {"description": "1 escumadeira", "grams": 90},
      {"description": "1 xicara", "grams": 160}
    ],
    "synonyms": ["arroz branco", "arroz cozido", "arroz"]
  },
  {
    "id": "taco_002",
    "name_pt": "Feijao, carioca, cozido",
    "category": "Leguminosas e derivados",
    "nutrients_per_100g": {
      "calories": 76.0,
      "protein": 4.8,
      "carbs": 13.6,
      "fat": 0.5,
      "fiber": 8.5
    },
    "common_portions": [
      {"description": "1 colher de sopa", "grams": 25},
      {"description": "1 concha", "grams": 120}
    ],
    "synonyms": ["feijao", "feijao carioca"]
  }
]
```

Include at least 30 foods covering: grains (arroz, macarrao, pao), proteins (frango, carne bovina, ovo, peixe), legumes (feijao, lentilha), vegetables (alface, tomate, cenoura, brocolis), fruits (banana, maca, laranja, manga), dairy (leite, queijo, iogurte), and common Brazilian foods (mandioca, acai, tapioca, coxinha, pao de queijo).


---

## Task 5: Embedding Model and ChromaDB Vector Store

**Files:**
- Create: `src/vectorstore/__init__.py`
- Create: `src/vectorstore/embeddings.py`
- Create: `src/vectorstore/store.py`
- Create: `tests/test_vectorstore/__init__.py` (empty)
- Create: `tests/test_vectorstore/test_store.py`
- Create: `scripts/__init__.py`
- Create: `scripts/seed_food_db.py`

- [ ] **Step 1: Create src/vectorstore/__init__.py**

```python
```

- [ ] **Step 2: Write src/vectorstore/embeddings.py**

```python
from sentence_transformers import SentenceTransformer

from src.core.config import settings

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()
    return model.encode(text).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.encode(texts).tolist()
```

- [ ] **Step 3: Write src/vectorstore/store.py**

```python
import json

import chromadb

from src.core.config import settings
from src.vectorstore.embeddings import embed_text, embed_texts


def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection(client: chromadb.ClientAPI | None = None) -> chromadb.Collection:
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_embedding_text(food: dict) -> str:
    parts = [food["name_pt"]]
    parts.extend(food.get("synonyms", []))
    parts.append(food.get("category", ""))
    return " ".join(parts).strip()


def seed_foods(foods: list[dict]) -> int:
    collection = get_collection()

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    texts_to_embed = []
    for food in foods:
        embedding_text = build_embedding_text(food)
        ids.append(food["id"])
        documents.append(embedding_text)
        texts_to_embed.append(embedding_text)
        metadatas.append({
            "name_pt": food["name_pt"],
            "category": food.get("category", ""),
            "nutrients_json": json.dumps(food["nutrients_per_100g"]),
            "portions_json": json.dumps(food.get("common_portions", [])),
            "synonyms_json": json.dumps(food.get("synonyms", [])),
        })

    embeddings = embed_texts(texts_to_embed)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(ids)


def search_foods(query: str, top_k: int = 5) -> list[dict]:
    collection = get_collection()
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    foods = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance
        metadata = results["metadatas"][0][i]
        foods.append({
            "id": results["ids"][0][i],
            "name_pt": metadata["name_pt"],
            "category": metadata["category"],
            "nutrients_per_100g": json.loads(metadata["nutrients_json"]),
            "common_portions": json.loads(metadata["portions_json"]),
            "synonyms": json.loads(metadata["synonyms_json"]),
            "similarity": round(similarity, 4),
        })

    return foods
```

- [ ] **Step 4: Write scripts/__init__.py and scripts/seed_food_db.py**

`scripts/__init__.py`:
```python
```

`scripts/seed_food_db.py`:
```python
import json
from pathlib import Path

from src.vectorstore.store import seed_foods


def main():
    data_path = Path("data/taco_foods.json")
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return

    with open(data_path) as f:
        foods = json.load(f)

    print(f"Seeding {len(foods)} foods into ChromaDB...")
    count = seed_foods(foods)
    print(f"Done. {count} foods indexed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write the failing test**

Create `tests/test_vectorstore/__init__.py` (empty) and `tests/test_vectorstore/test_store.py`:

```python
import pytest

from src.vectorstore.store import build_embedding_text, search_foods, seed_foods


@pytest.fixture
def sample_foods():
    return [
        {
            "id": "test_001",
            "name_pt": "Arroz, tipo 1, cozido",
            "category": "Cereais e derivados",
            "nutrients_per_100g": {
                "calories": 128.0,
                "protein": 2.5,
                "carbs": 28.1,
                "fat": 0.2,
                "fiber": 1.6,
            },
            "common_portions": [{"description": "1 xicara", "grams": 160}],
            "synonyms": ["arroz branco", "arroz cozido"],
        },
        {
            "id": "test_002",
            "name_pt": "Frango, peito, sem pele, grelhado",
            "category": "Carnes e derivados",
            "nutrients_per_100g": {
                "calories": 159.0,
                "protein": 32.0,
                "carbs": 0.0,
                "fat": 2.5,
                "fiber": 0.0,
            },
            "common_portions": [{"description": "1 file", "grams": 100}],
            "synonyms": ["frango grelhado", "peito de frango"],
        },
        {
            "id": "test_003",
            "name_pt": "Banana, prata, crua",
            "category": "Frutas e derivados",
            "nutrients_per_100g": {
                "calories": 98.0,
                "protein": 1.3,
                "carbs": 26.0,
                "fat": 0.1,
                "fiber": 2.0,
            },
            "common_portions": [{"description": "1 unidade media", "grams": 86}],
            "synonyms": ["banana", "banana prata"],
        },
    ]


@pytest.fixture(autouse=True)
def _clean_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr("src.vectorstore.store.settings.chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr("src.vectorstore.store.settings.chroma_collection_name", "test_foods")


def test_build_embedding_text(sample_foods):
    text = build_embedding_text(sample_foods[0])
    assert "Arroz, tipo 1, cozido" in text
    assert "arroz branco" in text
    assert "Cereais e derivados" in text


def test_seed_and_search(sample_foods):
    count = seed_foods(sample_foods)
    assert count == 3

    results = search_foods("frango grelhado", top_k=3)
    assert len(results) > 0
    assert results[0]["name_pt"] == "Frango, peito, sem pele, grelhado"
    assert results[0]["similarity"] > 0.5


def test_search_returns_nutrients(sample_foods):
    seed_foods(sample_foods)
    results = search_foods("banana")
    assert len(results) > 0
    nutrients = results[0]["nutrients_per_100g"]
    assert "calories" in nutrients
    assert nutrients["calories"] == 98.0


def test_search_no_results_returns_empty():
    results = search_foods("xyznotafood123")
    assert isinstance(results, list)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_vectorstore/test_store.py -v`
Expected: 4 tests PASS (the first run will download the embedding model — ~100MB — which may take a minute)

- [ ] **Step 7: Seed the real TACO data**

Run: `uv run python -m scripts.seed_food_db`
Expected: `Seeding N foods into ChromaDB... Done. N foods indexed.`

- [ ] **Step 8: Smoke test real searches**

Run:
```bash
uv run python -c "
from src.vectorstore.store import search_foods
for q in ['frango grelhado', 'arroz', 'banana', 'feijao', 'acai']:
    results = search_foods(q, top_k=1)
    if results:
        print(f'{q:20s} -> {results[0][\"name_pt\"]:40s} (sim: {results[0][\"similarity\"]:.3f})')
    else:
        print(f'{q:20s} -> NO MATCH')
"
```
Expected: Each query matches a sensible food with similarity > 0.5

```bash
git add src/vectorstore/ scripts/ tests/test_vectorstore/
git commit -m "feat: add ChromaDB vector store with TACO food search"
```

---

## Task 6: LLM Provider Factory

**Files:**
- Create: `src/core/llm.py`

- [ ] **Step 1: Write src/core/llm.py**

```python
from langchain_ollama import ChatOllama

from src.core.config import settings

_llm: ChatOllama | None = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=1024,
        )
    return _llm
```

- [ ] **Step 2: Verify LLM responds**

Run: `uv run python -c "from src.core.llm import get_llm; llm = get_llm(); print(llm.invoke('Diga oi em uma palavra').content)"`
Expected: A Portuguese greeting (e.g., "Oi!" or "Ola!")

Note: Ollama must be running (`ollama serve`).

```bash
git add src/core/llm.py
git commit -m "feat: add Ollama LLM provider factory"
```

---

## Task 7: Agent Schemas and Safety Utilities

**Files:**
- Create: `src/agents/__init__.py`
- Create: `src/agents/schemas.py`
- Create: `src/agents/safety.py`
- Create: `tests/test_agents/__init__.py` (empty)
- Create: `tests/test_agents/test_schemas.py`
- Create: `tests/test_agents/test_safety.py`

- [ ] **Step 1: Create src/agents/__init__.py**

```python
```

- [ ] **Step 2: Write src/agents/schemas.py**

```python
from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    LOG_FOOD = "log_food"
    QUERY_HISTORY = "query_history"
    EDIT_MEAL = "edit_meal"
    GENERAL = "general"


class IntentClassification(BaseModel):
    intent: Intent
    reasoning: str = Field(description="Brief explanation of why this intent was chosen")


class ExtractedFoodItem(BaseModel):
    food_name: str = Field(description="Normalized food name in Portuguese")
    original_text: str = Field(description="What the user typed")
    quantity: float | None = Field(default=None, description="Numeric quantity")
    unit: str | None = Field(default=None, description="Unit: g, ml, unidade, fatia, xicara")
    quantity_grams: float | None = Field(default=None, description="Estimated weight in grams")
    preparation_method: str | None = Field(
        default=None, description="Preparation: grelhado, frito, cozido, cru"
    )


class MealExtraction(BaseModel):
    items: list[ExtractedFoodItem]
    meal_type: str | None = Field(
        default=None, description="cafe_da_manha, almoco, jantar, lanche"
    )
    ambiguities: list[str] = Field(
        default_factory=list,
        description="Any ambiguities found that the user should clarify",
    )
```

- [ ] **Step 3: Write tests/test_agents/test_schemas.py**

Create `tests/test_agents/__init__.py` (empty) and `tests/test_agents/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError

from src.agents.schemas import (
    ExtractedFoodItem,
    Intent,
    IntentClassification,
    MealExtraction,
)


def test_intent_classification_valid():
    ic = IntentClassification(intent=Intent.LOG_FOOD, reasoning="User mentioned eating")
    assert ic.intent == Intent.LOG_FOOD


def test_intent_classification_from_string():
    ic = IntentClassification(intent="log_food", reasoning="test")
    assert ic.intent == Intent.LOG_FOOD


def test_intent_classification_invalid_intent():
    with pytest.raises(ValidationError):
        IntentClassification(intent="invalid", reasoning="test")


def test_extracted_food_item_minimal():
    item = ExtractedFoodItem(food_name="frango", original_text="frango grelhado")
    assert item.food_name == "frango"
    assert item.quantity is None
    assert item.unit is None


def test_extracted_food_item_full():
    item = ExtractedFoodItem(
        food_name="Frango, peito, grelhado",
        original_text="150g de frango grelhado",
        quantity=150,
        unit="g",
        quantity_grams=150,
        preparation_method="grelhado",
    )
    assert item.quantity_grams == 150


def test_meal_extraction():
    meal = MealExtraction(
        items=[
            ExtractedFoodItem(food_name="arroz", original_text="arroz"),
            ExtractedFoodItem(food_name="feijao", original_text="feijao"),
        ],
        meal_type="almoco",
    )
    assert len(meal.items) == 2
    assert meal.meal_type == "almoco"
    assert meal.ambiguities == []


def test_meal_extraction_empty_items_allowed():
    meal = MealExtraction(items=[])
    assert len(meal.items) == 0
```

- [ ] **Step 4: Run schema tests**

Run: `uv run pytest tests/test_agents/test_schemas.py -v`
Expected: All tests PASS

- [ ] **Step 5: Write src/agents/safety.py**

```python
import logging

from pydantic import BaseModel, ValidationError

from src.core.config import settings

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "Desculpa, tive um problema processando sua mensagem. Pode tentar de novo?"
OLLAMA_DOWN_MESSAGE = (
    "Estou com problemas tecnicos no momento. Tente novamente em alguns minutos."
)


class CircuitBreakerTripped(Exception):
    pass


class CircuitBreaker:
    def __init__(self, max_iterations: int | None = None):
        self.max_iterations = max_iterations or settings.max_tool_iterations
        self.iteration_count = 0

    def tick(self) -> None:
        self.iteration_count += 1
        if self.iteration_count > self.max_iterations:
            logger.warning(
                "Circuit breaker tripped after %d iterations", self.iteration_count - 1
            )
            raise CircuitBreakerTripped(
                f"Max tool iterations ({self.max_iterations}) exceeded"
            )

    def reset(self) -> None:
        self.iteration_count = 0


def validate_llm_output(raw_output: str, schema: type[BaseModel]) -> BaseModel | None:
    try:
        return schema.model_validate_json(raw_output)
    except (ValidationError, ValueError) as e:
        logger.warning("LLM output validation failed: %s", e)
        return None
```

- [ ] **Step 6: Write tests/test_agents/test_safety.py**

```python
import pytest

from src.agents.safety import (
    CircuitBreaker,
    CircuitBreakerTripped,
    validate_llm_output,
)
from src.agents.schemas import IntentClassification


def test_circuit_breaker_allows_under_limit():
    cb = CircuitBreaker(max_iterations=3)
    cb.tick()
    cb.tick()
    cb.tick()


def test_circuit_breaker_trips_over_limit():
    cb = CircuitBreaker(max_iterations=2)
    cb.tick()
    cb.tick()
    with pytest.raises(CircuitBreakerTripped):
        cb.tick()


def test_circuit_breaker_reset():
    cb = CircuitBreaker(max_iterations=2)
    cb.tick()
    cb.tick()
    cb.reset()
    cb.tick()


def test_validate_llm_output_valid():
    raw = '{"intent": "log_food", "reasoning": "user is logging food"}'
    result = validate_llm_output(raw, IntentClassification)
    assert result is not None
    assert result.intent.value == "log_food"


def test_validate_llm_output_invalid():
    raw = '{"intent": "not_real", "reasoning": "bad"}'
    result = validate_llm_output(raw, IntentClassification)
    assert result is None


def test_validate_llm_output_malformed_json():
    raw = "this is not json at all"
    result = validate_llm_output(raw, IntentClassification)
    assert result is None
```

- [ ] **Step 7: Run safety tests**

Run: `uv run pytest tests/test_agents/test_safety.py -v`
Expected: All 6 tests PASS

```bash
git add src/agents/__init__.py src/agents/schemas.py src/agents/safety.py tests/test_agents/
git commit -m "feat: add agent schemas (intent, food extraction) and safety utilities (circuit breaker, validation)"
```

---

## Task 8: Prompt Templates

**Files:**
- Create: `data/prompts/system_prompt.txt`
- Create: `data/prompts/intent_classifier.txt`
- Create: `data/prompts/food_extraction.txt`
- Create: `data/prompts/few_shot_examples.json`

- [ ] **Step 1: Write data/prompts/system_prompt.txt**

```text
Voce e um assistente pessoal de nutricao que ajuda o usuario a registrar suas refeicoes via mensagens de texto. Voce e amigavel, conciso e preciso.

## Capacidades
- Registrar refeicoes a partir de descricoes em linguagem natural
- Consultar informacoes nutricionais de qualquer alimento
- Responder perguntas sobre historico alimentar (totais diarios, tendencias semanais)
- Ajudar o usuario a entender seu consumo em relacao as suas metas

## Regras
1. SEMPRE busque no banco de alimentos antes de registrar. Nunca invente valores nutricionais.
2. Quando nao houver quantidade, estime uma porcao razoavel e AVISE o usuario.
3. Quando houver ambiguidade, escolha a opcao mais comum e mencione alternativas.
4. Mantenha respostas curtas e objetivas.
5. Infira o tipo de refeicao pelo horario se nao for dito:
   - Antes das 10:00 -> cafe da manha
   - 10:00-14:00 -> almoco
   - 14:00-17:00 -> lanche
   - 17:00-21:00 -> jantar
   - Apos 21:00 -> lanche
6. Sempre mostre o detalhamento por item E o total apos registrar.
7. Para correcoes, encontre a refeicao mais recente e atualize.
8. Nunca de conselhos medicos ou dieteticos. Voce registra, nao prescreve.

## Perfil do Usuario
Nome: {user_name}
Meta diaria de calorias: {calorie_goal} kcal
Meta diaria de proteina: {protein_goal}g
Fuso: {timezone}

## Contexto
Data/hora atual: {current_datetime}
Totais de hoje ate agora: {today_totals}
```

- [ ] **Step 2: Write data/prompts/intent_classifier.txt**

```text
Classifique a intencao do usuario em UMA das categorias abaixo. Responda APENAS com JSON valido.

Categorias:
- "log_food": o usuario quer registrar algo que comeu ou bebeu
- "query_history": o usuario quer consultar o que ja comeu, calorias, totais, historico
- "edit_meal": o usuario quer corrigir, atualizar ou deletar uma refeicao anterior
- "general": qualquer outra coisa (saudacao, agradecimento, pergunta generica)

Exemplos:
Mensagem: "Comi 2 bananas"
{{"intent": "log_food", "reasoning": "usuario relata que comeu algo"}}

Mensagem: "Quantas calorias comi hoje?"
{{"intent": "query_history", "reasoning": "usuario quer saber totais do dia"}}

Mensagem: "Na verdade foram 200g, nao 150g"
{{"intent": "edit_meal", "reasoning": "usuario quer corrigir uma refeicao anterior"}}

Mensagem: "Oi, tudo bem?"
{{"intent": "general", "reasoning": "saudacao, nao e sobre comida"}}

Mensagem: "{user_message}"
```

- [ ] **Step 3: Write data/prompts/food_extraction.txt**

```text
Extraia os alimentos mencionados na mensagem do usuario. Responda APENAS com JSON valido.

Para cada alimento, extraia:
- food_name: nome normalizado do alimento em portugues
- original_text: o texto exato que o usuario escreveu referente a esse alimento
- quantity: quantidade numerica (null se nao mencionada)
- unit: unidade (g, ml, unidade, fatia, xicara, colher, concha, null se nao mencionada)
- quantity_grams: estimativa em gramas (null se impossivel estimar)
- preparation_method: metodo de preparo se mencionado (grelhado, frito, cozido, cru, assado, null)

Se a quantidade nao for mencionada, defina quantity e quantity_grams como null.

Exemplos:

Mensagem: "Almocei 150g de frango grelhado com arroz"
{{"items": [{{"food_name": "frango grelhado", "original_text": "150g de frango grelhado", "quantity": 150, "unit": "g", "quantity_grams": 150, "preparation_method": "grelhado"}}, {{"food_name": "arroz branco cozido", "original_text": "arroz", "quantity": null, "unit": null, "quantity_grams": null, "preparation_method": "cozido"}}], "meal_type": "almoco", "ambiguities": ["quantidade de arroz nao especificada"]}}

Mensagem: "Comi 2 bananas"
{{"items": [{{"food_name": "banana", "original_text": "2 bananas", "quantity": 2, "unit": "unidade", "quantity_grams": 172, "preparation_method": null}}], "meal_type": null, "ambiguities": []}}

Mensagem: "Tomei um cafe com leite e comi pao com manteiga"
{{"items": [{{"food_name": "cafe com leite", "original_text": "cafe com leite", "quantity": 1, "unit": "xicara", "quantity_grams": 200, "preparation_method": null}}, {{"food_name": "pao frances", "original_text": "pao", "quantity": 1, "unit": "unidade", "quantity_grams": 50, "preparation_method": null}}, {{"food_name": "manteiga", "original_text": "manteiga", "quantity": 1, "unit": "porcao", "quantity_grams": 10, "preparation_method": null}}], "meal_type": "cafe_da_manha", "ambiguities": []}}

Mensagem: "{user_message}"
```

- [ ] **Step 4: Write data/prompts/few_shot_examples.json**

```json
{
  "intent_classification": [
    {
      "input": "Almocei arroz com feijao e bife",
      "output": {"intent": "log_food", "reasoning": "usuario relata o que almocou"}
    },
    {
      "input": "Quanto de proteina eu comi hoje?",
      "output": {"intent": "query_history", "reasoning": "usuario quer saber total de proteina do dia"}
    },
    {
      "input": "Apaga o almoco de hoje",
      "output": {"intent": "edit_meal", "reasoning": "usuario quer deletar uma refeicao"}
    },
    {
      "input": "Obrigado!",
      "output": {"intent": "general", "reasoning": "agradecimento, nao e sobre comida"}
    }
  ],
  "food_extraction": [
    {
      "input": "Comi uma fatia de pizza de calabresa",
      "output": {
        "items": [
          {
            "food_name": "pizza de calabresa",
            "original_text": "uma fatia de pizza de calabresa",
            "quantity": 1,
            "unit": "fatia",
            "quantity_grams": 107,
            "preparation_method": null
          }
        ],
        "meal_type": null,
        "ambiguities": []
      }
    },
    {
      "input": "No jantar comi salada com frango grelhado e tomei suco de laranja",
      "output": {
        "items": [
          {
            "food_name": "salada verde",
            "original_text": "salada",
            "quantity": 1,
            "unit": "porcao",
            "quantity_grams": 100,
            "preparation_method": "cru"
          },
          {
            "food_name": "frango grelhado",
            "original_text": "frango grelhado",
            "quantity": null,
            "unit": null,
            "quantity_grams": null,
            "preparation_method": "grelhado"
          },
          {
            "food_name": "suco de laranja",
            "original_text": "suco de laranja",
            "quantity": 1,
            "unit": "copo",
            "quantity_grams": 250,
            "preparation_method": null
          }
        ],
        "meal_type": "jantar",
        "ambiguities": ["quantidade de frango nao especificada"]
      }
    }
  ]
}
```

```bash
git add data/prompts/
git commit -m "feat: add prompt templates and few-shot examples for intent classification and food extraction"
```

---

## Task 9: Services — Nutrition and Meal

**Files:**
- Create: `src/services/__init__.py`
- Create: `src/services/nutrition_service.py`
- Create: `src/services/meal_service.py`
- Create: `tests/test_services/__init__.py` (empty)
- Create: `tests/test_services/test_nutrition_service.py`
- Create: `tests/test_services/test_meal_service.py`

- [ ] **Step 1: Create src/services/__init__.py**

```python
```

- [ ] **Step 2: Write src/services/nutrition_service.py**

```python
from src.vectorstore.store import search_foods

HIGH_CONFIDENCE_THRESHOLD = 0.90
MEDIUM_CONFIDENCE_THRESHOLD = 0.75


def lookup_food(query: str, top_k: int = 5) -> dict:
    results = search_foods(query, top_k=top_k)

    if not results:
        return {"status": "not_found", "matches": [], "query": query}

    top = results[0]

    if top["similarity"] >= HIGH_CONFIDENCE_THRESHOLD:
        return {"status": "matched", "matches": [top], "query": query}

    if top["similarity"] >= MEDIUM_CONFIDENCE_THRESHOLD:
        candidates = [r for r in results[:3] if r["similarity"] >= MEDIUM_CONFIDENCE_THRESHOLD]
        return {"status": "ambiguous", "matches": candidates, "query": query}

    return {"status": "low_confidence", "matches": results[:3], "query": query}


def calculate_nutrients(nutrients_per_100g: dict, grams: float) -> dict:
    factor = grams / 100.0
    return {
        "calories": round(nutrients_per_100g.get("calories", 0) * factor, 1),
        "protein": round(nutrients_per_100g.get("protein", 0) * factor, 1),
        "carbs": round(nutrients_per_100g.get("carbs", 0) * factor, 1),
        "fat": round(nutrients_per_100g.get("fat", 0) * factor, 1),
        "fiber": round(nutrients_per_100g.get("fiber", 0) * factor, 1),
    }


def estimate_grams_from_portion(food: dict, unit: str | None, quantity: float | None) -> float:
    if unit == "g" and quantity is not None:
        return quantity

    portions = food.get("common_portions", [])
    if portions and quantity is not None:
        for portion in portions:
            desc = portion["description"].lower()
            if unit and unit.lower() in desc:
                return portion["grams"] * quantity
        return portions[0]["grams"] * quantity

    if portions:
        return portions[0]["grams"]

    return 100.0
```

- [ ] **Step 3: Write the nutrition service test**

Create `tests/test_services/__init__.py` (empty) and `tests/test_services/test_nutrition_service.py`:

```python
from src.services.nutrition_service import calculate_nutrients, estimate_grams_from_portion


def test_calculate_nutrients_100g():
    nutrients = {"calories": 128.0, "protein": 2.5, "carbs": 28.1, "fat": 0.2, "fiber": 1.6}
    result = calculate_nutrients(nutrients, 100.0)
    assert result["calories"] == 128.0
    assert result["protein"] == 2.5


def test_calculate_nutrients_150g():
    nutrients = {"calories": 100.0, "protein": 10.0, "carbs": 20.0, "fat": 5.0, "fiber": 2.0}
    result = calculate_nutrients(nutrients, 150.0)
    assert result["calories"] == 150.0
    assert result["protein"] == 15.0


def test_calculate_nutrients_zero_grams():
    nutrients = {"calories": 100.0, "protein": 10.0, "carbs": 20.0, "fat": 5.0, "fiber": 2.0}
    result = calculate_nutrients(nutrients, 0.0)
    assert result["calories"] == 0.0


def test_estimate_grams_explicit_grams():
    food = {"common_portions": []}
    assert estimate_grams_from_portion(food, "g", 150.0) == 150.0


def test_estimate_grams_from_matching_portion():
    food = {
        "common_portions": [
            {"description": "1 xicara", "grams": 160},
            {"description": "1 colher de sopa", "grams": 25},
        ]
    }
    result = estimate_grams_from_portion(food, "xicara", 2)
    assert result == 320.0


def test_estimate_grams_falls_back_to_first_portion():
    food = {
        "common_portions": [
            {"description": "1 unidade media", "grams": 86},
        ]
    }
    result = estimate_grams_from_portion(food, "fatia", 1)
    assert result == 86.0


def test_estimate_grams_no_portions_defaults_100():
    food = {"common_portions": []}
    result = estimate_grams_from_portion(food, None, None)
    assert result == 100.0
```

- [ ] **Step 4: Run nutrition service tests**

Run: `uv run pytest tests/test_services/test_nutrition_service.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Write src/services/meal_service.py**

```python
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.db.models import FoodItem, Meal, User


def get_or_create_user(session: Session, phone_number: str, display_name: str | None = None) -> User:
    user = session.query(User).filter(User.phone_number == phone_number).first()
    if user is None:
        user = User(phone_number=phone_number, display_name=display_name)
        session.add(user)
        session.commit()
    return user


def log_meal(
    session: Session,
    user_id: str,
    description: str,
    meal_type: str | None,
    food_items: list[dict],
) -> Meal:
    total_cal = sum(item["calories"] for item in food_items)
    total_prot = sum(item["protein"] for item in food_items)
    total_carbs = sum(item["carbs"] for item in food_items)
    total_fat = sum(item["fat"] for item in food_items)

    meal = Meal(
        user_id=user_id,
        description=description,
        meal_type=meal_type,
        total_calories=round(total_cal, 1),
        total_protein=round(total_prot, 1),
        total_carbs=round(total_carbs, 1),
        total_fat=round(total_fat, 1),
    )
    session.add(meal)
    session.flush()

    for item in food_items:
        food_item = FoodItem(
            meal_id=meal.id,
            food_name=item["food_name"],
            original_text=item.get("original_text"),
            quantity=item["quantity"],
            unit=item["unit"],
            serving_grams=item.get("serving_grams"),
            calories=item["calories"],
            protein=item["protein"],
            carbs=item["carbs"],
            fat=item["fat"],
            fiber=item.get("fiber"),
            data_source=item.get("data_source"),
            confidence=item.get("confidence"),
        )
        session.add(food_item)

    session.commit()
    return meal


def get_today_totals(session: Session, user_id: str) -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    meals = (
        session.query(Meal)
        .filter(Meal.user_id == user_id, Meal.logged_at.like(f"{today}%"))
        .all()
    )

    return {
        "calories": round(sum(m.total_calories for m in meals), 1),
        "protein": round(sum(m.total_protein for m in meals), 1),
        "carbs": round(sum(m.total_carbs for m in meals), 1),
        "fat": round(sum(m.total_fat for m in meals), 1),
        "meal_count": len(meals),
    }


def get_meals_for_date(session: Session, user_id: str, date_str: str) -> list[Meal]:
    return (
        session.query(Meal)
        .filter(Meal.user_id == user_id, Meal.logged_at.like(f"{date_str}%"))
        .order_by(Meal.logged_at)
        .all()
    )


def delete_meal(session: Session, meal_id: str) -> bool:
    meal = session.query(Meal).filter(Meal.id == meal_id).first()
    if meal is None:
        return False
    session.delete(meal)
    session.commit()
    return True
```

- [ ] **Step 6: Write tests/test_services/test_meal_service.py**

```python
from src.db.models import User
from src.services.meal_service import (
    delete_meal,
    get_or_create_user,
    get_today_totals,
    log_meal,
)


def test_get_or_create_user_creates(db_session):
    user = get_or_create_user(db_session, "+5511999999999", "Alex")
    assert user.id is not None
    assert user.phone_number == "+5511999999999"


def test_get_or_create_user_returns_existing(db_session):
    user1 = get_or_create_user(db_session, "+5511999999999", "Alex")
    user2 = get_or_create_user(db_session, "+5511999999999", "Alex")
    assert user1.id == user2.id


def test_log_meal(db_session):
    user = get_or_create_user(db_session, "+5511999999999")

    meal = log_meal(
        db_session,
        user_id=user.id,
        description="frango grelhado com arroz",
        meal_type="almoco",
        food_items=[
            {
                "food_name": "Frango, peito, grelhado",
                "original_text": "frango grelhado",
                "quantity": 150,
                "unit": "g",
                "serving_grams": 150,
                "calories": 247.5,
                "protein": 47.0,
                "carbs": 0.0,
                "fat": 5.0,
                "data_source": "taco",
                "confidence": 0.95,
            },
            {
                "food_name": "Arroz, tipo 1, cozido",
                "original_text": "arroz",
                "quantity": 200,
                "unit": "g",
                "serving_grams": 200,
                "calories": 256.0,
                "protein": 5.0,
                "carbs": 56.2,
                "fat": 0.4,
                "data_source": "taco",
                "confidence": 0.92,
            },
        ],
    )

    assert meal.id is not None
    assert meal.total_calories == 503.5
    assert meal.total_protein == 52.0
    assert len(meal.food_items) == 2


def test_get_today_totals(db_session):
    user = get_or_create_user(db_session, "+5511999999999")

    log_meal(
        db_session,
        user_id=user.id,
        description="banana",
        meal_type="lanche",
        food_items=[
            {
                "food_name": "Banana",
                "quantity": 1,
                "unit": "unidade",
                "calories": 89,
                "protein": 1.1,
                "carbs": 22.8,
                "fat": 0.3,
            }
        ],
    )

    totals = get_today_totals(db_session, user.id)
    assert totals["calories"] == 89
    assert totals["meal_count"] == 1


def test_delete_meal(db_session):
    user = get_or_create_user(db_session, "+5511999999999")
    meal = log_meal(
        db_session,
        user_id=user.id,
        description="test",
        meal_type=None,
        food_items=[
            {
                "food_name": "Test",
                "quantity": 1,
                "unit": "g",
                "calories": 100,
                "protein": 10,
                "carbs": 10,
                "fat": 5,
            }
        ],
    )

    assert delete_meal(db_session, meal.id) is True
    assert delete_meal(db_session, meal.id) is False
```

- [ ] **Step 7: Run meal service tests**

Run: `uv run pytest tests/test_services/test_meal_service.py -v`
Expected: All 5 tests PASS

```bash
git add src/services/ tests/test_services/
git commit -m "feat: add nutrition lookup and meal CRUD services"
```

---

## Task 10: LangGraph State and Agent Nodes

**Files:**
- Create: `src/agents/state.py`
- Create: `src/agents/nodes/__init__.py`
- Create: `src/agents/nodes/intent_classifier.py`
- Create: `src/agents/nodes/food_logger.py`
- Create: `src/agents/nodes/query_handler.py`
- Create: `src/agents/nodes/fallback.py`
- Create: `tests/test_agents/test_intent_classifier.py`
- Create: `tests/test_agents/test_food_logger.py`
- Create: `tests/test_agents/test_query_handler.py`

- [ ] **Step 1: Write src/agents/state.py**

```python
from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    user_message: str
    intent: str | None
    response: str | None
    tool_iterations: int
    error: str | None
```

- [ ] **Step 2: Create src/agents/nodes/__init__.py**

```python
```

- [ ] **Step 3: Write src/agents/nodes/intent_classifier.py**

```python
import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.safety import validate_llm_output
from src.agents.schemas import Intent, IntentClassification
from src.agents.state import AgentState
from src.core.llm import get_llm

logger = logging.getLogger(__name__)


def _load_prompt(user_message: str) -> str:
    template_path = Path("data/prompts/intent_classifier.txt")
    template = template_path.read_text()
    return template.replace("{user_message}", user_message)


def classify_intent(state: AgentState) -> dict:
    user_message = state["user_message"]
    llm = get_llm()

    prompt = _load_prompt(user_message)
    messages = [
        SystemMessage(content="Voce e um classificador de intencao. Responda APENAS com JSON."),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = validate_llm_output(raw, IntentClassification)

    if result is None:
        logger.warning("Intent classification failed, retrying with error context")
        retry_messages = messages + [
            HumanMessage(
                content=f"Sua resposta anterior nao era JSON valido: {raw}\n\n"
                "Responda APENAS com JSON no formato: "
                '{"intent": "log_food|query_history|edit_meal|general", "reasoning": "..."}'
            ),
        ]
        response = llm.invoke(retry_messages)
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = validate_llm_output(raw, IntentClassification)

    if result is None:
        logger.warning("Intent classification failed twice, defaulting to general")
        return {"intent": Intent.GENERAL.value}

    return {"intent": result.intent.value}
```

- [ ] **Step 4: Write src/agents/nodes/food_logger.py**

```python
import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.orm import Session

from src.agents.safety import CircuitBreaker, validate_llm_output
from src.agents.schemas import MealExtraction
from src.agents.state import AgentState
from src.core.llm import get_llm
from src.db.engine import SessionLocal
from src.services.meal_service import log_meal
from src.services.nutrition_service import (
    calculate_nutrients,
    estimate_grams_from_portion,
    lookup_food,
)

logger = logging.getLogger(__name__)


def _load_prompt(user_message: str) -> str:
    template_path = Path("data/prompts/food_extraction.txt")
    template = template_path.read_text()
    return template.replace("{user_message}", user_message)


def _extract_foods(user_message: str) -> MealExtraction | None:
    llm = get_llm()
    prompt = _load_prompt(user_message)

    messages = [
        SystemMessage(content="Voce extrai alimentos de mensagens. Responda APENAS com JSON."),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = validate_llm_output(raw, MealExtraction)

    if result is None:
        logger.warning("Food extraction failed, retrying")
        retry_messages = messages + [
            HumanMessage(
                content=f"Sua resposta anterior nao era JSON valido: {raw}\n\n"
                "Responda APENAS com JSON no formato exato dos exemplos acima."
            ),
        ]
        response = llm.invoke(retry_messages)
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = validate_llm_output(raw, MealExtraction)

    return result


def _format_response(meal_data: dict) -> str:
    lines = [f"Registrado seu {meal_data.get('meal_type', 'refeicao')}!\n"]

    for item in meal_data["items"]:
        grams = item.get("serving_grams", "?")
        lines.append(
            f"  {item['food_name']} ({grams}g): "
            f"{item['calories']:.0f} cal | "
            f"{item['protein']:.0f}g prot | "
            f"{item['carbs']:.0f}g carb | "
            f"{item['fat']:.0f}g gord"
        )

    lines.append(
        f"\nTotal: {meal_data['total_calories']:.0f} cal | "
        f"{meal_data['total_protein']:.0f}g prot | "
        f"{meal_data['total_carbs']:.0f}g carb | "
        f"{meal_data['total_fat']:.0f}g gord"
    )

    ambiguities = meal_data.get("ambiguities", [])
    if ambiguities:
        lines.append("\nNota: " + "; ".join(ambiguities) + ". Quer ajustar?")

    return "\n".join(lines)


def log_food(state: AgentState) -> dict:
    user_message = state["user_message"]
    user_id = state["user_id"]
    breaker = CircuitBreaker()

    extraction = _extract_foods(user_message)
    breaker.tick()

    if extraction is None or not extraction.items:
        return {"response": "Nao consegui identificar os alimentos. Pode descrever de outra forma?"}

    food_items_data = []
    ambiguities = list(extraction.ambiguities)

    for item in extraction.items:
        breaker.tick()
        search_query = item.food_name
        if item.preparation_method:
            search_query += f" {item.preparation_method}"

        lookup = lookup_food(search_query)

        if lookup["status"] == "not_found":
            ambiguities.append(f"Nao encontrei '{item.original_text}' no banco de alimentos")
            continue

        matched_food = lookup["matches"][0]

        if lookup["status"] == "ambiguous":
            ambiguities.append(
                f"Para '{item.original_text}', usei '{matched_food['name_pt']}' "
                f"(confianca: {matched_food['similarity']:.0%})"
            )

        grams = estimate_grams_from_portion(
            matched_food, item.unit, item.quantity
        )

        if item.quantity_grams is not None:
            grams = item.quantity_grams

        nutrients = calculate_nutrients(matched_food["nutrients_per_100g"], grams)

        food_items_data.append({
            "food_name": matched_food["name_pt"],
            "original_text": item.original_text,
            "quantity": item.quantity or 1,
            "unit": item.unit or "porcao",
            "serving_grams": grams,
            "calories": nutrients["calories"],
            "protein": nutrients["protein"],
            "carbs": nutrients["carbs"],
            "fat": nutrients["fat"],
            "fiber": nutrients["fiber"],
            "data_source": "taco",
            "confidence": matched_food["similarity"],
        })

    if not food_items_data:
        return {
            "response": "Nao encontrei nenhum dos alimentos no banco de dados. "
            "Pode descrever de outra forma? Ex: 'frango grelhado' em vez de 'frango'"
        }

    session = SessionLocal()
    try:
        meal = log_meal(
            session,
            user_id=user_id,
            description=user_message,
            meal_type=extraction.meal_type,
            food_items=food_items_data,
        )

        response_data = {
            "meal_type": extraction.meal_type or "refeicao",
            "items": food_items_data,
            "total_calories": meal.total_calories,
            "total_protein": meal.total_protein,
            "total_carbs": meal.total_carbs,
            "total_fat": meal.total_fat,
            "ambiguities": ambiguities,
        }

        return {"response": _format_response(response_data)}
    finally:
        session.close()
```

- [ ] **Step 5: Write src/agents/nodes/query_handler.py**

```python
import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.orm import Session

from src.agents.state import AgentState
from src.core.llm import get_llm
from src.db.engine import SessionLocal
from src.services.meal_service import get_meals_for_date, get_today_totals

logger = logging.getLogger(__name__)


def _format_totals(totals: dict) -> str:
    if totals["meal_count"] == 0:
        return "Voce ainda nao registrou nenhuma refeicao hoje."

    return (
        f"Hoje ate agora ({totals['meal_count']} refeicoes):\n"
        f"  Calorias: {totals['calories']:.0f} kcal\n"
        f"  Proteina: {totals['protein']:.0f}g\n"
        f"  Carboidratos: {totals['carbs']:.0f}g\n"
        f"  Gordura: {totals['fat']:.0f}g"
    )


def _format_meals(meals: list, date_str: str) -> str:
    if not meals:
        return f"Nenhuma refeicao registrada em {date_str}."

    lines = [f"Refeicoes em {date_str}:\n"]
    for meal in meals:
        meal_label = meal.meal_type or "refeicao"
        lines.append(
            f"  {meal_label}: {meal.description} "
            f"({meal.total_calories:.0f} cal)"
        )

    return "\n".join(lines)


def handle_query(state: AgentState) -> dict:
    user_message = state["user_message"]
    user_id = state["user_id"]

    session = SessionLocal()
    try:
        totals = get_today_totals(session, user_id)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        meals = get_meals_for_date(session, user_id, today)

        context = _format_totals(totals)
        if meals:
            context += "\n\n" + _format_meals(meals, today)

        llm = get_llm()
        messages = [
            SystemMessage(
                content="Voce e um assistente de nutricao. Responda a pergunta do usuario "
                "com base nos dados abaixo. Seja conciso e direto.\n\n"
                f"Dados do usuario:\n{context}"
            ),
            HumanMessage(content=user_message),
        ]

        response = llm.invoke(messages)
        return {"response": response.content.strip()}
    finally:
        session.close()
```

- [ ] **Step 6: Write src/agents/nodes/fallback.py**

```python
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.core.llm import get_llm

logger = logging.getLogger(__name__)


def handle_fallback(state: AgentState) -> dict:
    user_message = state["user_message"]
    llm = get_llm()

    messages = [
        SystemMessage(
            content="Voce e um assistente de nutricao. O usuario enviou uma mensagem "
            "que nao e sobre registrar comida ou consultar historico. "
            "Responda de forma amigavel e concisa. Se possivel, lembre o usuario "
            "que voce pode ajudar a registrar refeicoes e consultar calorias."
        ),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    return {"response": response.content.strip()}
```

- [ ] **Step 7: Write tests/test_agents/test_intent_classifier.py**

This test uses a mock LLM to avoid requiring Ollama in CI:

```python
from unittest.mock import MagicMock, patch

from src.agents.nodes.intent_classifier import classify_intent
from src.agents.schemas import Intent


def _make_state(message: str) -> dict:
    return {
        "messages": [],
        "user_id": "test_user",
        "user_message": message,
        "intent": None,
        "response": None,
        "tool_iterations": 0,
        "error": None,
    }


@patch("src.agents.nodes.intent_classifier.get_llm")
def test_classify_log_food(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"intent": "log_food", "reasoning": "usuario comeu algo"}'
    )
    mock_get_llm.return_value = mock_llm

    result = classify_intent(_make_state("Comi 2 bananas"))
    assert result["intent"] == "log_food"


@patch("src.agents.nodes.intent_classifier.get_llm")
def test_classify_query_history(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"intent": "query_history", "reasoning": "quer saber calorias"}'
    )
    mock_get_llm.return_value = mock_llm

    result = classify_intent(_make_state("Quantas calorias comi hoje?"))
    assert result["intent"] == "query_history"


@patch("src.agents.nodes.intent_classifier.get_llm")
def test_classify_falls_back_on_invalid_json(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content="this is not json"),
        MagicMock(content="still not json"),
    ]
    mock_get_llm.return_value = mock_llm

    result = classify_intent(_make_state("random"))
    assert result["intent"] == "general"
```

- [ ] **Step 8: Write tests/test_agents/test_food_logger.py**

```python
from unittest.mock import MagicMock, patch

from src.agents.nodes.food_logger import _extract_foods, _format_response


@patch("src.agents.nodes.food_logger.get_llm")
def test_extract_foods_valid(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"items": [{"food_name": "banana", "original_text": "2 bananas", '
        '"quantity": 2, "unit": "unidade", "quantity_grams": 172, '
        '"preparation_method": null}], "meal_type": null, "ambiguities": []}'
    )
    mock_get_llm.return_value = mock_llm

    result = _extract_foods("Comi 2 bananas")
    assert result is not None
    assert len(result.items) == 1
    assert result.items[0].food_name == "banana"


@patch("src.agents.nodes.food_logger.get_llm")
def test_extract_foods_returns_none_on_failure(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="invalid json")
    mock_get_llm.return_value = mock_llm

    result = _extract_foods("test")
    assert result is None


def test_format_response():
    data = {
        "meal_type": "almoco",
        "items": [
            {
                "food_name": "Frango, peito, grelhado",
                "serving_grams": 150,
                "calories": 247.5,
                "protein": 47.0,
                "carbs": 0.0,
                "fat": 5.0,
            }
        ],
        "total_calories": 247.5,
        "total_protein": 47.0,
        "total_carbs": 0.0,
        "total_fat": 5.0,
        "ambiguities": [],
    }
    response = _format_response(data)
    assert "Registrado" in response
    assert "Frango" in response
    assert "248 cal" in response
```

- [ ] **Step 9: Write tests/test_agents/test_query_handler.py**

```python
from unittest.mock import MagicMock, patch

from src.agents.nodes.query_handler import _format_totals


def test_format_totals_no_meals():
    totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "meal_count": 0}
    result = _format_totals(totals)
    assert "nenhuma refeicao" in result.lower()


def test_format_totals_with_meals():
    totals = {"calories": 508, "protein": 52, "carbs": 56, "fat": 6, "meal_count": 1}
    result = _format_totals(totals)
    assert "508" in result
    assert "52" in result
```

- [ ] **Step 10: Run all agent tests**

Run: `uv run pytest tests/test_agents/ -v`
Expected: All tests PASS

```bash
git add src/agents/state.py src/agents/nodes/ tests/test_agents/
git commit -m "feat: add LangGraph state and agent nodes (intent classifier, food logger, query handler, fallback)"
```

---

## Task 11: LangGraph Graph Assembly

**Files:**
- Create: `src/agents/graph.py`

- [ ] **Step 1: Write src/agents/graph.py**

```python
from langgraph.graph import END, StateGraph

from src.agents.nodes.fallback import handle_fallback
from src.agents.nodes.food_logger import log_food
from src.agents.nodes.intent_classifier import classify_intent
from src.agents.nodes.query_handler import handle_query
from src.agents.state import AgentState


def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "log_food":
        return "food_logger"
    if intent == "query_history":
        return "query_handler"
    if intent == "edit_meal":
        return "food_logger"
    return "fallback"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("intent_classifier", classify_intent)
    graph.add_node("food_logger", log_food)
    graph.add_node("query_handler", handle_query)
    graph.add_node("fallback", handle_fallback)

    graph.set_entry_point("intent_classifier")

    graph.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "food_logger": "food_logger",
            "query_handler": "query_handler",
            "fallback": "fallback",
        },
    )

    graph.add_edge("food_logger", END)
    graph.add_edge("query_handler", END)
    graph.add_edge("fallback", END)

    return graph


agent = build_graph().compile()
```

- [ ] **Step 2: Verify graph compiles**

Run: `uv run python -c "from src.agents.graph import agent; print('Graph compiled:', agent)"`
Expected: `Graph compiled: <CompiledStateGraph ...>`

```bash
git add src/agents/graph.py
git commit -m "feat: assemble LangGraph agent with intent routing"
```

---

## Task 12: Interactive CLI Chat

**Files:**
- Create: `src/cli/__init__.py` (empty)
- Create: `src/cli/chat.py`

- [ ] **Step 1: Create src/cli/__init__.py**

```python
```

- [ ] **Step 2: Write src/cli/chat.py**

```python
import logging
import sys

from src.agents.graph import agent
from src.agents.safety import FALLBACK_MESSAGE, OLLAMA_DOWN_MESSAGE, CircuitBreakerTripped
from src.db.engine import SessionLocal, engine
from src.db.models import Base
from src.services.meal_service import get_or_create_user

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

WELCOME_MESSAGE = """
Ola! Sou seu assistente pessoal de nutricao.

Posso te ajudar a:
  - Registrar refeicoes: "Almocei 150g de frango com arroz"
  - Consultar calorias: "Quantas calorias comi hoje?"
  - Tirar duvidas: "Quantas calorias tem uma banana?"

Digite 'sair' para encerrar.
"""


def main():
    Base.metadata.create_all(engine)

    session = SessionLocal()
    user = get_or_create_user(session, phone_number="+5500000000000", display_name="CLI User")
    user_id = user.id
    session.close()

    print(WELCOME_MESSAGE)

    while True:
        try:
            user_input = input("\nVoce: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAte mais! Bom apetite!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("sair", "exit", "quit"):
            print("Ate mais! Bom apetite!")
            break

        try:
            result = agent.invoke({
                "messages": [],
                "user_id": user_id,
                "user_message": user_input,
                "intent": None,
                "response": None,
                "tool_iterations": 0,
                "error": None,
            })

            response = result.get("response", FALLBACK_MESSAGE)
            print(f"\nAssistente: {response}")

        except CircuitBreakerTripped:
            logger.warning("Circuit breaker tripped for message: %s", user_input)
            print(f"\nAssistente: {FALLBACK_MESSAGE}")
        except ConnectionError:
            logger.error("Cannot connect to Ollama")
            print(f"\nAssistente: {OLLAMA_DOWN_MESSAGE}")
        except Exception:
            logger.exception("Unexpected error")
            print(f"\nAssistente: {FALLBACK_MESSAGE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create the SQLite database tables**

Run: `uv run python -c "from src.db.engine import engine; from src.db.models import Base; Base.metadata.create_all(engine); print('DB created')"`
Expected: `DB created` and a `food_assistant.db` file appears in the project root.

- [ ] **Step 4: End-to-end test (manual, requires Ollama running)**

Run: `uv run python -m src.cli.chat`

Test these interactions:
1. Type "Comi 2 bananas" → should see food logged with ~180 kcal
2. Type "Almocei 150g de frango grelhado com arroz" → should see two items logged
3. Type "Quantas calorias comi hoje?" → should see correct total
4. Type "Oi, tudo bem?" → should see friendly response
5. Type "sair" → should exit

Note: The first message will be slow (model loading). Subsequent messages should be 5-15 seconds.

```bash
git add src/cli/
git commit -m "feat: add interactive CLI chat with LangGraph agent"
```

---

## Task 13: Run Full Test Suite

- [ ] **Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All tests pass. Count should be approximately:
- test_db: 3 tests
- test_vectorstore: 4 tests
- test_agents: ~10 tests
- test_services: ~12 tests

- [ ] **Step 2: Run linter**

Run: `uv run ruff check .`
Expected: No errors. If there are warnings, fix them.

- [ ] **Step 3: Run formatter**

Run: `uv run ruff format --check .`
Expected: All files formatted. If not, run `uv run ruff format .` and commit.

```bash
git add -u
git commit -m "chore: fix lint and formatting issues"
```
