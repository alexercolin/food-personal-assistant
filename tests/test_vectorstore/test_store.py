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
