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
