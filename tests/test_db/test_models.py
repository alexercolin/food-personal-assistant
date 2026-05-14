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
