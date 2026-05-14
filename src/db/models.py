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
    meal_id: Mapped[str] = mapped_column(
        ForeignKey("meals.id", ondelete="CASCADE"), nullable=False
    )
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
