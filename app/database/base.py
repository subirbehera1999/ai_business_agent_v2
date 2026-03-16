# ==============================================================================
# File: app/database/base.py
# Purpose: Defines the SQLAlchemy declarative base and shared mixins that every
#          ORM model in the system inherits from. Enforces consistent primary
#          key strategy, audit timestamps, and soft-delete capability across
#          all database tables.
# ==============================================================================

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedColumn, mapped_column


# ==============================================================================
# Declarative Base
# ==============================================================================

class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base class.

    All ORM models in the system must inherit from this class.
    Provides metadata shared across all tables and is used by
    Alembic for schema migration discovery.

    Usage:
        from app.database.base import Base

        class MyModel(Base):
            __tablename__ = "my_table"
            ...
    """
    pass


# ==============================================================================
# Primary Key Mixin
# ==============================================================================

class UUIDPrimaryKeyMixin:
    """
    Mixin that provides a UUID v4 primary key column named `id`.

    Uses PostgreSQL's native UUID type for storage efficiency.
    The default value is generated server-side via gen_random_uuid()
    to guarantee uniqueness even in bulk insert scenarios.

    Column:
        id (UUID): Primary key, non-nullable, auto-generated.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        nullable=False,
        comment="Unique record identifier (UUID v4)",
    )


# ==============================================================================
# Timestamp Mixin
# ==============================================================================

class TimestampMixin:
    """
    Mixin that provides `created_at` and `updated_at` audit timestamp columns.

    Both timestamps are timezone-aware and managed automatically:
    - created_at is set once at insert time by the database server.
    - updated_at is set at insert and refreshed on every update by the
      database server using onupdate, ensuring accuracy even for bulk
      operations that bypass the ORM.

    Columns:
        created_at (DateTime): Immutable creation timestamp (UTC).
        updated_at (DateTime): Mutable last-modified timestamp (UTC).
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Record creation timestamp (UTC)",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Record last-updated timestamp (UTC)",
    )


# ==============================================================================
# Soft Delete Mixin
# ==============================================================================

class SoftDeleteMixin:
    """
    Mixin that provides soft-delete capability via an `is_deleted` flag
    and a `deleted_at` timestamp.

    Records marked as deleted must be excluded from all application queries.
    Repositories are responsible for applying this filter consistently.

    Columns:
        is_deleted (bool): Soft-delete flag. Default False.
        deleted_at (DateTime | None): Timestamp of deletion if soft-deleted.
    """

    is_deleted: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        server_default=text("false"),
        comment="Soft-delete flag — True means logically deleted",
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment="Timestamp when the record was soft-deleted (UTC)",
    )

    def soft_delete(self) -> None:
        """
        Mark this record as soft-deleted.

        Sets is_deleted to True and records the deletion timestamp in UTC.
        The record remains in the database but must be excluded from queries.
        """
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)


# ==============================================================================
# Base Model — Composite Mixin for all application models
# ==============================================================================

class BaseModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Abstract base model that all application ORM models must inherit from.

    Combines:
        - UUIDPrimaryKeyMixin  → UUID v4 primary key (`id`)
        - TimestampMixin       → `created_at` and `updated_at` timestamps
        - Base                 → SQLAlchemy DeclarativeBase

    Marking this class as __abstract__ = True instructs SQLAlchemy not to
    create a table for BaseModel itself — only for its concrete subclasses.

    Usage:
        from app.database.base import BaseModel

        class BusinessModel(BaseModel):
            __tablename__ = "businesses"
            name: Mapped[str] = mapped_column(nullable=False)
    """

    __abstract__ = True

    def to_dict(self) -> dict:
        """
        Serialize the model instance to a plain Python dictionary.

        Converts UUID and datetime fields to their string representations
        for safe use in logging and JSON serialization contexts.

        Returns:
            dict: A key-value mapping of all column names to their values.
        """
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, uuid.UUID):
                value = str(value)
            elif isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result

    def __repr__(self) -> str:
        """
        Human-readable representation of the model instance.

        Returns the class name and primary key value for quick identification
        in logs and debugging sessions.
        """
        return f"<{self.__class__.__name__} id={self.id}>"


# ==============================================================================
# Soft-Deletable Base Model
# ==============================================================================

class SoftDeletableModel(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    """
    Abstract base model for entities that require soft-delete support.

    Extends BaseModel with SoftDeleteMixin.
    Use this for records that must never be permanently deleted from the
    database (e.g., businesses, subscriptions, payments).

    Usage:
        from app.database.base import SoftDeletableModel

        class SubscriptionModel(SoftDeletableModel):
            __tablename__ = "subscriptions"
            ...
    """

    __abstract__ = True

    def to_dict(self) -> dict:
        """
        Serialize the model instance to a plain Python dictionary,
        including soft-delete fields.

        Returns:
            dict: A key-value mapping of all column names to their values.
        """
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, uuid.UUID):
                value = str(value)
            elif isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id} deleted={self.is_deleted}>"