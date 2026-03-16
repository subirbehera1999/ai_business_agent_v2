# ==============================================================================
# File: app/database/db.py
# Purpose: Configures the async SQLAlchemy engine, session factory, and
#          provides the FastAPI dependency for injecting database sessions
#          into API routes and services. This is the single entry point for
#          all database connectivity in the application.
# ==============================================================================

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.constants import DB_HEALTH_PING_QUERY, ServiceName
from app.config.settings import get_settings

logger = logging.getLogger(ServiceName.API)

settings = get_settings()


# ==============================================================================
# Async Engine
# ==============================================================================

def _build_engine() -> AsyncEngine:
    """
    Construct and return the async SQLAlchemy engine.

    Engine is configured with:
    - Connection pool sized according to settings
    - Pre-ping enabled to detect and recycle stale connections
    - JSON serialisation via the default asyncpg codec
    - Echo disabled in production; configurable via DATABASE_ECHO

    Returns:
        AsyncEngine: Configured SQLAlchemy async engine instance.
    """
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_timeout=settings.DATABASE_POOL_TIMEOUT,
        pool_pre_ping=True,          # Verify connection health before use
        pool_recycle=1800,           # Recycle connections after 30 minutes
        connect_args={
            "server_settings": {
                "application_name": settings.APP_NAME,
            }
        },
    )

    logger.info(
        "Database engine created",
        extra={
            "service": ServiceName.API,
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
        },
    )

    return engine


# Module-level engine singleton — created once, shared across the application.
engine: AsyncEngine = _build_engine()


# ==============================================================================
# Async Session Factory
# ==============================================================================

AsyncSessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # Prevent expired-instance errors after commit
    autocommit=False,
    autoflush=False,
)
"""
Session factory bound to the async engine.

expire_on_commit=False is intentional — without it, accessing attributes on
a model instance after session.commit() would trigger lazy-load errors in
async contexts where the session is already closed.
"""


# ==============================================================================
# FastAPI Dependency — Request-scoped Session
# ==============================================================================

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a request-scoped async database session.

    Lifecycle:
    - A new session is opened at the start of each request.
    - The session is committed automatically if the handler completes without
      raising an exception.
    - On any exception the transaction is rolled back before re-raising.
    - The session is always closed in the finally block regardless of outcome.

    This ensures that:
    - Each request operates in its own isolated transaction.
    - No partial writes are committed on failure.
    - Connection pool resources are returned promptly after each request.

    Usage in route:
        from fastapi import Depends
        from sqlalchemy.ext.asyncio import AsyncSession
        from app.database.db import get_db_session

        @router.get("/example")
        async def example_route(db: AsyncSession = Depends(get_db_session)):
            result = await db.execute(select(MyModel))
            return result.scalars().all()

    Yields:
        AsyncSession: An active database session for the duration of the request.

    Raises:
        SQLAlchemyError: Re-raised after rollback if a database error occurs.
        Exception: Any other exception is re-raised after rollback.
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as exc:
            await session.rollback()
            logger.error(
                "Database error — transaction rolled back",
                extra={
                    "service": ServiceName.API,
                    "error": str(exc),
                },
            )
            raise
        except Exception as exc:
            await session.rollback()
            logger.error(
                "Unexpected error — transaction rolled back",
                extra={
                    "service": ServiceName.API,
                    "error": str(exc),
                },
            )
            raise
        finally:
            await session.close()


# ==============================================================================
# Context Manager — Service / Background Job Session
# ==============================================================================

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that provides a managed database session for use
    outside of the FastAPI request lifecycle.

    Intended for:
    - Background scheduler jobs
    - Service-layer utilities
    - Admin scripts
    - System health checks

    Lifecycle mirrors get_db_session() — commit on success, rollback on error,
    always closes the session.

    Usage:
        from app.database.db import get_session

        async with get_session() as db:
            result = await db.execute(select(MyModel))
            records = result.scalars().all()

    Yields:
        AsyncSession: An active database session.

    Raises:
        SQLAlchemyError: Re-raised after rollback if a database error occurs.
        Exception: Any other exception is re-raised after rollback.
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as exc:
            await session.rollback()
            logger.error(
                "Database error in background session — transaction rolled back",
                extra={
                    "service": ServiceName.SCHEDULER,
                    "error": str(exc),
                },
            )
            raise
        except Exception as exc:
            await session.rollback()
            logger.error(
                "Unexpected error in background session — transaction rolled back",
                extra={
                    "service": ServiceName.SCHEDULER,
                    "error": str(exc),
                },
            )
            raise
        finally:
            await session.close()


# ==============================================================================
# Database Health Check
# ==============================================================================

async def check_database_health() -> bool:
    """
    Verify that the database is reachable and accepting queries.

    Executes a lightweight ping query (SELECT 1) against the live connection
    pool. Used by system_health.py and the /health endpoint.

    Returns:
        bool: True if the database responded successfully, False otherwise.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text(DB_HEALTH_PING_QUERY))
        logger.debug(
            "Database health check passed",
            extra={"service": ServiceName.SYSTEM_HEALTH},
        )
        return True
    except OperationalError as exc:
        logger.error(
            "Database health check failed — OperationalError",
            extra={
                "service": ServiceName.SYSTEM_HEALTH,
                "error": str(exc),
            },
        )
        return False
    except Exception as exc:
        logger.error(
            "Database health check failed — unexpected error",
            extra={
                "service": ServiceName.SYSTEM_HEALTH,
                "error": str(exc),
            },
        )
        return False


# ==============================================================================
# Engine Lifecycle — Startup and Shutdown
# ==============================================================================

async def connect_database() -> None:
    """
    Validate database connectivity at application startup.

    Called from the FastAPI lifespan context in main.py.
    Raises RuntimeError if the database cannot be reached — this intentionally
    prevents the application from starting in a broken state.

    Raises:
        RuntimeError: If the database is unreachable at startup.
    """
    logger.info(
        "Verifying database connectivity at startup",
        extra={"service": ServiceName.API},
    )

    is_healthy = await check_database_health()

    if not is_healthy:
        raise RuntimeError(
            "Database is unreachable at startup. "
            "Check DATABASE_URL and ensure PostgreSQL is running."
        )

    logger.info(
        "Database connection verified successfully",
        extra={"service": ServiceName.API},
    )


async def disconnect_database() -> None:
    """
    Gracefully dispose of the engine connection pool at application shutdown.

    Called from the FastAPI lifespan context in main.py.
    Allows in-flight queries to complete before releasing all connections.
    """
    logger.info(
        "Disposing database engine connection pool",
        extra={"service": ServiceName.API},
    )
    await engine.dispose()
    logger.info(
        "Database engine disposed successfully",
        extra={"service": ServiceName.API},
    )


# ==============================================================================
# Raw Connection Accessor (for Alembic migrations and admin scripts only)
# ==============================================================================
@asynccontextmanager
async def get_raw_connection() -> AsyncGenerator[AsyncConnection, None]:
    """
    Yield a raw async connection from the engine pool.

    This is provided strictly for:
    - Alembic migration scripts that require a raw DBAPI connection
    - Low-level admin operations that cannot use the ORM session

    Application code (routes, services, repositories) must NEVER use this.
    Always use get_db_session() or get_session() instead.

    Yields:
        AsyncConnection: A raw SQLAlchemy async connection.
    """
    async with engine.connect() as conn:
        yield conn



# Alias used by auth_middleware.py and any route that imports get_db directly
get_db = get_db_session