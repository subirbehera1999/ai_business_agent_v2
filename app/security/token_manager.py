# ==============================================================================
# File: app/security/token_manager.py
# Purpose: JWT token generation and verification for API authentication.
#
#          This system uses two types of tokens:
#
#          1. ACCESS TOKEN (short-lived, 60 minutes)
#             Sent with every API request in the Authorization header.
#             Contains: business_id, token type, issued-at, expires-at.
#             Verified by auth_middleware.py on every protected endpoint.
#
#          2. REFRESH TOKEN (long-lived, 30 days)
#             Stored securely by the client. Used only to obtain a new
#             access token when the current one expires. NOT sent with
#             every request — only to the /auth/refresh endpoint.
#             Stored in the database so it can be revoked.
#
#          Why two tokens instead of one long-lived token?
#            A single long-lived token is a security risk — if stolen,
#            the attacker has access for 30 days with no way to stop them.
#            Short access tokens (60 min) limit the damage window.
#            Refresh tokens are stored server-side, so they can be
#            invalidated immediately by deleting the DB record (logout,
#            account compromise, subscription cancellation).
#
#          Token blacklist:
#            Tokens that have been explicitly revoked (logout, password
#            change, subscription cancellation) are stored in a blacklist
#            table. Every token verification checks this table.
#            Blacklist entries are cleaned up after the token's natural
#            expiry time passes — no point keeping them forever.
#
#          Signing algorithm:
#            HS256 (HMAC-SHA256) using the JWT_SECRET_KEY env variable.
#            The secret key must be at least 32 characters.
#            Generate with:
#              python -c "import secrets; print(secrets.token_urlsafe(64))"
#
#          Token payload (claims):
#            sub:         business_id (subject)
#            type:        "access" or "refresh"
#            iat:         issued-at (Unix timestamp)
#            exp:         expires-at (Unix timestamp)
#            jti:         unique token ID (JWT ID) — used for blacklisting
#
#          Usage:
#            from app.security.token_manager import TokenManager
#
#            manager = TokenManager()
#
#            # On login / subscription activation:
#            token_pair = await manager.create_token_pair(
#                db=db, business_id="uuid"
#            )
#
#            # On every API request (done by auth_middleware.py):
#            payload = await manager.verify_access_token(
#                db=db, token="Bearer eyJ..."
#            )
#
#            # On token refresh:
#            new_pair = await manager.refresh_tokens(
#                db=db, refresh_token="eyJ..."
#            )
#
#            # On logout:
#            await manager.revoke_tokens(db=db, business_id="uuid")
# ==============================================================================

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from sqlalchemy import Column, DateTime, String, Text, func, select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName
from app.database.base import Base
from app.utils.time_utils import now_utc

logger = logging.getLogger(ServiceName.SECURITY)

# ---------------------------------------------------------------------------
# Token lifetime configuration
# ---------------------------------------------------------------------------
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
)
REFRESH_TOKEN_EXPIRE_DAYS: int = int(
    os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30")
)
JWT_ALGORITHM: str = "HS256"

# Token type constants
_ACCESS = "access"
_REFRESH = "refresh"


# ==============================================================================
# Database models for refresh tokens and blacklist
# ==============================================================================

class RefreshTokenRecord(Base):
    """
    Persisted refresh token for server-side validation and revocation.

    Storing refresh tokens server-side means we can revoke them
    immediately (logout, account cancellation) rather than waiting
    for natural expiry.
    """
    __tablename__ = "refresh_tokens"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    jti = Column(String(64), nullable=False, unique=True, index=True)
    business_id = Column(String(36), nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<RefreshTokenRecord "
            f"business_id={self.business_id} "
            f"expires_at={self.expires_at}>"
        )


class TokenBlacklist(Base):
    """
    Blacklisted token JTIs — tokens that have been explicitly revoked.

    Both access and refresh token JTIs are blacklisted on logout or
    forced revocation. Entries are deleted after the token's natural
    expiry to keep the table lean.
    """
    __tablename__ = "token_blacklist"

    jti = Column(String(64), primary_key=True)
    business_id = Column(String(36), nullable=False, index=True)
    token_type = Column(String(16), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<TokenBlacklist "
            f"jti={self.jti[:8]}... "
            f"business_id={self.business_id}>"
        )


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class TokenPair:
    """
    Access + refresh token pair returned on login or refresh.

    Attributes:
        access_token:       Short-lived JWT for API requests.
        refresh_token:      Long-lived JWT for obtaining new access tokens.
        access_expires_at:  UTC datetime when access token expires.
        refresh_expires_at: UTC datetime when refresh token expires.
        token_type:         Always "bearer".
    """
    access_token: str
    refresh_token: str
    access_expires_at: datetime
    refresh_expires_at: datetime
    token_type: str = "bearer"


@dataclass
class TokenPayload:
    """
    Decoded and validated JWT payload.

    Attributes:
        business_id:  Business UUID (subject claim).
        jti:          Unique token ID.
        token_type:   "access" or "refresh".
        issued_at:    When the token was issued (UTC).
        expires_at:   When the token expires (UTC).
    """
    business_id: str
    jti: str
    token_type: str
    issued_at: datetime
    expires_at: datetime


@dataclass
class VerificationResult:
    """
    Result of token verification.

    Attributes:
        valid:       True if token passed all checks.
        payload:     Decoded payload (None if invalid).
        error:       Reason for failure (None if valid).
    """
    valid: bool
    payload: Optional[TokenPayload] = None
    error: Optional[str] = None


# ==============================================================================
# Token Manager
# ==============================================================================

class TokenManager:
    """
    Generates, verifies, and revokes JWT tokens for business API access.

    All public methods are async because they interact with the database
    for refresh token storage, blacklist checks, and cleanup.

    Raises:
        TokenConfigError: At init if JWT_SECRET_KEY is missing or too short.
    """

    def __init__(self) -> None:
        secret = os.getenv("JWT_SECRET_KEY", "")
        if not secret:
            raise TokenConfigError(
                "JWT_SECRET_KEY is not set. "
                "Generate with: python -c \"import secrets; "
                "print(secrets.token_urlsafe(64))\""
            )
        if len(secret) < 32:
            raise TokenConfigError(
                f"JWT_SECRET_KEY is too short ({len(secret)} chars). "
                f"Minimum 32 characters required for HS256 security."
            )
        self._secret = secret

    # ------------------------------------------------------------------
    # Create token pair
    # ------------------------------------------------------------------

    async def create_token_pair(
        self,
        db: AsyncSession,
        business_id: str,
    ) -> TokenPair:
        """
        Generate a new access + refresh token pair for a business.

        Stores the refresh token in the database.
        Called after successful payment/login.

        Args:
            db:           AsyncSession.
            business_id:  Business UUID.

        Returns:
            TokenPair with both tokens and expiry datetimes.

        Raises:
            TokenCreationError: If token generation or DB storage fails.
        """
        try:
            now = now_utc()

            # Access token
            access_jti = _new_jti()
            access_expires = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = self._sign(
                business_id=business_id,
                jti=access_jti,
                token_type=_ACCESS,
                issued_at=now,
                expires_at=access_expires,
            )

            # Refresh token
            refresh_jti = _new_jti()
            refresh_expires = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
            refresh_token = self._sign(
                business_id=business_id,
                jti=refresh_jti,
                token_type=_REFRESH,
                issued_at=now,
                expires_at=refresh_expires,
            )

            # Persist refresh token record
            # Delete any existing refresh tokens for this business first
            # (one active refresh token per business)
            await self._delete_existing_refresh_tokens(
                db=db, business_id=business_id
            )

            record = RefreshTokenRecord(
                jti=refresh_jti,
                business_id=business_id,
                expires_at=refresh_expires,
            )
            db.add(record)
            await db.commit()

            logger.info(
                "Token pair created",
                extra={
                    "service": ServiceName.SECURITY,
                    "business_id": business_id,
                    "access_expires_at": access_expires.isoformat(),
                    "refresh_expires_at": refresh_expires.isoformat(),
                },
            )

            return TokenPair(
                access_token=access_token,
                refresh_token=refresh_token,
                access_expires_at=access_expires,
                refresh_expires_at=refresh_expires,
            )

        except TokenCreationError:
            raise
        except Exception as exc:
            await db.rollback()
            raise TokenCreationError(
                f"Failed to create token pair: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Verify access token
    # ------------------------------------------------------------------

    async def verify_access_token(
        self,
        db: AsyncSession,
        token: str,
    ) -> VerificationResult:
        """
        Verify an access token from an API request.

        Checks performed (in order):
          1. Strip "Bearer " prefix if present
          2. Decode and verify JWT signature
          3. Verify token type is "access"
          4. Verify token has not expired (jwt library checks this)
          5. Check token JTI is not in the blacklist

        Called by auth_middleware.py on every protected request.

        Args:
            db:    AsyncSession.
            token: Raw Authorization header value or bare token string.

        Returns:
            VerificationResult. Never raises.
        """
        try:
            raw_token = _strip_bearer(token)

            # Decode and verify signature + expiry
            payload = self._decode(raw_token)
            if not payload:
                return VerificationResult(
                    valid=False,
                    error="invalid_token: malformed or expired",
                )

            # Must be an access token
            if payload.token_type != _ACCESS:
                return VerificationResult(
                    valid=False,
                    error="invalid_token_type: expected access token",
                )

            # Check blacklist
            if await self._is_blacklisted(db=db, jti=payload.jti):
                return VerificationResult(
                    valid=False,
                    error="token_revoked: token has been invalidated",
                )

            return VerificationResult(valid=True, payload=payload)

        except Exception as exc:
            logger.warning(
                "Access token verification error",
                extra={
                    "service": ServiceName.SECURITY,
                    "error": str(exc),
                },
            )
            return VerificationResult(
                valid=False,
                error=f"verification_error: {exc}",
            )

    # ------------------------------------------------------------------
    # Refresh tokens
    # ------------------------------------------------------------------

    async def refresh_tokens(
        self,
        db: AsyncSession,
        refresh_token: str,
    ) -> Optional[TokenPair]:
        """
        Exchange a valid refresh token for a new access + refresh token pair.

        The old refresh token is invalidated (blacklisted) and a new pair
        is issued. This is called "refresh token rotation" — it means
        stolen refresh tokens can only be used once.

        Args:
            db:            AsyncSession.
            refresh_token: The refresh token string.

        Returns:
            New TokenPair, or None if the refresh token is invalid/expired.
        """
        try:
            payload = self._decode(refresh_token)
            if not payload:
                logger.warning(
                    "Refresh attempt with invalid token",
                    extra={"service": ServiceName.SECURITY},
                )
                return None

            if payload.token_type != _REFRESH:
                logger.warning(
                    "Refresh attempt with non-refresh token",
                    extra={
                        "service": ServiceName.SECURITY,
                        "token_type": payload.token_type,
                    },
                )
                return None

            # Check blacklist
            if await self._is_blacklisted(db=db, jti=payload.jti):
                logger.warning(
                    "Refresh attempt with revoked token",
                    extra={
                        "service": ServiceName.SECURITY,
                        "business_id": payload.business_id,
                    },
                )
                return None

            # Verify refresh token exists in DB
            db_record = await self._get_refresh_record(
                db=db, jti=payload.jti
            )
            if not db_record:
                logger.warning(
                    "Refresh token not found in database",
                    extra={
                        "service": ServiceName.SECURITY,
                        "business_id": payload.business_id,
                    },
                )
                return None

            # Blacklist the old refresh token (rotation)
            await self._blacklist_jti(
                db=db,
                jti=payload.jti,
                business_id=payload.business_id,
                token_type=_REFRESH,
                expires_at=payload.expires_at,
            )

            # Issue new token pair
            new_pair = await self.create_token_pair(
                db=db,
                business_id=payload.business_id,
            )

            logger.info(
                "Tokens refreshed",
                extra={
                    "service": ServiceName.SECURITY,
                    "business_id": payload.business_id,
                },
            )
            return new_pair

        except Exception as exc:
            logger.error(
                "Token refresh failed",
                extra={
                    "service": ServiceName.SECURITY,
                    "error": str(exc),
                },
            )
            return None

    # ------------------------------------------------------------------
    # Revoke tokens (logout / cancellation)
    # ------------------------------------------------------------------

    async def revoke_tokens(
        self,
        db: AsyncSession,
        business_id: str,
        access_token: Optional[str] = None,
    ) -> bool:
        """
        Revoke all tokens for a business (logout or subscription cancellation).

        Deletes the stored refresh token record and blacklists the
        current access token if provided.

        Args:
            db:            AsyncSession.
            business_id:   Business UUID.
            access_token:  Current access token to blacklist (optional).

        Returns:
            True if revocation was successful.
        """
        try:
            # Blacklist current access token if provided
            if access_token:
                raw = _strip_bearer(access_token)
                payload = self._decode(raw)
                if payload:
                    await self._blacklist_jti(
                        db=db,
                        jti=payload.jti,
                        business_id=business_id,
                        token_type=_ACCESS,
                        expires_at=payload.expires_at,
                    )

            # Delete all refresh tokens for this business
            await self._delete_existing_refresh_tokens(
                db=db, business_id=business_id
            )

            await db.commit()

            logger.info(
                "Tokens revoked",
                extra={
                    "service": ServiceName.SECURITY,
                    "business_id": business_id,
                },
            )
            return True

        except Exception as exc:
            await db.rollback()
            logger.error(
                "Token revocation failed",
                extra={
                    "service": ServiceName.SECURITY,
                    "business_id": business_id,
                    "error": str(exc),
                },
            )
            return False

    # ------------------------------------------------------------------
    # Cleanup — called by daily maintenance job
    # ------------------------------------------------------------------

    async def cleanup_expired_tokens(self, db: AsyncSession) -> dict:
        """
        Delete expired refresh token records and blacklist entries.

        Called by a daily maintenance scheduler to keep token tables lean.
        Expired tokens are already invalid — no need to keep their records.

        Args:
            db: AsyncSession.

        Returns:
            Dict with counts: {"refresh_deleted": N, "blacklist_deleted": N}
        """
        try:
            now = now_utc()

            # Delete expired refresh tokens
            refresh_stmt = delete(RefreshTokenRecord).where(
                RefreshTokenRecord.expires_at < now
            )
            refresh_result = await db.execute(refresh_stmt)

            # Delete expired blacklist entries
            blacklist_stmt = delete(TokenBlacklist).where(
                TokenBlacklist.expires_at < now
            )
            blacklist_result = await db.execute(blacklist_stmt)

            await db.commit()

            deleted_refresh = refresh_result.rowcount or 0
            deleted_blacklist = blacklist_result.rowcount or 0

            logger.info(
                "Expired tokens cleaned up",
                extra={
                    "service": ServiceName.SECURITY,
                    "refresh_deleted": deleted_refresh,
                    "blacklist_deleted": deleted_blacklist,
                },
            )

            return {
                "refresh_deleted": deleted_refresh,
                "blacklist_deleted": deleted_blacklist,
            }

        except Exception as exc:
            await db.rollback()
            logger.error(
                "Token cleanup failed",
                extra={
                    "service": ServiceName.SECURITY,
                    "error": str(exc),
                },
            )
            return {"refresh_deleted": 0, "blacklist_deleted": 0}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sign(
        self,
        business_id: str,
        jti: str,
        token_type: str,
        issued_at: datetime,
        expires_at: datetime,
    ) -> str:
        """Sign and return a JWT string."""
        payload = {
            "sub": business_id,
            "jti": jti,
            "type": token_type,
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        return jwt.encode(payload, self._secret, algorithm=JWT_ALGORITHM)

    def _decode(self, token: str) -> Optional[TokenPayload]:
        """
        Decode and validate a JWT. Returns None on any failure.

        PyJWT automatically verifies signature, expiry (exp), and
        issued-at (iat) claims. We only need to extract and map the fields.
        """
        try:
            raw = jwt.decode(
                token,
                self._secret,
                algorithms=[JWT_ALGORITHM],
                options={"require": ["sub", "jti", "type", "iat", "exp"]},
            )
            return TokenPayload(
                business_id=raw["sub"],
                jti=raw["jti"],
                token_type=raw["type"],
                issued_at=datetime.fromtimestamp(raw["iat"], tz=timezone.utc),
                expires_at=datetime.fromtimestamp(raw["exp"], tz=timezone.utc),
            )
        except jwt.ExpiredSignatureError:
            logger.debug(
                "Token expired",
                extra={"service": ServiceName.SECURITY},
            )
            return None
        except jwt.InvalidTokenError as exc:
            logger.debug(
                "Invalid token",
                extra={"service": ServiceName.SECURITY, "error": str(exc)},
            )
            return None

    async def _is_blacklisted(self, db: AsyncSession, jti: str) -> bool:
        """Return True if the JTI is in the blacklist table."""
        try:
            stmt = select(TokenBlacklist.jti).where(
                TokenBlacklist.jti == jti
            ).limit(1)
            result = await db.execute(stmt)
            return result.scalar_one_or_none() is not None
        except Exception as exc:
            logger.error(
                "Blacklist check failed — denying token by default",
                extra={
                    "service": ServiceName.SECURITY,
                    "error": str(exc),
                },
            )
            return True   # fail-closed: deny on error

    async def _blacklist_jti(
        self,
        db: AsyncSession,
        jti: str,
        business_id: str,
        token_type: str,
        expires_at: datetime,
    ) -> None:
        """Insert a JTI into the blacklist. Ignores duplicates."""
        try:
            entry = TokenBlacklist(
                jti=jti,
                business_id=business_id,
                token_type=token_type,
                expires_at=expires_at,
            )
            db.add(entry)
            await db.flush()
        except Exception:
            # Duplicate insert (already blacklisted) — safe to ignore
            pass

    async def _get_refresh_record(
        self,
        db: AsyncSession,
        jti: str,
    ) -> Optional[RefreshTokenRecord]:
        """Fetch a refresh token record by JTI."""
        stmt = select(RefreshTokenRecord).where(
            RefreshTokenRecord.jti == jti
        ).limit(1)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def _delete_existing_refresh_tokens(
        self,
        db: AsyncSession,
        business_id: str,
    ) -> None:
        """Delete all stored refresh tokens for a business."""
        stmt = delete(RefreshTokenRecord).where(
            RefreshTokenRecord.business_id == business_id
        )
        await db.execute(stmt)


# ==============================================================================
# Pure helpers
# ==============================================================================

def _new_jti() -> str:
    """Generate a unique JWT ID (128-bit random hex string)."""
    return uuid.uuid4().hex


def _strip_bearer(token: str) -> str:
    """
    Remove the 'Bearer ' prefix from an Authorization header value.

    Handles both 'Bearer eyJ...' and bare 'eyJ...' formats.
    """
    if token and token.lower().startswith("bearer "):
        return token[7:].strip()
    return token.strip() if token else ""


# ==============================================================================
# Exceptions
# ==============================================================================

class TokenConfigError(Exception):
    """
    Raised at startup when JWT_SECRET_KEY is missing or too short.
    Prevents the application from running without a valid signing key.
    """


class TokenCreationError(Exception):
    """Raised when token generation or database storage fails."""


class TokenVerificationError(Exception):
    """Raised when token verification encounters an unexpected error."""