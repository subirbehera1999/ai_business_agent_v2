# ==============================================================================
# File: app/security/encryption.py
# Purpose: Symmetric encryption for sensitive data stored in the database.
#
#          What gets encrypted:
#            Certain fields in the database contain sensitive business data
#            that must not be stored in plain text. If the database were ever
#            compromised, encrypted fields would be unreadable without the key.
#
#          Fields that should be encrypted before storing:
#            - Business WhatsApp number
#            - Business phone number
#            - Google Sheets URL (contains access tokens in some formats)
#            - Google location ID (can be used to impersonate the business)
#            - Any API credentials stored per-business
#
#          Encryption algorithm:
#            AES-256-GCM (Galois/Counter Mode)
#
#            Why AES-256-GCM:
#              - AES-256: industry standard, FIPS-approved, 256-bit key
#              - GCM mode: provides both encryption AND authentication
#                (detects tampering — if ciphertext is modified, decryption fails)
#              - Much better than AES-CBC which only encrypts, not authenticates
#
#          Stored format:
#            Encrypted values are stored as base64-encoded strings in this format:
#              v1:<base64(nonce)>:<base64(ciphertext+tag)>
#
#            The "v1:" prefix enables key rotation — future versions can use
#            "v2:", "v3:" while still decrypting old values with their original key.
#
#          Key rotation:
#            The encryption key is loaded from ENCRYPTION_KEY env var.
#            A secondary ENCRYPTION_KEY_PREVIOUS env var can hold an older key.
#            On decryption, the system tries the current key first, then
#            the previous key — allowing gradual re-encryption without downtime.
#
#          Deterministic encryption (for searchable fields):
#            Standard AES-GCM uses a random nonce each time, meaning the same
#            plaintext encrypts to a different ciphertext every time.
#            This is secure but makes database lookups impossible
#            (you can't find a business by phone number if the encrypted value
#            is different every time).
#
#            For fields that need to be searchable (e.g. looking up a business
#            by WhatsApp number), use encrypt_deterministic() which uses
#            a fixed HMAC-derived nonce. Same plaintext → same ciphertext.
#            This is a deliberate security tradeoff — only use for lookup fields.
#
#          Key requirements:
#            ENCRYPTION_KEY must be exactly 32 bytes when base64-decoded.
#            Generate with:
#              python -c "import secrets, base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
#
#          Usage:
#            from app.security.encryption import EncryptionService
#
#            svc = EncryptionService()
#
#            # Standard (random nonce — different ciphertext each call):
#            encrypted = svc.encrypt("9876543210")
#            original  = svc.decrypt(encrypted)
#
#            # Deterministic (fixed nonce — same ciphertext each call):
#            encrypted = svc.encrypt_deterministic("9876543210")
#            original  = svc.decrypt(encrypted)   # same decrypt() call
# ==============================================================================

import base64
import hashlib
import hmac
import logging
import os
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.config.constants import ServiceName

logger = logging.getLogger(ServiceName.SECURITY)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ENCRYPTION_VERSION = "v1"
_NONCE_BYTES = 12          # AES-GCM standard nonce length (96 bits)
_KEY_BYTES = 32            # AES-256 requires exactly 32 bytes
_SEPARATOR = ":"           # separates version:nonce:ciphertext in stored value


# ==============================================================================
# Encryption Service
# ==============================================================================

class EncryptionService:
    """
    AES-256-GCM encryption service for sensitive database fields.

    Loads encryption key(s) from environment variables at initialisation.
    All encrypt/decrypt methods are synchronous — they perform pure CPU
    operations with no I/O and do not need to be async.

    Environment variables:
        ENCRYPTION_KEY:          Required. Current active encryption key.
                                 Must be base64url-encoded 32-byte value.
        ENCRYPTION_KEY_PREVIOUS: Optional. Previous key for rotation support.

    Raises:
        EncryptionKeyError: On init if ENCRYPTION_KEY is missing or invalid.
    """

    def __init__(self) -> None:
        self._current_key: bytes = _load_key("ENCRYPTION_KEY", required=True)
        self._previous_key: Optional[bytes] = _load_key(
            "ENCRYPTION_KEY_PREVIOUS", required=False
        )

        # Build AESGCM cipher objects — created once, reused for all operations
        self._current_cipher = AESGCM(self._current_key)
        self._previous_cipher = (
            AESGCM(self._previous_key) if self._previous_key else None
        )

        logger.info(
            "EncryptionService initialised",
            extra={
                "service": ServiceName.SECURITY,
                "key_rotation_enabled": self._previous_key is not None,
            },
        )

    # ------------------------------------------------------------------
    # Standard encryption (random nonce — different output each call)
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a plaintext string using AES-256-GCM with a random nonce.

        Each call produces a different ciphertext even for the same input.
        Use this for fields that do NOT need to be searchable.

        Args:
            plaintext: The string to encrypt.

        Returns:
            Encrypted string in format: v1:<nonce_b64>:<ciphertext_b64>

        Raises:
            EncryptionError: If encryption fails.
        """
        if not plaintext:
            return plaintext  # preserve empty strings and None as-is

        try:
            nonce = os.urandom(_NONCE_BYTES)
            ciphertext = self._current_cipher.encrypt(
                nonce,
                plaintext.encode("utf-8"),
                None,  # no additional authenticated data
            )
            return _pack(nonce, ciphertext)

        except Exception as exc:
            raise EncryptionError(f"Encryption failed: {exc}") from exc

    def decrypt(self, ciphertext_str: str) -> str:
        """
        Decrypt a value previously encrypted by encrypt() or encrypt_deterministic().

        Tries the current key first. If decryption fails (wrong key),
        falls back to the previous key to support key rotation.

        Args:
            ciphertext_str: Encrypted string in v1:<nonce_b64>:<ciphertext_b64> format.

        Returns:
            Original plaintext string.

        Raises:
            DecryptionError: If decryption fails with all available keys.
            EncryptionFormatError: If the stored value format is invalid.
        """
        if not ciphertext_str:
            return ciphertext_str  # preserve empty/None

        # If value looks unencrypted (no version prefix), return as-is
        # This handles migration: old plain-text fields can be read safely
        if not ciphertext_str.startswith(f"{_ENCRYPTION_VERSION}{_SEPARATOR}"):
            return ciphertext_str

        try:
            nonce, raw_ciphertext = _unpack(ciphertext_str)
        except Exception as exc:
            raise EncryptionFormatError(
                f"Invalid encrypted value format: {exc}"
            ) from exc

        # Try current key
        try:
            plaintext_bytes = self._current_cipher.decrypt(
                nonce, raw_ciphertext, None
            )
            return plaintext_bytes.decode("utf-8")
        except InvalidTag:
            pass  # wrong key or tampered — try previous key

        # Try previous key (key rotation fallback)
        if self._previous_cipher:
            try:
                plaintext_bytes = self._previous_cipher.decrypt(
                    nonce, raw_ciphertext, None
                )
                logger.debug(
                    "Decrypted with previous key — re-encrypt with current key on next write",
                    extra={"service": ServiceName.SECURITY},
                )
                return plaintext_bytes.decode("utf-8")
            except InvalidTag:
                pass

        raise DecryptionError(
            "Decryption failed: invalid key or tampered ciphertext. "
            "Check ENCRYPTION_KEY and ENCRYPTION_KEY_PREVIOUS."
        )

    # ------------------------------------------------------------------
    # Deterministic encryption (fixed nonce — same output each call)
    # ------------------------------------------------------------------

    def encrypt_deterministic(self, plaintext: str) -> str:
        """
        Encrypt using a deterministic nonce derived from the plaintext.

        Same input always produces the same output — enabling database
        lookups (WHERE encrypted_phone = encrypt_deterministic(search_value)).

        Security tradeoff:
            Deterministic encryption leaks that two values are identical
            (an attacker can see that business A and B have the same phone
            number without knowing what it is). Only use for fields where
            this tradeoff is acceptable (lookup keys, not secrets).

        Args:
            plaintext: The string to encrypt.

        Returns:
            Encrypted string in format: v1:<nonce_b64>:<ciphertext_b64>
            (identical format to encrypt() — same decrypt() call works)

        Raises:
            EncryptionError: If encryption fails.
        """
        if not plaintext:
            return plaintext

        try:
            # Derive a fixed nonce from HMAC(key, plaintext)
            # This is deterministic: same key + same plaintext = same nonce
            nonce = _derive_deterministic_nonce(
                key=self._current_key,
                plaintext=plaintext,
            )
            ciphertext = self._current_cipher.encrypt(
                nonce,
                plaintext.encode("utf-8"),
                None,
            )
            return _pack(nonce, ciphertext)

        except Exception as exc:
            raise EncryptionError(
                f"Deterministic encryption failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Batch operations — for encrypting/decrypting multiple fields
    # ------------------------------------------------------------------

    def encrypt_fields(self, data: dict, fields: tuple[str, ...]) -> dict:
        """
        Encrypt specified fields in a dictionary in-place.

        Returns a new dict with the specified fields encrypted.
        Non-specified fields are copied unchanged.

        Args:
            data:   Source dictionary.
            fields: Tuple of field names to encrypt.

        Returns:
            New dict with encrypted fields.

        Example:
            safe = encryption.encrypt_fields(
                data={"phone": "9876543210", "name": "Cafe Sunrise"},
                fields=("phone",),
            )
            # safe["phone"] is now encrypted; safe["name"] is unchanged
        """
        result = dict(data)
        for field in fields:
            if field in result and result[field]:
                try:
                    result[field] = self.encrypt(str(result[field]))
                except EncryptionError as exc:
                    logger.warning(
                        "Failed to encrypt field",
                        extra={
                            "service": ServiceName.SECURITY,
                            "field": field,
                            "error": str(exc),
                        },
                    )
        return result

    def decrypt_fields(self, data: dict, fields: tuple[str, ...]) -> dict:
        """
        Decrypt specified fields in a dictionary.

        Returns a new dict with the specified fields decrypted.
        Fields that are not encrypted (no v1: prefix) are returned as-is —
        this allows safe operation during migration when some rows are
        already encrypted and others are not yet.

        Args:
            data:   Source dictionary.
            fields: Tuple of field names to decrypt.

        Returns:
            New dict with decrypted fields.
        """
        result = dict(data)
        for field in fields:
            if field in result and result[field]:
                try:
                    result[field] = self.decrypt(str(result[field]))
                except (DecryptionError, EncryptionFormatError) as exc:
                    logger.warning(
                        "Failed to decrypt field — returning raw value",
                        extra={
                            "service": ServiceName.SECURITY,
                            "field": field,
                            "error": str(exc),
                        },
                    )
        return result

    # ------------------------------------------------------------------
    # Key rotation utility
    # ------------------------------------------------------------------

    def needs_re_encryption(self, ciphertext_str: str) -> bool:
        """
        Return True if a stored value was encrypted with the previous key.

        Use this to identify fields that need to be re-encrypted with
        the current key during a key rotation migration.

        Args:
            ciphertext_str: Encrypted value from the database.

        Returns:
            True if the value was encrypted with ENCRYPTION_KEY_PREVIOUS.
        """
        if not ciphertext_str or not self._previous_cipher:
            return False

        if not ciphertext_str.startswith(f"{_ENCRYPTION_VERSION}{_SEPARATOR}"):
            return False

        try:
            nonce, raw_ciphertext = _unpack(ciphertext_str)
        except Exception:
            return False

        # Try to decrypt with previous key
        try:
            self._previous_cipher.decrypt(nonce, raw_ciphertext, None)
            return True   # decrypted with previous key → needs re-encryption
        except InvalidTag:
            return False  # decrypts with current key (or invalid) → no rotation needed

    def re_encrypt(self, ciphertext_str: str) -> str:
        """
        Decrypt with the previous key and re-encrypt with the current key.

        Used during key rotation to migrate stored values.

        Args:
            ciphertext_str: Value encrypted with the previous key.

        Returns:
            Same plaintext re-encrypted with the current key.

        Raises:
            DecryptionError: If value cannot be decrypted.
            EncryptionError: If re-encryption fails.
        """
        plaintext = self.decrypt(ciphertext_str)
        return self.encrypt(plaintext)


# ==============================================================================
# Pure helpers
# ==============================================================================

def _load_key(env_var: str, required: bool) -> Optional[bytes]:
    """
    Load and validate an encryption key from an environment variable.

    The key must be a base64url-encoded 32-byte value.

    Args:
        env_var:  Environment variable name.
        required: Raise EncryptionKeyError if True and var is missing.

    Returns:
        32-byte key as bytes, or None if env_var is not set and not required.

    Raises:
        EncryptionKeyError: If key is missing (when required) or invalid.
    """
    raw = os.getenv(env_var)

    if not raw:
        if required:
            raise EncryptionKeyError(
                f"{env_var} is not set. "
                f"Generate with: python -c \""
                f"import secrets, base64; "
                f"print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
                f"\""
            )
        return None

    try:
        # Accept both standard and URL-safe base64
        key_bytes = base64.urlsafe_b64decode(
            raw + "==" * (4 - len(raw) % 4)  # pad if necessary
        )
    except Exception as exc:
        raise EncryptionKeyError(
            f"{env_var} is not valid base64: {exc}"
        ) from exc

    if len(key_bytes) != _KEY_BYTES:
        raise EncryptionKeyError(
            f"{env_var} must decode to exactly {_KEY_BYTES} bytes "
            f"(got {len(key_bytes)}). "
            f"AES-256 requires a 256-bit key."
        )

    return key_bytes


def _pack(nonce: bytes, ciphertext: bytes) -> str:
    """
    Pack nonce and ciphertext into the stored string format.

    Format: v1:<nonce_base64url>:<ciphertext_base64url>
    """
    nonce_b64 = base64.urlsafe_b64encode(nonce).decode("ascii")
    ct_b64 = base64.urlsafe_b64encode(ciphertext).decode("ascii")
    return f"{_ENCRYPTION_VERSION}{_SEPARATOR}{nonce_b64}{_SEPARATOR}{ct_b64}"


def _unpack(packed: str) -> tuple[bytes, bytes]:
    """
    Unpack stored string format back into (nonce, ciphertext) bytes.

    Args:
        packed: v1:<nonce_base64url>:<ciphertext_base64url>

    Returns:
        Tuple of (nonce_bytes, ciphertext_bytes).

    Raises:
        EncryptionFormatError: If the format is unexpected.
    """
    parts = packed.split(_SEPARATOR, 2)
    if len(parts) != 3:
        raise EncryptionFormatError(
            f"Expected 3 parts separated by '{_SEPARATOR}', got {len(parts)}"
        )

    version, nonce_b64, ct_b64 = parts

    if version != _ENCRYPTION_VERSION:
        raise EncryptionFormatError(
            f"Unknown encryption version '{version}'. "
            f"Only '{_ENCRYPTION_VERSION}' is supported."
        )

    try:
        nonce = base64.urlsafe_b64decode(nonce_b64 + "==")
        ciphertext = base64.urlsafe_b64decode(ct_b64 + "==")
    except Exception as exc:
        raise EncryptionFormatError(
            f"Failed to base64-decode encrypted value: {exc}"
        ) from exc

    if len(nonce) != _NONCE_BYTES:
        raise EncryptionFormatError(
            f"Invalid nonce length: expected {_NONCE_BYTES}, got {len(nonce)}"
        )

    return nonce, ciphertext


def _derive_deterministic_nonce(key: bytes, plaintext: str) -> bytes:
    """
    Derive a fixed 12-byte nonce from the key and plaintext using HMAC-SHA256.

    Same key + same plaintext → same 12-byte nonce, every time.
    This makes the encryption deterministic — identical inputs produce
    identical ciphertexts, enabling database lookups.

    Security note:
        Using HMAC(key, plaintext) as the nonce means the nonce is
        secret (derived from the key) — preventing nonce-reuse attacks
        that would break AES-GCM confidentiality.
    """
    h = hmac.new(key=key, msg=plaintext.encode("utf-8"), digestmod=hashlib.sha256)
    # Take first 12 bytes of the 32-byte HMAC digest
    return h.digest()[:_NONCE_BYTES]


# ==============================================================================
# Exceptions
# ==============================================================================

class EncryptionKeyError(Exception):
    """
    Raised when the encryption key is missing, invalid, or wrong length.
    Raised at application startup — prevents the app from running without
    a valid encryption key configured.
    """


class EncryptionError(Exception):
    """Raised when encryption of a value fails."""


class DecryptionError(Exception):
    """
    Raised when decryption fails.
    Usually means the data was encrypted with a different key,
    or the ciphertext has been tampered with.
    """


class EncryptionFormatError(Exception):
    """
    Raised when a stored value does not match the expected format.
    Usually means a non-encrypted value was passed to decrypt(),
    or the stored value is corrupted.
    """