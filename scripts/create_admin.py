#!/usr/bin/env python3
# ==============================================================================
# File: scripts/create_admin.py
# Purpose: Interactive setup script for platform administrator configuration.
#
#          The AI Business Agent platform does not store admin credentials in
#          the database. Instead, admin identity is defined entirely through
#          three environment variables in the .env file:
#
#            ADMIN_WHATSAPP_NUMBER  — WhatsApp number that receives system alerts
#            ADMIN_EMAIL            — Admin email address for reference
#            ADMIN_SECRET_TOKEN     — Internal API token for admin endpoints
#
#          This script:
#            1. Prompts for all three admin values interactively
#            2. Validates each value before accepting it
#            3. Generates a cryptographically secure token if not provided
#            4. Writes the values into the .env file (creates or updates)
#            5. Verifies database connectivity using the configured DATABASE_URL
#            6. Prints a summary of what was configured
#
#          Usage:
#            conda activate ai_business_agent_env
#            python scripts/create_admin.py
#
#          Safe to re-run — existing .env values for non-admin fields are
#          preserved. Only ADMIN_* keys are updated.
#
#          Requirements:
#            - .env file must exist (copy from .env.example if missing)
#            - DATABASE_URL must already be set in .env
# ==============================================================================

from __future__ import annotations

import asyncio
import getpass
import os
import re
import secrets
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — scripts/ is one level below the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE     = PROJECT_ROOT / ".env"
ENV_EXAMPLE  = PROJECT_ROOT / ".env.example"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ADMIN_TOKEN_BYTES      = 32          # 256-bit token → 64 hex chars
MIN_TOKEN_LENGTH       = 16
PHONE_E164_PATTERN     = re.compile(r"^\+\d{7,15}$")
EMAIL_PATTERN          = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ADMIN_ENV_KEYS         = ("ADMIN_WHATSAPP_NUMBER", "ADMIN_EMAIL", "ADMIN_SECRET_TOKEN")

# ANSI colours — disabled automatically on non-TTY environments
_IS_TTY = sys.stdout.isatty()


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m" if _IS_TTY else text


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m" if _IS_TTY else text


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if _IS_TTY else text


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _IS_TTY else text


# ==============================================================================
# .env File Helpers
# ==============================================================================

def _read_env_file(path: Path) -> dict[str, str]:
    """
    Parse a .env file into a key→value dict.

    Rules:
      - Lines starting with '#' are ignored (comments).
      - Blank lines are ignored.
      - Values may be optionally quoted with single or double quotes.
      - Inline comments after the value are NOT supported (KISS).

    Args:
        path: Path to the .env file.

    Returns:
        dict[str, str]: Parsed key-value pairs. Empty dict if file missing.
    """
    if not path.exists():
        return {}

    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key   = key.strip()
        value = value.strip()
        # Strip optional wrapping quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value

    return result


def _write_env_file(path: Path, env: dict[str, str]) -> None:
    """
    Write the env dict back to a .env file.

    All values are written unquoted. Existing comment blocks and ordering
    are not preserved — the file is reconstructed from the dict.

    Writes atomically by writing to a temp file then renaming.

    Args:
        path: Destination .env file path.
        env:  Key→value pairs to write.
    """
    lines = [
        "# ============================================================",
        "# .env — AI Business Agent platform configuration",
        "# Auto-updated by scripts/create_admin.py",
        "# ============================================================",
        "",
    ]
    for key, value in sorted(env.items()):
        lines.append(f"{key}={value}")

    content = "\n".join(lines) + "\n"

    # Atomic write — temp file + rename prevents partial writes
    tmp = path.with_suffix(".env.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _update_env_keys(env: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    """
    Return a new env dict with the given keys updated.

    Args:
        env:     Existing env key-value pairs.
        updates: New values to apply (overrides existing values for same keys).

    Returns:
        dict[str, str]: Updated env dict.
    """
    result = dict(env)
    result.update(updates)
    return result


# ==============================================================================
# Validation Helpers
# ==============================================================================

def _validate_whatsapp_number(number: str) -> str | None:
    """
    Validate a WhatsApp number in E.164 format.

    Args:
        number: Raw input string.

    Returns:
        None if valid, or an error message string if invalid.
    """
    cleaned = number.strip()
    if not cleaned:
        return "WhatsApp number cannot be empty."
    if not cleaned.startswith("+"):
        return "Number must start with '+' and country code (e.g., +919876543210)."
    if not PHONE_E164_PATTERN.match(cleaned):
        return "Number must be in E.164 format: +[country_code][number], digits only after '+'."
    return None


def _validate_email(email: str) -> str | None:
    """
    Validate an email address format.

    Args:
        email: Raw input string.

    Returns:
        None if valid, or an error message string if invalid.
    """
    cleaned = email.strip().lower()
    if not cleaned:
        return "Email cannot be empty."
    if not EMAIL_PATTERN.match(cleaned):
        return "Enter a valid email address (e.g., admin@yourdomain.com)."
    return None


def _validate_token(token: str) -> str | None:
    """
    Validate an admin secret token.

    Args:
        token: Raw input string.

    Returns:
        None if valid, or an error message string if invalid.
    """
    if not token.strip():
        return "Token cannot be empty."
    if len(token.strip()) < MIN_TOKEN_LENGTH:
        return f"Token must be at least {MIN_TOKEN_LENGTH} characters long."
    if " " in token:
        return "Token must not contain spaces."
    return None


# ==============================================================================
# Prompt Helpers
# ==============================================================================

def _prompt(
    label: str,
    validator,
    default: str | None = None,
    secret: bool = False,
) -> str:
    """
    Prompt the user for input with validation and retry.

    Args:
        label:     Display label for the prompt.
        validator: Callable(str) → str|None. Returns error message or None.
        default:   Pre-filled default value (shown in brackets).
        secret:    If True, input is hidden (for tokens/passwords).

    Returns:
        str: Validated, stripped input value.
    """
    default_hint = f" [{_yellow(default[:8] + '...' if default and len(default) > 8 else default or '')}]" if default else ""

    while True:
        try:
            if secret:
                raw = getpass.getpass(f"  {_bold(label)}{default_hint}: ")
            else:
                raw = input(f"  {_bold(label)}{default_hint}: ")
        except (EOFError, KeyboardInterrupt):
            print()
            _abort("Setup cancelled by user.")

        value = raw.strip()

        # Accept default if user presses Enter with nothing
        if not value and default is not None:
            value = default

        error = validator(value)
        if error is None:
            return value

        print(f"  {_red('✗')} {error}")


def _confirm(question: str, default: bool = False) -> bool:
    """
    Ask a yes/no confirmation question.

    Args:
        question: Question text.
        default:  Default answer if user presses Enter.

    Returns:
        bool: True for yes, False for no.
    """
    hint = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"  {question} {_yellow(hint)}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default

    if not raw:
        return default
    return raw in ("y", "yes")


def _abort(message: str) -> None:
    """Print an error and exit with code 1."""
    print(f"\n{_red('✗')} {message}\n")
    sys.exit(1)


# ==============================================================================
# Database Connectivity Check
# ==============================================================================

async def _check_database_connectivity(database_url: str) -> bool:
    """
    Verify that the database is reachable using the given URL.

    Attempts a simple connection and SELECT 1 query. Does not modify
    any data.

    Args:
        database_url: SQLAlchemy async database URL from .env.

    Returns:
        bool: True if connection succeeded, False otherwise.
    """
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text

        engine = create_async_engine(database_url, echo=False, pool_pre_ping=True)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        return True

    except Exception as exc:  # noqa: BLE001
        print(f"  {_red('Database error:')} {exc}")
        return False


# ==============================================================================
# Main Setup Flow
# ==============================================================================

def _print_banner() -> None:
    print()
    print(_bold("=" * 60))
    print(_bold("  AI Business Agent — Admin Setup"))
    print(_bold("=" * 60))
    print()
    print("  This script configures the platform administrator.")
    print("  Admin credentials are stored in the .env file.")
    print()


def _print_section(title: str) -> None:
    print()
    print(f"  {_bold('──')} {_bold(title)}")
    print()


def _print_success(message: str) -> None:
    print(f"  {_green('✓')} {message}")


def _print_warning(message: str) -> None:
    print(f"  {_yellow('!')} {message}")


async def main() -> None:
    _print_banner()

    # ------------------------------------------------------------------
    # Step 1: Locate and load .env file
    # ------------------------------------------------------------------
    _print_section("Step 1 — Load .env file")

    if not ENV_FILE.exists():
        if ENV_EXAMPLE.exists():
            _print_warning(f".env file not found at {ENV_FILE}")
            print(f"  Found .env.example — you should copy it first:")
            print(f"  {_yellow(f'cp {ENV_EXAMPLE} {ENV_FILE}')}")
            print()
            if not _confirm("Continue anyway and create a new .env file?", default=False):
                _abort("Setup cancelled. Copy .env.example to .env and try again.")
        else:
            _print_warning(f".env file not found at {ENV_FILE}")
            print()
            if not _confirm("Create a new .env file?", default=True):
                _abort("Setup cancelled.")

    env = _read_env_file(ENV_FILE)
    _print_success(f"Loaded .env from {ENV_FILE} ({len(env)} keys)")

    # ------------------------------------------------------------------
    # Step 2: Database connectivity check
    # ------------------------------------------------------------------
    _print_section("Step 2 — Database connectivity")

    database_url = env.get("DATABASE_URL", "").strip()
    if not database_url:
        _print_warning("DATABASE_URL not found in .env — skipping connectivity check.")
        print("  Set DATABASE_URL in .env before starting the application.")
    else:
        print(f"  Checking connection to database...")
        ok = await _check_database_connectivity(database_url)
        if ok:
            _print_success("Database connection successful.")
        else:
            _print_warning("Database connection failed.")
            print("  The admin values will still be saved to .env.")
            print("  Fix DATABASE_URL and ensure PostgreSQL is running before starting the app.")

    # ------------------------------------------------------------------
    # Step 3: Show existing admin values (if any)
    # ------------------------------------------------------------------
    _print_section("Step 3 — Admin configuration")

    existing: dict[str, str] = {
        k: env.get(k, "")
        for k in ADMIN_ENV_KEYS
    }

    any_existing = any(v for v in existing.values())
    if any_existing:
        print("  Existing admin configuration detected:")
        for key, val in existing.items():
            display = val[:8] + "..." if len(val) > 8 else val or _yellow("(not set)")
            print(f"    {key} = {display}")
        print()
        if not _confirm("Update admin configuration?", default=True):
            _abort("Setup cancelled — existing configuration kept.")

    # ------------------------------------------------------------------
    # Step 4: Collect admin values
    # ------------------------------------------------------------------
    _print_section("Step 4 — Enter admin details")

    # WhatsApp number
    print("  Admin WhatsApp number (receives system alerts).")
    print(f"  Format: E.164 e.g. {_yellow('+919876543210')}")
    whatsapp_number = _prompt(
        label     = "Admin WhatsApp number",
        validator = _validate_whatsapp_number,
        default   = existing.get("ADMIN_WHATSAPP_NUMBER") or None,
    )

    print()

    # Email
    print("  Admin email address.")
    admin_email = _prompt(
        label     = "Admin email",
        validator = _validate_email,
        default   = existing.get("ADMIN_EMAIL") or None,
    )

    print()

    # Secret token
    generated_token = secrets.token_hex(ADMIN_TOKEN_BYTES)
    current_token   = existing.get("ADMIN_SECRET_TOKEN", "")

    print("  Admin secret token (used to authenticate internal API calls).")
    print(f"  Press Enter to auto-generate a secure token, or type your own.")
    print(f"  {_yellow('Auto-generated token:')} {generated_token}")

    admin_token = _prompt(
        label     = "Admin secret token (Enter to use generated)",
        validator = _validate_token,
        default   = generated_token if not current_token else current_token,
        secret    = False,
    )

    # ------------------------------------------------------------------
    # Step 5: Confirm and write
    # ------------------------------------------------------------------
    _print_section("Step 5 — Confirm")

    print("  The following values will be written to .env:")
    print()
    print(f"    ADMIN_WHATSAPP_NUMBER = {_green(whatsapp_number)}")
    print(f"    ADMIN_EMAIL           = {_green(admin_email)}")
    print(f"    ADMIN_SECRET_TOKEN    = {_green(admin_token[:8])}...")
    print()

    if not _confirm("Write these values to .env?", default=True):
        _abort("Setup cancelled — .env was not modified.")

    updates = {
        "ADMIN_WHATSAPP_NUMBER": whatsapp_number,
        "ADMIN_EMAIL":           admin_email,
        "ADMIN_SECRET_TOKEN":    admin_token,
    }

    updated_env = _update_env_keys(env, updates)
    _write_env_file(ENV_FILE, updated_env)

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------
    _print_section("Setup Complete")

    _print_success("Admin configuration saved to .env")
    _print_success(f"WhatsApp alerts will be sent to: {whatsapp_number}")
    _print_success(f"Admin email: {admin_email}")
    _print_success("Secret token written (keep it safe — do not commit .env to git)")

    print()
    print("  Next steps:")
    print(f"  1. Ensure all other values in .env are filled (see .env.example)")
    print(f"  2. Run database migrations:  {_yellow('alembic upgrade head')}")
    print(f"  3. Start the application:    {_yellow('uvicorn app.main:app --host 0.0.0.0 --port 8000')}")
    print()


if __name__ == "__main__":
    # Ensure project root is on sys.path so app imports work
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        _abort("Setup interrupted.")