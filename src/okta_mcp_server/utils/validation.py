# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2026-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Input validation utilities for Okta MCP Server.

This module provides validation functions to prevent path traversal and SSRF attacks
by ensuring user-supplied IDs do not contain malicious characters that could manipulate
URL paths when passed to the Okta SDK client.
"""

import functools
import inspect
import os
import re
from typing import Any, Callable

from loguru import logger


class InvalidOktaIdError(ValueError):
    """Exception raised when an Okta ID contains invalid characters."""

    pass


# Characters that are not allowed in Okta IDs to prevent path traversal
# This includes path separators, URL-reserved characters, and traversal sequences
FORBIDDEN_PATTERNS = [
    "/",  # Path separator
    "\\",  # Windows path separator
    "..",  # Path traversal
    "?",  # Query string delimiter
    "#",  # Fragment delimiter
    "%2f",  # URL-encoded forward slash
    "%2F",  # URL-encoded forward slash (uppercase)
    "%5c",  # URL-encoded backslash
    "%5C",  # URL-encoded backslash (uppercase)
    "%2e%2e",  # URL-encoded ..
    "%2E%2E",  # URL-encoded .. (uppercase)
]

# Regex pattern for valid Okta IDs
# Okta IDs are typically alphanumeric strings, sometimes with hyphens or underscores
# They may also be email addresses (for user lookups)
#
# IMPORTANT: The forbidden patterns check MUST run BEFORE the regex validation.
# The regex allows dots (for email addresses like user@example.com), but ".."
# is caught by the forbidden patterns list. This ordering ensures path traversal
# sequences like ".." are rejected even though single dots are allowed.
VALID_OKTA_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-@.+]+$")

# Maximum length of ID to log (to prevent log injection attacks)
MAX_LOG_ID_LENGTH = 100


def _sanitize_for_log(value: str) -> str:
    """Sanitize a value for safe logging by truncating and escaping."""
    if len(value) > MAX_LOG_ID_LENGTH:
        return f"{value[:MAX_LOG_ID_LENGTH]}... (truncated)"
    return value


def validate_okta_id(id_value: str, id_type: str = "ID") -> str:
    """
    Validate that an Okta ID does not contain path traversal or injection characters.

    This function prevents SSRF attacks where malicious IDs like '../groups/00g123'
    could be used to target unintended Okta APIs.

    Args:
        id_value: The ID value to validate (user_id, group_id, policy_id, rule_id, etc.)
        id_type: A descriptive name for the ID type (used in error messages)

    Returns:
        The validated ID value (unchanged if valid)

    Raises:
        InvalidOktaIdError: If the ID contains forbidden characters or patterns
    """
    if not id_value:
        raise InvalidOktaIdError(f"{id_type} cannot be empty")

    if not isinstance(id_value, str):
        raise InvalidOktaIdError(f"{id_type} must be a string")

    # IMPORTANT: Check forbidden patterns FIRST before regex validation.
    # The regex allows dots (for emails), but we must reject ".." sequences.
    id_lower = id_value.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.lower() in id_lower:
            logger.warning(
                f"Rejected {id_type} containing forbidden pattern '{pattern}': "
                f"{_sanitize_for_log(id_value)}"
            )
            raise InvalidOktaIdError(
                f"Invalid {id_type}: contains forbidden character or pattern '{pattern}'. "
                f"IDs must not contain path traversal sequences or URL-reserved characters."
            )

    # Validate against allowed character pattern
    if not VALID_OKTA_ID_PATTERN.match(id_value):
        logger.warning(f"Rejected {id_type} with invalid characters: {_sanitize_for_log(id_value)}")
        raise InvalidOktaIdError(
            f"Invalid {id_type}: contains invalid characters. "
            f"IDs must contain only alphanumeric characters, hyphens, underscores, "
            f"at signs, dots, and plus signs."
        )

    return id_value


class InvalidFilePathError(ValueError):
    """Exception raised when a file path is invalid or potentially unsafe."""

    pass


# Path traversal sequences to reject in file paths (URL-NEG-003)
FILE_PATH_TRAVERSAL_PATTERNS = [
    "..",
    "%2e%2e",  # URL-encoded ..
    "%2E%2E",  # URL-encoded .. (uppercase)
]

# Absolute OS directory prefixes the server must never read from (URL-NEG-004).
# Prevents MCP tools from exfiltrating host credentials or sensitive system
# files when a caller supplies a crafted path.
FORBIDDEN_SYSTEM_PATH_PREFIXES = (
    "/etc",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/root",
    "/var/log",
    "/var/run",
    "/run",
    "/usr/bin",
    "/usr/sbin",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/private/etc",   # macOS resolves /etc → /private/etc
    "/private/var",   # macOS resolves /var  → /private/var
)


def validate_file_path(file_path: str, param_name: str = "file_path") -> str:
    """
    Validate that a file path is safe to open.

    Rejects path traversal sequences and absolute paths that target OS system
    directories, preventing the server from reading sensitive host files.

    Args:
        file_path: The path to validate.
        param_name: Descriptive name used in error messages.

    Returns:
        The validated path (unchanged if valid).

    Raises:
        InvalidFilePathError: If the path contains traversal sequences         [URL-NEG-003]
            or targets a forbidden system location                              [URL-NEG-004].
    """
    if not file_path:
        raise InvalidFilePathError(f"{param_name} cannot be empty")

    if not isinstance(file_path, str):
        raise InvalidFilePathError(f"{param_name} must be a string")

    # URL-NEG-003: reject traversal sequences before any normalization so that
    # patterns like "images/../../etc/passwd" are caught at the lexical level.
    path_lower = file_path.lower()
    for pattern in FILE_PATH_TRAVERSAL_PATTERNS:
        if pattern in path_lower:
            logger.warning(
                f"Rejected {param_name} containing path traversal pattern '{pattern}': "
                f"{_sanitize_for_log(file_path)}"
            )
            raise InvalidFilePathError(
                f"Invalid {param_name}: path traversal sequences are not allowed."
            )

    # URL-NEG-004: reject absolute paths into OS system directories.
    # os.path.normpath collapses redundant separators (e.g. /etc//passwd →
    # /etc/passwd) without resolving symlinks, keeping the check lexical.
    normalized = os.path.normpath(file_path)
    for prefix in FORBIDDEN_SYSTEM_PATH_PREFIXES:
        if normalized == prefix or normalized.startswith(prefix + os.sep):
            logger.warning(
                f"Rejected {param_name} targeting system path '{prefix}': "
                f"{_sanitize_for_log(file_path)}"
            )
            raise InvalidFilePathError(
                f"Invalid {param_name}: access to system paths is not allowed. "
                f"Provide a path to a user-owned file (e.g. under /tmp or your home directory)."
            )

    return file_path


def validate_ids(*id_params: str, error_return_type: str = "list"):
    """
    Decorator that validates specified ID parameters before function execution.

    This decorator extracts the named parameters from the function call and validates
    each one using validate_okta_id(). If any validation fails, it returns an error
    response in the specified format without executing the wrapped function.

    Args:
        *id_params: Names of function parameters to validate (e.g., "user_id", "group_id")
        error_return_type: Format of error response - "list" or "dict"

    Usage:
        @validate_ids("user_id")
        async def get_user(user_id: str, ctx: Context = None) -> list:
            ...

        @validate_ids("group_id", "user_id")
        async def add_user_to_group(group_id: str, user_id: str, ctx: Context = None) -> list:
            ...

        @validate_ids("policy_id", error_return_type="dict")
        async def get_policy(ctx: Context, policy_id: str) -> dict:
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Get function signature to map positional args to parameter names
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each specified ID parameter
            for param_name in id_params:
                if param_name in bound_args.arguments:
                    id_value = bound_args.arguments[param_name]
                    if id_value is not None:  # Skip None values (optional params)
                        try:
                            validate_okta_id(id_value, param_name)
                        except InvalidOktaIdError as e:
                            logger.error(f"Invalid {param_name}: {e}")
                            if error_return_type == "dict":
                                return {"error": str(e)}
                            else:  # default to list
                                return [f"Error: {e}"]

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Get function signature to map positional args to parameter names
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each specified ID parameter
            for param_name in id_params:
                if param_name in bound_args.arguments:
                    id_value = bound_args.arguments[param_name]
                    if id_value is not None:
                        try:
                            validate_okta_id(id_value, param_name)
                        except InvalidOktaIdError as e:
                            logger.error(f"Invalid {param_name}: {e}")
                            if error_return_type == "dict":
                                return {"error": str(e)}
                            else:
                                return [f"Error: {e}"]

            return func(*args, **kwargs)

        # Return appropriate wrapper based on whether function is async
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
