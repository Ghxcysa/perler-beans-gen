"""Utility helpers."""
from __future__ import annotations

import string


_ALPHABET = string.ascii_uppercase


def index_to_symbol(i: int) -> str:
    """Map a zero-based index to a symbol (A-Z, AA, AB, ...)."""
    if i < 0:
        raise ValueError("Index must be non-negative.")
    base = len(_ALPHABET)
    # Excel-style column naming
    chars: list[str] = []
    n = i
    while True:
        n, rem = divmod(n, base)
        chars.append(_ALPHABET[rem])
        if n == 0:
            break
        n -= 1
    return "".join(reversed(chars))
