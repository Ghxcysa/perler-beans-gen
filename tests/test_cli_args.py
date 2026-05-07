"""Tests for command-line argument validation."""
from __future__ import annotations

import sys

import pytest

from perler_gen.cli import _parse_args, _parse_rgb


def test_parse_rgb_accepts_hex_and_csv_values():
    assert _parse_rgb("#0a64ff") == (10, 100, 255)
    assert _parse_rgb("10,100,255") == (10, 100, 255)


@pytest.mark.parametrize(
    "args",
    [
        ["--grid", "0", "48"],
        ["--max-colors", "0"],
        ["--rows-per-step", "0"],
        ["--denoise", "-1"],
        ["--post-smooth", "-1"],
        ["--speckle-size", "0"],
        ["--grid-interval", "0"],
    ],
)
def test_parse_args_rejects_invalid_numeric_values(monkeypatch, args):
    argv = ["perler-gen", "--input", "in.png", "--outdir", "out", *args]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        _parse_args()

