"""Tests for the `substitute` CLI subcommand."""

from __future__ import annotations

import contextlib

from pantrychef.cli import main


def test_substitute_missing_model_is_graceful(capsys, tmp_path) -> None:
    # No trained model present -> clear message, non-crashing exit code 1.
    rc = main(["substitute", "butter", "--model", str(tmp_path / "nope.kv")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not found" in out.lower()


def test_substitute_help_lists_diet(capsys) -> None:
    with contextlib.suppress(SystemExit):
        main(["substitute", "--help"])
    out = capsys.readouterr().out
    assert "--diet" in out
