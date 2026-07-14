"""Regression tests for explicit and future Claude model resolution.

Runs standalone (``uv run python tests/test_model_resolver.py``) and under pytest.

WHY these tests exist: the resolver may translate documented pseudo aliases and
unambiguous Anthropic API IDs, but arbitrary Cursor-native or future
``claude-*`` IDs must reach Cursor unchanged. Replacing an unavailable model
with Haiku hides routing failures and changes requested capability silently.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code.normalizers import resolve_model_name  # noqa: E402


def test_resolves_explicit_pseudo_aliases() -> None:
    """Convenience aliases retain their established, intentional mappings."""
    assert resolve_model_name("opus") == "claude-4.5-opus-high"
    assert resolve_model_name("sonnet") == "claude-4.5-sonnet"
    assert resolve_model_name("haiku") == "claude-4.5-haiku"
    assert resolve_model_name("inherit") == "claude-4.5-haiku"


def test_resolves_known_dated_anthropic_api_ids() -> None:
    """Dated Anthropic IDs map only through known unambiguous families."""
    assert resolve_model_name("claude-sonnet-4-20250514") == "claude-4-sonnet"
    assert resolve_model_name("claude-opus-4-20250514") == "claude-4.5-opus-high"
    assert resolve_model_name("claude-3-5-haiku-20241022") == "claude-4.5-haiku"
    assert resolve_model_name("claude-sonnet-4-5-20250929") == "claude-4.5-sonnet"
    assert resolve_model_name("claude-opus-4-5-20251101") == "claude-4.5-opus-high"
    assert resolve_model_name("claude-haiku-4-5-20251001") == "claude-4.5-haiku"


def test_passes_through_live_cursor_native_claude_ids() -> None:
    """Cursor-specific capability variants must not depend on a static catalog."""
    assert resolve_model_name("claude-4.6-opus-low") == "claude-4.6-opus-low"
    assert (
        resolve_model_name("claude-opus-4-8-low-fast")
        == "claude-opus-4-8-low-fast"
    )


def test_passes_through_unknown_future_claude_ids() -> None:
    """Unknown Claude IDs surface upstream availability instead of becoming Haiku."""
    assert (
        resolve_model_name("claude-9.9-sonnet-experimental-thinking")
        == "claude-9.9-sonnet-experimental-thinking"
    )


def _run_all() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
