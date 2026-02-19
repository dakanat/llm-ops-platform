"""Tests for the loading indicator component."""

from __future__ import annotations

from pathlib import Path

_LOADING_HTML = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "web"
    / "templates"
    / "components"
    / "loading.html"
)


class TestLoadingIndicatorPointerEvents:
    """The loading overlay must not block clicks when invisible."""

    def test_loading_indicator_has_pointer_events_none(self) -> None:
        """#loading-indicator should include pointer-events-none so clicks pass through."""
        content = _LOADING_HTML.read_text()
        assert "pointer-events-none" in content

    def test_loading_indicator_restores_pointer_events_on_htmx_request(self) -> None:
        """A <style> block should restore pointer-events: auto during .htmx-request."""
        content = _LOADING_HTML.read_text()
        assert "pointer-events: auto" in content
        assert ".htmx-request" in content
