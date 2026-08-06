"""Sanitizing text that PostgreSQL will not store."""

from __future__ import annotations

from coscientist.evidence import _content_text
from coscientist.normalization import (
    strip_unstorable_characters,
    strip_unstorable_values,
)


def test_a_nul_in_a_report_is_dropped_rather_than_carried_into_a_commit():
    """It cost a whole wave of Deep Research.

    Every specialist turn is persisted by ADK as a JSON event, and PostgreSQL
    rejects ``\\u0000`` inside JSON: the commit raised
    ``UntranslatableCharacterError``, the discovery call failed with it, and the
    stage reported nothing found after seven completed passes.
    """
    assert strip_unstorable_characters("before\x00after") == "beforeafter"


def test_a_lone_surrogate_goes_the_same_way():
    """It survives in a Python string and is unencodable the moment it is written."""
    assert strip_unstorable_characters("a\ud800b") == "ab"
    assert strip_unstorable_characters("a\ud800b").encode("utf-8") == b"ab"


def test_ordinary_prose_is_returned_unchanged():
    """Including the non-ASCII a research report is full of."""
    text = "Fe³⁺ chelation reduced β-amyloid by 40 % (p < 0.01) — 中文."
    assert strip_unstorable_characters(text) == text


def test_deep_research_text_is_sanitized_where_it_enters():
    """The one door every scrap of report prose comes through."""
    assert _content_text("report\x00body") == "reportbody"
    assert _content_text({"text": "part\x00one"}) == "partone"
    assert _content_text([{"text": "a\x00"}, {"text": "\x00b"}]) == "a\nb"


def test_a_structure_is_sanitized_all_the_way_down():
    """A tool response is a dict of strings, lists and nested dicts."""
    payload = {
        "tier": "verified",
        "text": "page\x00one",
        "registry_authors": ["A\x00uthor", "Second"],
        "nested": {"title": "T\ud800itle"},
        "http_status": 200,
        "retracted": False,
    }
    assert strip_unstorable_values(payload) == {
        "tier": "verified",
        "text": "pageone",
        "registry_authors": ["Author", "Second"],
        "nested": {"title": "Title"},
        "http_status": 200,
        "retracted": False,
    }


def test_a_fetched_document_is_sanitized_before_a_specialist_ever_sees_it():
    """A PDF is a container for arbitrary bytes, and one of them was a NUL."""
    import json

    assert json.dumps(strip_unstorable_values({"text": "a\x00b"})) == '{"text": "ab"}'
