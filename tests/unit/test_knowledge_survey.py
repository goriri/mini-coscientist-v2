"""One survey across every pass, cited in the report's own numbers.

The Knowledge Base used to reproduce each search pass one after another. On a live
run that was seven literature reviews of the same question, each written as if it
were the only one, repeating each other's background and disagreeing without
noticing -- and every citation marker in all seven had to be struck before printing,
because each pass numbers its own source list and ``[cite: 4]`` in the third pass is
not the fourth reference of this report. The longest section of the dossier was
therefore also the only one with no references in it.

These pin the merge: what the passes cited is renumbered onto the run's own leads on
the way into the synthesis, and back out into the report's reference numbers on the
way to the page.
"""

from __future__ import annotations

import json

from coscientist.models import (
    Artifact,
    DeepResearchRun,
    DiscoveryManifest,
    DiscoveryNarrative,
    KnowledgeSurvey,
    KnowledgeSurveySection,
    Session,
    SourceLead,
)
from coscientist.narrative import _knowledge_summary, load_record
from coscientist.survey import (
    MIN_REPORT_CHARACTERS,
    SourceIndex,
    _trimmed,
    renumber_report,
    write_knowledge_survey,
)

RESISTANCE = "Acquired resistance to sotorasib in KRAS G12C mutant NSCLC"
BIOMARKERS = "Advances in biomarkers of resistance to KRAS mutation-targeted therapy"
REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIY"


def _lead(identifier: str, title: str, url: str) -> SourceLead:
    return SourceLead(id=identifier, canonical_url=url, title=title)


LEADS = [
    _lead("lead_a", RESISTANCE, "https://pubmed.ncbi.nlm.nih.gov/12345678/"),
    _lead("lead_b", BIOMARKERS, "https://doi.org/10.1000/biomarkers"),
]


def _annotation(title: str, url: str, start: int, end: int) -> dict:
    return {
        "type": "url_citation",
        "title": title,
        "url": url,
        "start_index": start,
        "end_index": end,
    }


def _payload(text: str, annotations: list[dict]) -> dict:
    return {
        "output_text": text,
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": text, "annotations": annotations}],
            }
        ],
    }


REPORT = "Resistance is polyclonal [cite: 1, 4]. Onset is early [cite: 2].\n"


def _report_payload() -> dict:
    first = REPORT.index("[cite: 1, 4]")
    second = REPORT.index("[cite: 2]")
    return _payload(
        REPORT,
        [
            _annotation(
                RESISTANCE, f"{REDIRECT}aaa", first, first + len("[cite: 1, 4]")
            ),
            _annotation(
                BIOMARKERS, f"{REDIRECT}bbb", first, first + len("[cite: 1, 4]")
            ),
            _annotation(
                BIOMARKERS, f"{REDIRECT}ccc", second, second + len("[cite: 2]")
            ),
        ],
    )


def test_a_passs_own_citation_numbers_are_replaced_by_the_runs():
    """``[cite: 1, 4]`` is numbered against a source list only that pass held.

    The two numbers inside it cannot be matched to the two annotations beside it --
    on a live pass a hundred and seventeen of a hundred and twenty-nine spans had a
    different count of each -- so the span is rewritten whole, from the documents its
    annotations name.
    """
    rewritten = renumber_report(_report_payload(), SourceIndex(LEADS).token)

    assert rewritten == "Resistance is polyclonal [S1, S2]. Onset is early [S2]."


def test_a_grounding_redirect_still_reaches_the_lead_it_produced():
    """Locator resolution rewrites a lead's URL to the publisher's and keeps no map.

    The annotation is left holding the redirect, so the URL join answers for nothing
    on a run where resolution worked. The title join answers instead: the lead's
    title is the annotation's title, because that is where discovery took it from.
    """
    index = SourceIndex(LEADS)

    assert index.token(_annotation(RESISTANCE, f"{REDIRECT}zzz", 0, 1)) == "S1"
    assert (
        index.token({"title": "", "url": "https://doi.org/10.1000/biomarkers"}) == "S2"
    )


def test_a_span_whose_sources_were_all_cut_is_removed_rather_than_left_standing():
    """The retention ceiling drops leads, and a marker for one names nothing."""
    rewritten = renumber_report(_report_payload(), SourceIndex(LEADS[:1]).token)

    assert rewritten == "Resistance is polyclonal [S1]. Onset is early."


def test_a_marker_the_payload_recorded_no_annotation_for_is_struck():
    payload = _payload("Onset is early [cite: 9].\n", [])

    assert renumber_report(payload, SourceIndex(LEADS).token) == "Onset is early."


def test_a_span_nested_inside_another_does_not_cut_the_report_twice():
    """Two overlapping rewrites would splice the text at crossing offsets."""
    text = "Resistance is polyclonal [cite: 1, 4].\n"
    start = text.index("[cite: 1, 4]")
    payload = _payload(
        text,
        [
            _annotation(RESISTANCE, f"{REDIRECT}aaa", start, start + 12),
            _annotation(BIOMARKERS, f"{REDIRECT}bbb", start + 1, start + 11),
        ],
    )

    assert renumber_report(payload, SourceIndex(LEADS).token) == (
        "Resistance is polyclonal [S1]."
    )


def test_offsets_that_point_past_the_marker_still_rewrite_the_marker():
    """A live pass's offsets sat four characters right of the marker they meant.

    Cutting at them left the pass's own bracket standing, spliced the run's number
    into the middle of it and swallowed the words behind it: the Knowledge Base read
    "outcomes [cit[cite: 1, 2, 3]ile heavily theorized", which is a broken marker and
    a lost "While" in one sentence.
    """
    text = "Resistance is polyclonal [cite: 1, 4]. Onset is early.\n"
    marker = text.index("[cite: 1, 4]")
    payload = _payload(
        text,
        [
            _annotation(
                RESISTANCE,
                f"{REDIRECT}aaa",
                marker + 4,
                marker + 4 + len("[cite: 1, 4]"),
            )
        ],
    )

    assert renumber_report(payload, SourceIndex(LEADS).token) == (
        "Resistance is polyclonal [S1]. Onset is early."
    )


def test_offsets_that_point_at_no_marker_leave_the_sentence_whole():
    """Nothing there to replace, so nothing is cut: the number joins the prose.

    The offsets were trusted as the place to cut, so a span pointing at ordinary words
    would have deleted them and put a reference number where the finding had been.
    """
    text = "Resistance is polyclonal and early in every cohort measured.\n"
    payload = _payload(
        text, [_annotation(RESISTANCE, f"{REDIRECT}aaa", 11, len("Resistance is poly"))]
    )

    assert renumber_report(payload, SourceIndex(LEADS).token) == (
        "Resistance is polyclonal[S1] and early in every cohort measured."
    )


def test_two_drifting_spans_do_not_land_on_one_marker():
    """Realignment can walk two annotations onto the same bracket, and cutting it
    twice would splice the report at offsets that have already been passed."""
    text = "Resistance is polyclonal [cite: 1, 4]. Onset is early.\n"
    marker = text.index("[cite: 1, 4]")
    payload = _payload(
        text,
        [
            _annotation(RESISTANCE, f"{REDIRECT}aaa", marker + 4, marker + 16),
            _annotation(BIOMARKERS, f"{REDIRECT}bbb", marker + 6, marker + 18),
        ],
    )

    assert renumber_report(payload, SourceIndex(LEADS).token) == (
        "Resistance is polyclonal [S1]. Onset is early."
    )


def test_the_source_list_the_survey_is_given_is_the_one_it_is_read_back_against():
    listing = SourceIndex(LEADS).listing().splitlines()

    assert listing[0].startswith("S1. ")
    assert RESISTANCE in listing[0]
    assert listing[1].startswith("S2. ")
    assert SourceIndex(LEADS).ids == ["lead_a", "lead_b"]


def test_a_report_past_the_budget_is_cut_on_a_paragraph_boundary():
    body = "\n\n".join("A paragraph about resistance." for _ in range(4000))
    ((_number, _heading, trimmed),) = _trimmed([(1, "PASS 1", body)], 1000)

    assert len(trimmed) < len(body)
    assert "cut off here" in trimmed
    assert trimmed.split("\n\n[This report")[0].endswith("resistance.")


def test_a_short_report_is_left_exactly_as_it_arrived():
    bodies = [(1, "PASS 1", "Short."), (2, "PASS 2", "Also short.")]

    assert _trimmed(bodies, 1000) == bodies


def test_no_pass_is_cut_below_the_floor_however_many_share_the_budget():
    body = "x" * 50_000
    bodies = [(number, f"PASS {number}", body) for number in range(1, 21)]

    for _number, _heading, trimmed in _trimmed(bodies, 1000):
        assert len(trimmed) >= MIN_REPORT_CHARACTERS


class _Store:
    """The artifact store, holding what the manifest does not: the whole report."""

    def __init__(self, payloads: dict[str, dict]):
        self.payloads = payloads

    def get(self, uri: str) -> dict:
        return self.payloads.get(uri, {})


class _Provider:
    def __init__(self, answer: str):
        self.answer = answer
        self.prompts: list[str] = []

    def complete(self, *, role: str, prompt: str) -> str:
        assert role == "evidence_synthesis"
        self.prompts.append(prompt)
        return self.answer


ANSWER = json.dumps(
    {
        "overview": "Resistance is polyclonal [S1, S2].",
        "sections": [{"heading": "Mechanisms", "prose": "Onset is early [S2]."}],
        "contested": ["Whether onset precedes treatment [S1]."],
        "not_found": ["No trial reported a rechallenge arm."],
        "sources": ["invented_by_the_model"],
    }
)


def _manifest(passes: int = 2) -> DiscoveryManifest:
    return DiscoveryManifest(
        question="What drives sotorasib resistance?",
        runs=[
            DeepResearchRun(
                pass_number=number,
                facet="contradictory",
                status="completed",
                raw_artifact_reference=f"gs://bucket/pass-{number}.json",
            )
            for number in range(1, passes + 1)
        ],
        narratives=[
            DiscoveryNarrative(
                question="What drives sotorasib resistance?",
                summary="A paragraph the normalizer kept.",
                pass_number=number,
            )
            for number in range(1, passes + 1)
        ],
        source_leads=list(LEADS),
    )


def _store(passes: int = 2) -> _Store:
    return _Store(
        {
            f"gs://bucket/pass-{number}.json": _report_payload()
            for number in range(1, passes + 1)
        }
    )


def test_the_survey_is_written_from_the_stored_reports_not_the_kept_paragraphs():
    """The manifest keeps the normalizer's paragraph -- a hundred and fifty words
    against the thirty thousand characters the provider wrote. Summarising from it
    would be summarising a summary."""
    provider = _Provider(ANSWER)

    write_knowledge_survey(_manifest(), provider, store=_store())

    ((prompt,),) = (provider.prompts,)
    assert "Resistance is polyclonal [S1, S2]." in prompt
    assert "A paragraph the normalizer kept." not in prompt
    assert "PASS 1" in prompt and "PASS 2" in prompt


def test_a_pass_whose_artifact_has_gone_falls_back_to_the_paragraph_it_kept():
    provider = _Provider(ANSWER)

    write_knowledge_survey(_manifest(), provider, store=_Store({}))

    assert "A paragraph the normalizer kept." in provider.prompts[0]


def test_the_survey_records_the_source_list_it_was_given_not_the_one_it_returned():
    """``source_ids`` is what resolves an [S7] back to a lead once the manifest has
    been revised and re-sorted under it, so it may not be the model's to write."""
    survey = write_knowledge_survey(_manifest(), _Provider(ANSWER), store=_store())

    assert survey is not None
    assert survey.source_ids == ["lead_a", "lead_b"]
    assert survey.question == "What drives sotorasib resistance?"
    assert survey.not_found == ["No trial reported a rechallenge arm."]


def test_a_single_pass_is_not_merged_with_itself():
    """One report already reads as one survey, and merging it costs a model call."""
    provider = _Provider(ANSWER)

    assert write_knowledge_survey(_manifest(1), provider, store=_store(1)) is None
    assert provider.prompts == []


def test_an_answer_that_fails_its_contract_leaves_the_passes_to_be_reproduced():
    assert (
        write_knowledge_survey(
            _manifest(),
            _Provider("Knowledge survey status: NOTHING TO MERGE."),
            store=_store(),
        )
        is None
    )


def test_a_survey_with_no_sections_is_not_a_survey():
    empty = json.dumps({"overview": "Resistance is polyclonal.", "sections": []})

    assert write_knowledge_survey(_manifest(), _Provider(empty), store=_store()) is None


def test_the_contract_holds_the_two_things_a_careless_merge_destroys():
    survey = KnowledgeSurvey(
        sections=[
            KnowledgeSurveySection(heading="Mechanisms", prose="Onset is early.")
        ],
        contested=["Whether onset precedes treatment."],
        not_found=["No trial reported a rechallenge arm."],
    )

    assert survey.contested and survey.not_found


def _rendered(manifest: DiscoveryManifest) -> str:
    """The Knowledge Base as a reader gets it: stored, loaded back, rendered.

    The trip is the test. Every step of it had its own passing tests while the live
    section carried no citations at all, because what broke the survey happened
    between them -- in the pass that loads a session for rendering, over a field that
    each side of the seam was right about on its own.
    """
    session = Session(question=manifest.question)
    session.artifacts.append(
        Artifact(
            stage="evidence",
            agent="deep_research_discovery",
            artifact_type="specialist_output",
            content="Two passes, merged into one survey.",
            schema_name="DiscoveryManifest",
            payload=manifest.model_dump(mode="json"),
        )
    )
    return _knowledge_summary(load_record(session))


def test_the_leads_the_survey_recorded_are_still_the_leads_when_it_is_rendered():
    """A live Knowledge Base of eighteen thousand characters carried not one citation
    while the survey behind it cited forty-eight sources. ``_scrub_prose`` names every
    id it finds in a stored contract after the thing that id points at, and it spares
    a field by its name -- ``id``, ``_id``, ``_ids``. This list of lead ids was called
    ``sources``, so all forty-eight were rewritten into phrases like "The unverified
    source Coating study 1" before the renderer looked one up, and every marker over
    them was struck as naming a lead the manifest no longer held."""
    survey = write_knowledge_survey(_manifest(), _Provider(ANSWER), store=_store())
    assert survey is not None
    manifest = _manifest().model_copy(update={"knowledge_survey": survey})

    section = _rendered(manifest)

    assert "Resistance is polyclonal [1, 2]." in section
    assert "Onset is early [2]." in section
    assert "[S1" not in section and "[S2" not in section


def test_a_survey_stored_under_the_earlier_field_name_still_cites_its_sources():
    """The report is computed on demand, so every session already on disk is rendered
    by today's code and every one of them wrote this list as ``sources``. What they
    stored is sound -- the ids were only ever rewritten on the way to the page -- so
    the rename reads them rather than refusing the manifest they are part of."""
    survey = write_knowledge_survey(_manifest(), _Provider(ANSWER), store=_store())
    assert survey is not None
    payload = _manifest().model_copy(update={"knowledge_survey": survey}).model_dump()
    payload["knowledge_survey"]["sources"] = payload["knowledge_survey"].pop(
        "source_ids"
    )

    section = _rendered(DiscoveryManifest.model_validate(payload))

    assert "Resistance is polyclonal [1, 2]." in section
