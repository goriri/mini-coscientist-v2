"""The two things a researcher may now configure: which model, and which language.

Both used to be constants, and a constant needs no tests. Making them choices puts
the same value in six places -- the flag, the request body, the persisted session,
the A2A card path, the ADK agent name, and the prompt -- and every one of those is a
place they can disagree. A disagreement is not a crash: a run configured for Claude
that quietly addresses the Gemini card produces a complete, plausible, wrongly
attributed report. So what these tests hold is agreement, not behaviour.
"""

from __future__ import annotations

import pytest

from app.agent import MODEL_TREES
from app.research_api import research_options
from coscientist.agents import (
    A2AProvider,
    DeterministicProvider,
    Specialist,
    bind_provider_model,
)
from coscientist.cli import main as cli_main
from coscientist.debate import JudgeContext, comparison_prompt, debate_prompt
from coscientist.model_catalog import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    LANGUAGE_CHOICES,
    LANGUAGE_CODES,
    MODEL_CHOICES,
    MODEL_IDS,
    language_choice,
    language_clause,
    model_choice,
    model_slug,
    session_language_clause,
    specialist_agent_name,
)
from coscientist.models import Candidate, CandidatePopulation, Session
from coscientist.orchestration import CoScientistWorkflow

_NON_DEFAULT = [choice.id for choice in MODEL_CHOICES if choice.id != DEFAULT_MODEL]


def _population() -> CandidatePopulation:
    return CandidatePopulation(
        candidates=[
            Candidate(
                id=f"candidate_{index:04d}",
                title=claim,
                claim=claim,
                rationale="Surface passivation limits electrolyte reduction.",
                mechanism_model="Surface passivation limits electrolyte reduction.",
                validation_protocol="Coin cells against an uncoated control.",
                predictions=["Capacity fade halves over 500 cycles."],
                falsifier="Fade is unchanged at matched C-rate.",
            )
            for index, claim in enumerate(
                ("An alumina coating helps", "A binder swap helps")
            )
        ],
        comparison_criteria=["Novelty", "Testability"],
    )


# ---------------------------------------------------------------------------
# The catalogue itself
# ---------------------------------------------------------------------------


def test_the_default_is_the_model_the_sample_dossiers_were_produced_with():
    assert DEFAULT_MODEL == "gemini-3.1-pro-preview"
    assert MODEL_IDS[0] == DEFAULT_MODEL
    assert DEFAULT_LANGUAGE == "en"
    assert LANGUAGE_CODES == ("en", "zh-Hans")


@pytest.mark.parametrize(
    ("lookup", "bad", "allowlist"),
    [
        (model_choice, "gemini-9-ultra", MODEL_IDS),
        (language_choice, "fr", LANGUAGE_CODES),
    ],
)
def test_an_unknown_choice_is_refused_and_the_message_names_the_alternatives(
    lookup, bad: str, allowlist: tuple[str, ...]
):
    """An id nobody built a tree for is a 404 partway through a stage otherwise.

    Failing at the point the run is configured is the whole reason the allowlist
    exists, and a refusal that does not say what *is* allowed sends the reader to
    the source.
    """
    with pytest.raises(ValueError) as raised:
        lookup(bad)

    assert bad in str(raised.value)
    for allowed in allowlist:
        assert allowed in str(raised.value)


def test_only_the_gemini_models_can_carry_search_grounding():
    """Search grounding is a Gemini server-side feature, not a callable tool.

    ``GoogleSearchTool.process_llm_request`` raises for anything else, so this
    property is what keeps evidence discovery off a tree that cannot serve it.
    """
    grounded = {choice.id for choice in MODEL_CHOICES if choice.search_grounded}

    assert grounded == {
        choice.id for choice in MODEL_CHOICES if choice.family == "gemini"
    }
    assert DEFAULT_MODEL in grounded


def test_the_claude_ceiling_stays_under_the_ten_minute_non_streaming_limit():
    """Anthropic's SDK refuses a non-streaming request implying over ten minutes
    of generation, which puts the hard ceiling at 21333 tokens. The largest
    single specialist response observed across both live runs was 7.4k."""
    claude = model_choice("claude-opus-5")

    ceiling = 600 * 128000 // 3600
    assert claude.max_output_tokens <= ceiling
    # And comfortably above the observed peak, so the limit is a safety margin
    # rather than a figure chosen to fit.
    assert claude.max_output_tokens >= 2 * 7400


# ---------------------------------------------------------------------------
# Routing: the agent name, the card path, the tree
# ---------------------------------------------------------------------------


def test_the_default_model_keeps_the_bare_role_name():
    """Adding a second model must not move the first one.

    Every integration test, the published card path, and any client written
    against an earlier build all address ``/a2a/specialists/generation``.
    """
    assert specialist_agent_name("generation", DEFAULT_MODEL) == "generation"


@pytest.mark.parametrize("model", _NON_DEFAULT)
def test_a_non_default_model_gets_a_suffixed_name_that_is_a_valid_identifier(
    model: str,
):
    """The A2A route and the ADK agent name behind it are the same string, and
    ADK rejects an agent name that is not a Python identifier."""
    name = specialist_agent_name("generation", model)

    assert name == f"generation__{model_slug(model)}"
    assert name.isidentifier()
    assert name != "generation"


def test_two_models_never_collide_on_one_agent_name():
    names = {specialist_agent_name("generation", model) for model in MODEL_IDS}

    assert len(names) == len(MODEL_IDS)


@pytest.mark.parametrize("model", MODEL_IDS)
def test_the_provider_dials_the_card_for_the_model_it_was_given(
    monkeypatch, model: str
):
    """The model is in the path rather than in the message, because an
    ``LlmAgent`` binds its model at construction. Dialling the wrong card runs
    the stage on a model the session did not choose and says nothing."""
    provider = A2AProvider("http://127.0.0.1:8000/", model=model)
    dialled: dict[str, str] = {}

    async def _fake_connect(base_url, *, client_config, relative_card_path):
        dialled["base"] = base_url
        dialled["path"] = relative_card_path
        raise RuntimeError("stop before the network call")

    from a2a.client import ClientFactory

    monkeypatch.setattr(ClientFactory, "connect", staticmethod(_fake_connect))
    with pytest.raises(RuntimeError, match="stop before"):
        provider.complete(role="generation", prompt="anything")

    expected = specialist_agent_name("generation", model)
    assert dialled["path"] == f"/a2a/specialists/{expected}/.well-known/agent-card.json"
    # The trailing slash on the base URL would otherwise double up in the path.
    assert dialled["base"] == "http://127.0.0.1:8000"


def test_there_is_one_published_tree_per_allowed_model():
    assert sorted(MODEL_TREES) == sorted(MODEL_IDS)
    for model, tree in MODEL_TREES.items():
        assert tree.name == specialist_agent_name("co_scientist_supervisor", model)
        assert len(tree.sub_agents) == len(MODEL_TREES[DEFAULT_MODEL].sub_agents)


@pytest.mark.parametrize(
    "model", [choice.id for choice in MODEL_CHOICES if not choice.search_grounded]
)
def test_a_tree_that_cannot_search_keeps_discovery_on_gemini(model: str):
    """The alternative -- dropping the tool -- turns evidence discovery into a
    model recalling papers from memory, which is where fabricated citations come
    from. So the one role that needs search stays on a model that has it."""
    discovery = next(
        agent
        for agent in MODEL_TREES[model].sub_agents
        if agent.name.startswith("evidence_discovery")
    )
    tool_names = [getattr(tool, "name", "") for tool in discovery.tools]

    assert "google_search" in tool_names
    assert discovery.model == DEFAULT_MODEL
    # Everything else on the tree is the model that was actually asked for.
    generation = next(
        agent
        for agent in MODEL_TREES[model].sub_agents
        if agent.name.startswith("generation")
    )
    assert generation.model != DEFAULT_MODEL


# ---------------------------------------------------------------------------
# The session is the authority
# ---------------------------------------------------------------------------


def test_a_session_saved_before_the_choice_existed_loads_on_the_defaults():
    """Sessions on disk predate both fields. Refusing to load one, or loading it
    with an empty model, would strand every run started before this change."""
    legacy = Session(question="Can a coating help?").to_dict()
    legacy.pop("model")
    legacy.pop("language")

    restored = Session.from_dict(legacy)

    assert restored.model == DEFAULT_MODEL
    assert restored.language == DEFAULT_LANGUAGE


@pytest.mark.parametrize("model", MODEL_IDS)
@pytest.mark.parametrize("language", LANGUAGE_CODES)
def test_the_choice_survives_a_save_and_a_load(model: str, language: str):
    flow = CoScientistWorkflow(
        "Can a coating help?",
        DeterministicProvider(),
        model=model,
        language=language,
    )

    restored = Session.from_dict(flow.session.to_dict())

    assert (restored.model, restored.language) == (model, language)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model", _NON_DEFAULT[0], "cannot be resumed on"),
        ("language", "zh-Hans", "cannot be resumed in"),
    ],
)
def test_a_resumed_run_refuses_to_change_the_configuration_it_started_on(
    field: str, value: str, message: str
):
    """Finishing on one model a run that was started on another produces a report
    whose provenance names two models for work that reads as one voice."""
    started = CoScientistWorkflow(
        "Can a coating help?", DeterministicProvider()
    ).session

    with pytest.raises(ValueError, match=message):
        CoScientistWorkflow(
            started.question, DeterministicProvider(), started, **{field: value}
        )
    # The value the session already holds is not a conflict, so a caller that
    # passes its own defaults through on every construction still works.
    CoScientistWorkflow(
        started.question,
        DeterministicProvider(),
        started,
        model=DEFAULT_MODEL,
        language=DEFAULT_LANGUAGE,
    )


@pytest.mark.parametrize("model", _NON_DEFAULT)
def test_loading_a_session_repoints_the_provider_at_the_model_it_names(
    tmp_path, model: str
):
    """Both the CLI and the web API have to build a provider before they can read
    the session that says which model it should address."""
    path = tmp_path / "session.json"
    CoScientistWorkflow(
        "Can a coating help?", DeterministicProvider(), model=model
    ).save(path)
    provider = A2AProvider("http://127.0.0.1:8000")
    assert provider.model_id == DEFAULT_MODEL

    CoScientistWorkflow.load(path, provider=provider)

    assert provider.model_id == model


def test_rebinding_leaves_alone_every_provider_that_is_not_an_a2a_one():
    """The offline provider *is* its own model, and a test double's ``model_id``
    is usually the thing being asserted on."""

    class Double:
        model_id = "a-double"

    offline, double = DeterministicProvider(), Double()

    bind_provider_model(offline, "claude-opus-5")
    bind_provider_model(double, "claude-opus-5")

    assert offline.model_id == "deterministic-offline"
    assert double.model_id == "a-double"


def test_two_providers_pointed_at_different_models_do_not_share_one():
    """``model_id`` was a class attribute when there was one supported model. The
    second provider constructed would have relabelled the first one's artifacts."""
    first = A2AProvider("http://127.0.0.1:8000", model=DEFAULT_MODEL)
    second = A2AProvider("http://127.0.0.1:8000", model=_NON_DEFAULT[0])

    assert first.model_id == DEFAULT_MODEL
    assert second.model_id == _NON_DEFAULT[0]


# ---------------------------------------------------------------------------
# The language clause, at every prompt surface
# ---------------------------------------------------------------------------


def test_english_asks_for_nothing_because_every_prompt_is_english_already():
    """An instruction to answer in English spends tokens restating the obvious and
    gives the model one more thing to weigh against its actual task."""
    assert language_clause("en") == ""
    assert session_language_clause(Session(question="Can a coating help?")) == ""


def test_the_chinese_clause_protects_the_parts_of_the_contract_it_must_not_touch():
    """A model told simply to answer in Chinese translates the enum values with
    everything else, the payload fails validation, and the stage is discarded --
    the language choice would break the run rather than localise it."""
    clause = language_clause("zh-Hans")

    assert "简体中文" in clause
    for protected in ("supports", "contradicts", "verified", "experimental"):
        assert protected in clause
    assert "Search in English" in clause
    assert "DOI" in clause


def test_a_session_missing_the_field_entirely_still_yields_a_clause():
    """``session_language_clause`` reads by attribute so that ``debate`` can use it
    without importing ``agents``, which already imports ``debate``."""

    class Bare:
        pass

    assert session_language_clause(Bare()) == ""


@pytest.mark.parametrize("language", LANGUAGE_CODES)
def test_the_specialist_prompt_carries_the_clause_directly_above_the_contract(
    language: str,
):
    """What the clause mostly does is carve out the parts of the contract it must
    not touch, so several hundred lines of prior work must not separate the
    exception from the rule it excepts."""
    session = Session(question="Can a coating help?", language=language)
    specialist = Specialist(
        stage="generate", role="generation", instruction="Propose hypotheses."
    )
    seen: dict[str, str] = {}

    class Capturing(DeterministicProvider):
        def complete(self, *, role: str, prompt: str) -> str:
            seen["prompt"] = prompt
            return super().complete(role=role, prompt=prompt)

    specialist.run(session, Capturing())
    prompt = seen["prompt"]
    clause = language_clause(language)

    if not clause:
        assert "简体中文" not in prompt
        return
    assert clause in prompt
    assert prompt.index(clause) > prompt.index("Prior work:")
    assert prompt.index(clause) < prompt.index("Return exactly one JSON object")


@pytest.mark.parametrize("build", [comparison_prompt, debate_prompt])
@pytest.mark.parametrize("language", LANGUAGE_CODES)
def test_the_tournament_judges_argue_in_the_language_the_run_was_configured_for(
    build, language: str
):
    """The tournament is the one specialist prompt not assembled by
    ``Specialist.run``. Without the clause the ranking stage argues in English
    about hypotheses written in Chinese, and the verdicts quoted in the dossier
    switch language halfway down the page."""
    session = Session(question="Can a coating help?", language=language)
    population = _population()
    context = JudgeContext.build(session, population)

    prompt = build(context, *population.candidates)

    assert prompt.startswith(session_language_clause(session))
    assert ("简体中文" in prompt) == (language != DEFAULT_LANGUAGE)


def test_a_hand_built_judge_context_still_constructs_without_a_language():
    """The field is defaulted so that the tournament's own tests, which build a
    context directly, did not all have to learn about the language."""
    context = JudgeContext(
        goal="A goal", preferences="Novelty", idea_attributes="", notes="", reviews={}
    )

    assert context.language_preamble == ""


# ---------------------------------------------------------------------------
# The two surfaces a researcher chooses from
# ---------------------------------------------------------------------------


def test_the_cli_refuses_a_model_flag_on_a_resumed_run(tmp_path):
    """Passing the flag's default through on a resume would reject every session
    that had been started on anything else, so the flag is rejected instead."""
    path = tmp_path / "session.json"
    CoScientistWorkflow("Can a coating help?", DeterministicProvider()).save(path)

    with pytest.raises(SystemExit, match="configure a new run"):
        cli_main(["run", "--resume", str(path), "--model", _NON_DEFAULT[0]])
    with pytest.raises(SystemExit, match="configure a new run"):
        cli_main(["run", "--resume", str(path), "--language", "zh-Hans"])


def test_the_cli_rejects_a_model_that_is_not_on_the_allowlist(capsys):
    with pytest.raises(SystemExit):
        cli_main(["run", "A question", "--model", "gemini-9-ultra"])

    assert DEFAULT_MODEL in capsys.readouterr().err


def test_the_form_reads_its_choices_from_the_server_that_validates_them():
    """A browser cache would otherwise keep offering a model the server has since
    retired, and the researcher would meet the allowlist as a 422."""
    options = research_options()

    assert [item["id"] for item in options["models"]] == list(MODEL_IDS)
    assert [item["code"] for item in options["languages"]] == list(LANGUAGE_CODES)
    assert [item["id"] for item in options["models"] if item["default"]] == [
        DEFAULT_MODEL
    ]
    assert [item["code"] for item in options["languages"] if item["default"]] == [
        DEFAULT_LANGUAGE
    ]
    # Every model carries the sentence that says what picking it costs or buys.
    assert all(item["note"] for item in options["models"])
    assert [item["endonym"] for item in options["languages"]] == [
        choice.endonym for choice in LANGUAGE_CHOICES
    ]
