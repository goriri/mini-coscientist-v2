"""The reasoning models and working languages a run may be configured with.

Both choices used to be constants. The model was one string in ``agents.py``
baked into every specialist at server start, and the language was whatever the
model happened to answer in, which was English. Making either one selectable
means every layer -- the CLI, the web form, the persisted session, the A2A card
path, the ADK agent tree -- has to agree on the same small set of values, so
the set is defined once, here, and nothing else is allowed its own list.

This module deliberately imports nothing from the rest of the package.
``models.py`` needs the defaults to type a :class:`Session` field and
``agents.py`` needs the whole catalogue to build agents, and those two already
depend on each other in one direction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_MODEL = "gemini-3.1-pro-preview"
"""The model a run uses when nobody chooses one."""


@dataclass(frozen=True)
class ModelChoice:
    """One model a run may be pointed at, and what the rest of the stack must
    do differently to talk to it."""

    id: str
    family: str
    label: str
    note: str
    max_output_tokens: int

    @property
    def search_grounded(self) -> bool:
        """Whether this model can carry ADK's built-in Google Search tool.

        ``GoogleSearchTool.process_llm_request`` raises outright for a model
        whose id is not a Gemini one -- search grounding is a Gemini server-side
        feature, not a function call that any model could make. Evidence
        discovery is the one role that needs it, so on a tree built for a model
        that answers False here, that role alone stays on :data:`DEFAULT_MODEL`.
        """
        return self.family == "gemini"


MODEL_CHOICES: tuple[ModelChoice, ...] = (
    ModelChoice(
        id=DEFAULT_MODEL,
        family="gemini",
        label="Gemini 3.1 Pro",
        note=(
            "The default. Deepest reasoning of the three, and the only "
            "configuration the sample dossiers in this repository were "
            "produced with."
        ),
        max_output_tokens=65536,
    ),
    ModelChoice(
        id="gemini-3.5-flash",
        family="gemini",
        label="Gemini 3.5 Flash",
        note=(
            "Faster and materially cheaper. Suitable for rehearsing a question "
            "or exercising the approval gates before committing to a full run."
        ),
        max_output_tokens=65536,
    ),
    ModelChoice(
        id="claude-opus-5",
        family="claude",
        label="Claude Opus 5",
        note=(
            "Served from Vertex AI Model Garden. Evidence discovery still runs "
            "on Gemini, because Google Search grounding is a Gemini-only tool; "
            "every other specialist is Claude."
        ),
        # Anthropic's SDK refuses a non-streaming request whose ceiling implies
        # more than ten minutes of generation, which puts the hard limit at
        # 21333 tokens. The largest single specialist response across both live
        # runs was the evolution protocol at roughly 7.4k, so this is about
        # three times the observed peak rather than a figure chosen to fit.
        max_output_tokens=21000,
    ),
)

MODEL_IDS: tuple[str, ...] = tuple(choice.id for choice in MODEL_CHOICES)


def model_choice(model_id: str) -> ModelChoice:
    """Look up a model, refusing anything not on the list.

    An unknown id is not a soft failure. The A2A card for a specialist is
    published per model, so an id nobody built a tree for is a 404 partway
    through a stage rather than an error at the point the run was configured.
    """
    for choice in MODEL_CHOICES:
        if choice.id == model_id:
            return choice
    raise ValueError(
        f"Unknown model {model_id!r}. Choose one of: {', '.join(MODEL_IDS)}."
    )


def model_slug(model_id: str) -> str:
    """A model id in a form that is both a URL segment and a Python identifier.

    The A2A route for a specialist and the ADK agent name behind it are the
    same string, so this has to satisfy the stricter of the two: ADK rejects an
    agent name that is not an identifier.
    """
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def specialist_agent_name(role: str, model_id: str) -> str:
    """Name the agent that serves ``role`` on ``model_id``.

    The default model keeps the bare role name. That is not cosmetic: the
    published card path, every integration test, and any client written against
    an earlier build all address ``/a2a/specialists/generation``, and adding a
    second model should not move the first one.
    """
    if model_id == DEFAULT_MODEL:
        return role
    return f"{role}__{model_slug(model_id)}"


DEFAULT_LANGUAGE = "en"
"""The language a run reports in when nobody chooses one."""


@dataclass(frozen=True)
class LanguageChoice:
    """One working language, and the prompt clause that asks for it."""

    code: str
    label: str
    endonym: str
    clause: str


# What the clause has to protect, and why each sentence of it is there. Every
# specialist returns JSON against a strict contract, so a model told simply to
# "answer in Chinese" translates the enum values and the field names with
# everything else, the payload fails validation, and the stage is discarded --
# the language choice would break the run rather than localise it. Identifiers
# are keys into other artifacts. A source's title is quoted from a document
# that exists; translating it names a document that does not. And retrieval
# stays in English because that is where the peer-reviewed literature is: a
# Chinese query returns a thinner slice of it, and the verifier then cannot
# resolve what it does return to a DOI.
_SIMPLIFIED_CHINESE_CLAUSE = (
    "Working language: Simplified Chinese (简体中文). Write every free-text "
    "field -- claims, rationales, predictions, assumptions, protocols, review "
    "findings, justifications, and limitations -- in Simplified Chinese.\n"
    "Four things stay in English whatever the working language. JSON field "
    "names and every enumerated value the schema lists, such as supports, "
    "contradicts, verified, and experimental, are part of the contract: "
    "reproduce them exactly as written, because a translated enum fails "
    "validation and the whole stage is thrown away. Identifiers -- candidate, "
    "claim, and source ids -- are keys into other artifacts, not prose. A "
    "source's own title, authors, publisher, DOI, and URL are quoted from a "
    "document that exists and are never translated, though a Chinese gloss "
    "after the original is welcome. Quantities, units, gene names, and "
    "chemical formulae keep their standard notation.\n"
    "Search in English. The literature this work rests on is published in "
    "English, and a Chinese query returns a thinner and less verifiable slice "
    "of it. Retrieve in English, then report what you found in Simplified "
    "Chinese."
)

LANGUAGE_CHOICES: tuple[LanguageChoice, ...] = (
    LanguageChoice(
        code=DEFAULT_LANGUAGE,
        label="English",
        endonym="English",
        # No clause at all rather than an English one. Every prompt in this
        # system is written in English already, so an instruction to answer in
        # English spends tokens restating the obvious and gives the model one
        # more thing to weigh against its actual task.
        clause="",
    ),
    LanguageChoice(
        code="zh-Hans",
        label="Simplified Chinese",
        endonym="简体中文",
        clause=_SIMPLIFIED_CHINESE_CLAUSE,
    ),
)

LANGUAGE_CODES: tuple[str, ...] = tuple(choice.code for choice in LANGUAGE_CHOICES)


def language_choice(code: str) -> LanguageChoice:
    for choice in LANGUAGE_CHOICES:
        if choice.code == code:
            return choice
    raise ValueError(
        f"Unknown language {code!r}. Choose one of: {', '.join(LANGUAGE_CODES)}."
    )


def language_clause(code: str) -> str:
    """The prompt clause for a language, or an empty string for English."""
    return language_choice(code).clause


def source_language_rule(code: str) -> str:
    """What to do with a source published in some other language.

    Separate from :func:`language_clause`, and spliced only into the two prompts
    that read documents, because it is the one thing an English run needs said
    about language and the rest of the prompts do not read documents. The
    English clause is empty on the argument that every prompt here is written in
    English already, which holds until a source is not: a German paper on
    coating lifetimes put "Die Lebensdauer der Elektrodenmaterialien wird durch
    die Beschichtung stark erhöht" into the evidence findings of an English
    report, one line of a corpus summary its reader could not read.
    """
    return (
        "A source is in whatever language it was published in, and the working "
        f"language is what you report in regardless. Summarise it in "
        f"{language_choice(code).label}, translate anything you quote, and never "
        "carry a sentence across in the language you found it in. A source's own "
        "title, authors, and publisher are the exception: those name a document "
        "that exists."
    )


def session_language_clause(session: object) -> str:
    """The working-language clause for a session, ready to concatenate.

    Returns an empty string for English and a clause ending in a blank line
    otherwise, so a caller can splice it in front of any block without having
    to know which case it got. The session is read by attribute rather than
    imported as a type because both prompt-assembling modules -- ``agents`` and
    ``debate`` -- need this, and one of them already imports the other.
    """
    clause = language_clause(getattr(session, "language", "") or DEFAULT_LANGUAGE)
    return f"{clause}\n\n" if clause else ""
