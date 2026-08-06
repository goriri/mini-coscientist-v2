"""LLM-as-judge ranking tournament with simulated scientific debate.

The ranking stage used to ask a model to *report* a tournament in one shot, so
every Elo number it produced described matches that were never played. This
module plays them: each pairing is an independent judgement by the model, and
the ratings are a consequence of those judgements rather than a claim about
them.

Both prompts are reproduced from the Co-Scientist supplementary material,
section 9.3 ("Prompts for the Ranking agent"): a cheap single-turn comparison
for the Swiss rounds and the full self-play scientific debate for the top-four
round robin. Only the variable substitution is adapted to this repository's
typed artifacts; the wording and structure are the paper's.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import Protocol

from .methods import method_requirements
from .model_catalog import session_language_clause
from .models import (
    Candidate,
    CandidatePopulation,
    CandidateReview,
    PairwiseComparison,
    ResearchPlan,
    ReviewSet,
    Session,
    TournamentState,
)
from .parity import (
    DEFAULT_ELO,
    ELO_K,
    SETTLED_MOVEMENT,
    population_from_artifacts,
    score_movement,
    stable_rounds,
)


class _Completer(Protocol):
    """The slice of ``agents.Provider`` a judge needs.

    Declared locally so ``debate`` never imports ``agents``; ``agents`` imports
    this module for the ranking stage.
    """

    def complete(self, *, role: str, prompt: str) -> str: ...


SWISS_ROUNDS = 3
TOP_ROUND_ROBIN_SIZE = 4
DEBATE_ROUND_NUMBER = SWISS_ROUNDS + 1
JUDGE_ROLE = "ranking"
# Section 9.3 asks for "typically 3 to 5" turns. A debate that comes back with
# fewer has not deliberated, whatever it concluded.
MIN_DEBATE_TURNS = 3

DEFAULT_IDEA_ATTRIBUTES = (
    "novelty, correctness/validity, feasibility, expected information gain, "
    "impact, and safety"
)
DEFAULT_PREFERENCES = (
    "claim-level evidence strength and contradiction status; falsifiability and "
    "discriminating information gain; mode-appropriate feasibility and "
    "reproducibility; expected impact, cost, time, and external validity; "
    "safety, ethics, privacy, and unresolved fatal flaws"
)

# Section 9.3, first prompt: "Prompt for hypothesis comparison during
# tournament". Reproduced verbatim. The paper's own text opens by asking for
# "better idea: <1 or 2>" and closes by asking for "better hypothesis: <1 or
# 2>"; both spellings are kept here and both are accepted by ``parse_verdict``.
COMPARISON_PROMPT = """You are an expert evaluator tasked with comparing two hypotheses.

Evaluate the two provided hypotheses (hypothesis 1 and hypothesis 2) and determine which one is superior based on the specified {idea_attributes}. Provide a concise rationale for your selection, concluding with the phrase "better idea: <1 or 2>".

Goal: {goal}

Evaluation criteria: {preferences}

Considerations: {notes}
Each hypothesis includes an independent review. These reviews may contain numerical scores. Disregard these scores in your comparative analysis, as they may not be directly comparable across reviews.

Hypothesis 1: {hypothesis 1}

Hypothesis 2: {hypothesis 2}

Review of hypothesis 1: {review 1}

Review of hypothesis 2: {review 2}

Reasoning and conclusion (end with "better hypothesis: <1 or 2>"):
"""

# Section 9.3, second prompt: "Prompt for hypothesis comparison via simulated
# scientific debate during tournament". Reproduced verbatim.
DEBATE_PROMPT = """You are an expert in comparative analysis, simulating a panel of domain experts engaged in a structured discussion to evaluate two competing hypotheses. The objective is to rigorously determine which hypothesis is superior based on a predefined set of attributes and criteria. The experts possess no pre-existing biases toward either hypothesis and are solely focused on identifying the optimal choice, given that only one can be implemented.

Goal: {goal}

Criteria for hypothesis superiority: {preferences}

Hypothesis 1: {hypothesis 1}

Hypothesis 2: {hypothesis 2}

Initial review of hypothesis 1: {review 1}

Initial review of hypothesis 2: {review 2}

Debate procedure:

The discussion will unfold in a series of turns, typically ranging from 3 to 5, with a maximum of 10.

Turn 1: begin with a concise summary of both hypotheses and their respective initial reviews.

Subsequent turns:
* Pose clarifying questions to address any ambiguities or uncertainties.
* Critically evaluate each hypothesis in relation to the stated Goal and Criteria. This evaluation should consider aspects such as:
    - Potential for correctness/validity.
    - Utility and practical applicability.
    - Sufficiency of detail and specificity.
    - Novelty and originality.
    - Desirability for implementation.
* Identify and articulate any weaknesses, limitations, or potential flaws in either hypothesis.

Additional notes: {notes}

Termination and judgment:

Once the discussion has reached a point of sufficient depth (typically 3-5 turns, up to 10 turns) and all relevant questions and concerns have been thoroughly addressed, provide a conclusive judgment. This judgment should succinctly state the rationale for the selection. Then, indicate the superior hypothesis by writing the phrase "better idea: ", followed by "1" (for hypothesis 1) or "2" (for hypothesis 2).
"""


def _render(template: str, **values: str) -> str:
    """Substitute the paper's placeholders without ``str.format``.

    Hypotheses and reviews routinely contain braces (JSON fragments, set
    notation), which ``str.format`` would try to interpret as fields of its own.
    """
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace("{" + name + "}", value)
    return rendered


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

# The paper terminates a match with the literal phrase; models wrap it in bold,
# angle brackets, or trailing punctuation, and the single-turn prompt asks for
# "better hypothesis" while its own header asks for "better idea".
_TERMINATOR_RE = re.compile(
    r"better\s+(?:idea|hypothesis)\s*:?\s*\**\s*<?\s*([12])\s*>?(?!\s*or\b)",
    re.IGNORECASE,
)
# A judge that echoes the instruction ("better idea: <1 or 2>") has not decided
# anything, so the negative lookahead above must reject it.
_TURN_RE = re.compile(
    r"^[ \t]*(?:[*#>\-]+[ \t]*)*(?:\*\*)?turn\s*\d+\b",
    re.IGNORECASE | re.MULTILINE,
)
_CONFIDENCE_RE = re.compile(
    r"confidence\s*(?:level|score)?\s*[:=]\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*(%?)",
    re.IGNORECASE,
)
_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n")
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)
# A judge that emits any of these quoted keys is filling in a contract, not
# debating. Two or more is conclusive; the live regression had all of them.
_CONTRACT_KEYS = (
    "ratings",
    "comparisons",
    "shortlist_ids",
    "elo_before",
    "elo_after",
    "swiss_rounds",
    "presented_first_id",
    "candidate_a_id",
    "candidate_b_id",
    "rubric_version",
    "debate_turns",
    "score_movement",
)
# Candidate-identifier-shaped tokens. The two prompts never show a candidate id,
# so a judge emitting one has invented it.
_ID_TOKEN_RE = re.compile(
    r"\b(?:cand|candidate|hyp|hypothesis|idea)_[A-Za-z0-9][A-Za-z0-9_]*\b"
)
# Best-effort capture of "Novelty: 4/5" style scoring lines the judge writes of
# its own accord. Deliberately anchored to a short label at the start of a line.
_CRITERION_SCORE_RE = re.compile(
    r"^[ \t]*(?:[-*•+]\s*)?(?:\*\*|__)?"
    # U+2019 is the curly apostrophe models emit ("Reviewer's risk").
    r"([A-Za-z][A-Za-z /'\u2019-]{2,38}?)"
    r"(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?"
    r"(\d{1,3}(?:\.\d+)?)[ \t]*(?:/[ \t]*(\d{1,3}(?:\.\d+)?)|(%))?",
    re.MULTILINE,
)
_SLOT_RE = re.compile(r"hypothesis[ _]*([12])\b", re.IGNORECASE)
# Labels that are bookkeeping rather than a judgement of the hypothesis.
_NON_CRITERION_LABELS = frozenset(
    {
        "confidence",
        "turn",
        "goal",
        "note",
        "notes",
        "score",
        "scores",
        "rating",
        "ratings",
        "elo",
        "better idea",
        "better hypothesis",
        "hypothesis",
        "verdict",
        "conclusion",
        "summary",
        "rationale",
        "judgment",
        "judgement",
        "criteria",
        "response",
    }
)
_MAX_CRITERION_SCORES = 12

NO_VERDICT_RATIONALE = (
    "The judge did not reach a verdict: the response never emitted the "
    'terminating "better idea: <1|2>" phrase, so the match is recorded as a '
    "draw and no Elo was exchanged. The winner was not guessed."
)


@dataclass(frozen=True)
class Verdict:
    """A parsed judgement of one match.

    ``winner`` is the *presented* position (1 or 2), not a candidate id, because
    the presentation order is rotated to control for position bias and only the
    caller knows which candidate occupied which slot.

    ``confidence`` is NOT a calibrated probability and must never be averaged as
    one. When ``confidence_is_stated`` is false it is a monotone function of how
    many turns the judge took -- a deliberation-depth proxy, nothing more. The
    section 9.3 prompts ask for no confidence at all, so the only honest
    alternatives were this or a constant.
    """

    winner: int | None
    rationale: str
    confidence: float
    turns: list[str] = field(default_factory=list)
    text: str = ""
    confidence_is_stated: bool = False
    criterion_scores: dict[str, float] = field(default_factory=dict)

    @property
    def decided(self) -> bool:
        return self.winner is not None

    @property
    def confidence_basis(self) -> str:
        return (
            "judge-stated"
            if self.confidence_is_stated
            else "deliberation depth, not a calibrated probability"
        )


class DegenerateJudge(RuntimeError):
    """The judge answered something that is not a judgement of this match.

    Kept separate from a mere draw: a draw is a judge that debated and declined
    to conclude, while this is a judge that never debated at all. It is
    translated into ``agents.ContractViolation`` at the tournament boundary.
    """

    def __init__(self, reason: str, text: str):
        super().__init__(reason)
        self.reason = reason
        self.text = text


def _strip_code_fences(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _looks_structured(text: str) -> str | None:
    """Why this response is a serialized contract rather than a debate.

    The live failure that motivated this check: the A2A ``ranking`` specialist
    carried a server-side instruction to emit a ``TournamentState``, which beat
    the section 9.3 user prompt. The model returned a whole tournament -- with
    invented candidate ids and invented Elo -- whose nested ``rationale`` field
    happened to end in "better idea: 2", so the verdict parsed and the match
    scored. A degenerate judge that looks healthy is worse than one that fails.
    """
    body = _strip_code_fences(text).strip()
    if body.startswith(("{", "[")):
        try:
            json.loads(body)
        except ValueError:
            pass
        else:
            return "the response is a JSON document, not a debate transcript"
    tells = sorted({key for key in _CONTRACT_KEYS if f'"{key}"' in body})
    if len(tells) >= 2:
        return (
            "the response embeds a serialized contract instead of prose "
            f"(quoted keys: {', '.join(tells)})"
        )
    return None


def _foreign_ids(text: str, allowed: set[str]) -> list[str]:
    """Identifier-shaped tokens the judge invented for this match."""
    permitted = {item.lower() for item in allowed}
    found = {
        token for token in _ID_TOKEN_RE.findall(text) if token.lower() not in permitted
    }
    if not found:
        return []
    keyed = [token for token in found if f'"{token}"' in text]
    # One stray identifier-ish word in prose is not evidence of a broken judge;
    # several of them, or one used as a JSON key, is.
    return sorted(found) if len(found) >= 2 or keyed else []


def _criterion_scores(text: str) -> dict[str, float]:
    """Recover per-criterion scores the judge volunteered.

    The judge is told to disregard the reviews' numbers, but it frequently
    scores the criteria itself. Those numbers were being discarded. Keys are
    prefixed with the presenting slot ("hypothesis 1"/"hypothesis 2") when the
    surrounding text attributes them, because a bare criterion name would not
    say which side it describes.
    """
    scores: dict[str, float] = {}
    for match in _CRITERION_SCORE_RE.finditer(text):
        label = " ".join(match.group(1).split()).strip(" -").lower()
        if not label or label in _NON_CRITERION_LABELS or len(label.split()) > 4:
            continue
        value = float(match.group(2))
        if match.group(3):
            denominator = float(match.group(3))
            if denominator <= 0:
                continue
            value /= denominator
        elif match.group(4):
            value /= 100.0
        elif value > 1.0:
            value /= 10.0 if value <= 10.0 else 100.0
        if not 0.0 <= value <= 1.0:
            continue
        # The *nearest* preceding mention, not the first one in the window: a
        # transcript that discusses hypothesis 1 and then hypothesis 2 would
        # otherwise file both sets of scores under hypothesis 1.
        nearby = list(
            _SLOT_RE.finditer(text, max(0, match.start() - 400), match.start())
        )
        key = f"hypothesis {nearby[-1].group(1)}: {label}" if nearby else label
        scores.setdefault(key, round(value, 4))
        if len(scores) >= _MAX_CRITERION_SCORES:
            break
    return scores


def _split_turns(text: str) -> list[str]:
    """Split a self-play transcript into its turns.

    Prefers the paper's explicit "Turn N" structure and falls back to
    paragraphs, so a single-turn comparison still yields a usable transcript.
    """
    marks = [match.start() for match in _TURN_RE.finditer(text)]
    if len(marks) >= 2:
        bounds = [*marks, len(text)]
        chunks = [text[bounds[i] : bounds[i + 1]].strip() for i in range(len(marks))]
        preamble = text[: marks[0]].strip()
        parts = ([preamble] if preamble else []) + chunks
    else:
        parts = _PARAGRAPH_RE.split(text)
    return [cleaned for block in parts if (cleaned := readable_turn(block))]


def readable_turn(block: str) -> str:
    """One turn as a reader should see it, without the protocol around it.

    The terminator is how the judge signals its choice to the parser; printed
    in a report it reads as "... is the vastly superior choice. better idea: 1."
    The turn label is normalised for the same reason: some judges bold it and
    some do not, so a transcript alternated between "Turn 1:" and "**Turn 1:**"
    down the same page.
    """
    stripped = _TERMINATOR_RE.sub("", block).strip()
    stripped = re.sub(r"[\s*_.:;,-]+$", "", stripped).strip()
    # Cutting the terminator takes the sentence's full stop with it, and every
    # turn in a rendered transcript then ended bare -- "... for their specific
    # claims" -- which reads as truncation rather than as the end of a turn.
    # Read behind whatever closes the sentence rather than treating the closer as
    # the terminator. A turn ending "... (see the validation protocol)" has a
    # bracket last, and the old test took that for punctuation and left the turn
    # without a full stop -- the very truncation the stop was added to prevent.
    tail = stripped.rstrip("\"')]}\u201d\u2019")
    if stripped and (not tail or tail[-1] not in ".!?"):
        stripped += "."
    return _TURN_LABEL_RE.sub(lambda m: f"Turn {m.group(1)}:", stripped)


_TURN_LABEL_RE = re.compile(
    r"^[ \t]*(?:[*#>\-]+[ \t]*)*\**turn\s*(\d+)\**\s*[:.]?\**", re.IGNORECASE
)
_NUMBERED_TURN_RE = re.compile(r"^Turn (\d+):\s*")
# The roles the debate prompt's panel gives itself. Restricted to a known list rather
# than "any capitalised word before a colon", which would split on "Goal:", on
# "Criteria:", and on every mid-sentence colon a judge writes.
_SPEAKER_LABEL_RE = re.compile(
    # The curly closing quotes are spelled as escapes: a judge ends a quoted phrase
    # with them, so the sentence boundary the speaker follows sits behind one.
    "(?:\\A|(?<=[.?!\"'\u201d\u2019])[ \t]|(?<=\n))[ \t]*\\**[ \t]*"
    r"((?:Expert|Panelist|Panellist|Reviewer|Scientist|Discussant|Moderator|Chair)"
    r"[ \t]*[A-Z0-9][A-Za-z0-9]?)"
    r"[ \t]*\**[ \t]*:[ \t]*\**[ \t]*"
)
# A judge that labels its conclusion inside the final turn, which is where the
# section 9.3 prompt asks for it. Split out rather than left inline so the closing
# argument is not buried at the end of the last speaker's paragraph.
_INLINE_RATIONALE_RE = re.compile(
    r"(?<=[.?!])[ \t]*\**[ \t]*rationale[ \t]*:[ \t]*\**[ \t]*", re.IGNORECASE
)


def _speaker_parts(body: str) -> list[tuple[str, str]]:
    """One turn split into who said what, in order."""
    marks = list(_SPEAKER_LABEL_RE.finditer(body))
    if not marks:
        return [("", body.strip())]
    parts: list[tuple[str, str]] = []
    lead = body[: marks[0].start()].strip()
    if lead:
        parts.append(("", lead))
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(body)
        speaker = _lettered(" ".join(mark.group(1).split()))
        parts.append((speaker, body[mark.end() : end].strip()))
    return parts


def _lettered(speaker: str) -> str:
    """One naming scheme for the panel across the whole report.

    Each match is judged by its own call, and the judge picks the scheme: ten matches
    of one live run came back as Expert A/B/C and two as Expert 1/2/3. Within a match
    the labels are consistent, so nothing is wrong on the page a reader is looking at
    -- but a reader who has followed Expert A through four transcripts meets Expert 1
    in the fifth and has to work out whether it is a different panel. It is not; the
    panel is anonymous either way, so the two schemes are one.
    """
    numbered = re.fullmatch(r"(\D+?)\s*(\d{1,2})", speaker)
    if not numbered or not 1 <= int(numbered.group(2)) <= 26:
        return speaker
    return f"{numbered.group(1)} {chr(ord('A') + int(numbered.group(2)) - 1)}"


def _opened(text: str) -> str:
    """Open a contribution on a capital, once its label has been taken off the front.

    "Closing rationale: Hypothesis 1 provides ..." is one sentence in the record, and
    the label carried its capital. Split into a label and a body, the body opens the
    bullet -- and the substitution that turns the slot name into "this idea" leaves an
    ordinary lower-case word standing where the sentence's capital used to be.

    Only a plainly lower-case word is raised. A contribution can open on a formula or
    an instrument name, and pH, mAh and cm-3 are not sentences missing a capital.
    """
    head, separator, tail = text.partition(" ")
    if head.islower() and not any(character.isdigit() for character in head):
        head = head[:1].upper() + head[1:]
    return head + separator + tail


def readable_exchange(turn: str) -> list[tuple[str, str]]:
    """One recorded turn as the separate contributions it actually contains.

    A "turn" as the judge writes it is routinely a whole exchange: three experts
    answering each other, plus the closing rationale, inside one numbered block. Set
    down as a single paragraph that is a wall of text with its attributions buried
    mid-sentence, and the markup is not even consistent -- one live transcript
    alternated between "**Expert 1:**" and "Expert A:" down the same page, because
    each came from a different judge call. Each contribution is returned with a label
    of its own so the renderer can set them apart and mark them the same way.
    """
    cleaned = readable_turn(turn)
    if not cleaned:
        return []
    numbered = _NUMBERED_TURN_RE.match(cleaned)
    stem = f"Turn {numbered.group(1)}" if numbered else ""
    body = cleaned[numbered.end() :] if numbered else cleaned
    exchange: list[tuple[str, str]] = []
    for speaker, text in _speaker_parts(body):
        pieces = _INLINE_RATIONALE_RE.split(text, maxsplit=1)
        labelled = (
            [(speaker, pieces[0]), ("Closing rationale", pieces[1])]
            if len(pieces) == 2 and pieces[0].strip() and pieces[1].strip()
            else [(speaker, text)]
        )
        for label, chunk in labelled:
            stated = _opened(chunk.strip())
            if not stated:
                continue
            if stated[-1] not in ".!?\"')":
                stated += "."
            prefix = ", ".join(item for item in (stem, label) if item)
            exchange.append((prefix, stated))
    return exchange


def _stated_confidence(text: str) -> float | None:
    matches = list(_CONFIDENCE_RE.finditer(text))
    if not matches:
        return None
    # The judge's own final statement, not one quoted from a review.
    match = matches[-1]
    value = float(match.group(1))
    if match.group(2) == "%" or value > 1.0:
        value /= 100.0
    return min(max(value, 0.0), 1.0)


def parse_verdict(text: str) -> Verdict:
    """Parse a judge response into a verdict, or into no verdict at all.

    Absence of the terminator is reported as ``winner=None`` rather than
    inferred from the prose: a tournament that silently invents a winner for an
    inconclusive debate produces ratings that look decided and are not.
    """
    body = text or ""
    turns = _split_turns(body)
    scores = _criterion_scores(body)
    matches = list(_TERMINATOR_RE.finditer(body))
    if not matches:
        return Verdict(
            winner=None,
            rationale=NO_VERDICT_RATIONALE,
            confidence=0.0,
            turns=turns,
            text=body,
            criterion_scores=scores,
        )
    final = matches[-1]
    winner = int(final.group(1))
    rationale = _rationale_before(body, final.start()) or (
        f"The judge selected hypothesis {winner}."
    )
    stated = _stated_confidence(body)
    # The paper's judge emits no calibrated probability. When it does not state
    # one, record deliberation depth instead of pretending to a calibration we
    # cannot support, and cap it well below certainty.
    confidence = stated if stated is not None else min(0.9, 0.5 + 0.05 * len(turns))
    return Verdict(
        winner=winner,
        rationale=rationale,
        confidence=round(confidence, 3),
        turns=turns,
        text=body,
        confidence_is_stated=stated is not None,
        criterion_scores=scores,
    )


def _rationale_before(body: str, terminator_start: int) -> str:
    """The judge's concluding prose, never a serialized payload.

    A rationale that renders as raw JSON in a report is not a rationale, so
    structured blocks are skipped in favour of the nearest prose above them.
    """
    prefix = body[:terminator_start]
    # Judges routinely bold the terminator, leaving a dangling "**" as the last
    # "paragraph"; markdown punctuation alone is not a rationale.
    blocks = [
        stripped
        for block in _PARAGRAPH_RE.split(prefix)
        if (stripped := block.strip(" \t\n*#->_"))
    ]
    prose = [
        block
        for block in blocks
        if not block.lstrip().startswith(("{", "[", '"'))
        and not _looks_structured(block)
    ]
    if not prose:
        return ""
    # The last prose paragraph, and whatever it is the second half of. A judge that
    # writes its conclusion over two paragraphs put the reasoning in the first and
    # "Therefore, hypothesis 2 is the stronger choice." in the second, and the
    # report printed the second alone -- a "therefore" with nothing before it.
    chosen = [prose[-1]]
    while len(chosen) < 3 and len(chosen) < len(prose) and _continues(chosen[0]):
        chosen.insert(0, prose[-1 - len(chosen)])
    # Judges label their conclusion, and bold the label: stripping the outer
    # asterisks off "**Rationale:** ..." left "Rationale:** ..." in the report,
    # and "**Conclusion:**" was not in the family of labels at all, so a live
    # report carried the asterisks around it. The report writes the label itself.
    joined = unemphasised(" ".join(chosen))
    return strip_rationale_label(joined) or joined


# A paragraph that reads as the second half of one. Only openers that cannot begin
# a standalone conclusion are listed: "This" and "The" routinely do.
_CONTINUATION_RE = re.compile(
    r"^(?:and|but|or|so|therefore|thus|hence|however|moreover|furthermore|"
    r"additionally|nevertheless|nonetheless|conversely|(?:in|by) contrast|"
    r"consequently|accordingly|in conclusion|for (?:these|those|this) reasons?|"
    r"on balance|overall|given (?:this|that|these))\b[,\s]",
    re.IGNORECASE,
)


def standalone_opening(text: str) -> str:
    """A recorded reason with no connective reaching back to prose nobody kept.

    Only the judge's closing paragraphs are kept, so a reason that opens on a
    contrast contrasts with a sentence the report cannot print: a live report
    began a match bullet "However, the deciding factor is safety and feasibility."
    What follows the connective is the whole of the recorded reason, so the
    connective goes and the reason stays. The openers are the same ones the
    parser reads as leaning on a paragraph above, so what cannot be recovered
    there is what is dropped here.
    """
    note, _, tail = text.strip().partition("]")
    # A rematch note is prepended in brackets, and the connective the reader trips
    # over is the one after it rather than the "[" the note opens on.
    if note.startswith("[") and tail.strip():
        return f"{note}] {standalone_opening(tail)}"
    opened = _CONTINUATION_RE.sub("", text.strip(), count=1).lstrip()
    return f"{opened[:1].upper()}{opened[1:]}" if opened else text.strip()


def _continues(block: str) -> str | re.Match[str] | None:
    """Whether a paragraph leans on one above it for its subject."""
    opening = block.lstrip()
    return (opening[:1].islower() and opening[:1].isalpha()) or _CONTINUATION_RE.match(
        opening
    )


# Emphasis runs a judge wrote around its own words. Matched only where they hug a
# non-space character on the inside, so the "*" in "2 * 3" survives untouched.
_EMPHASIS_RE = re.compile(r"(?<![\w*])\*{1,3}(?=\S)|(?<=\S)\*{1,3}(?![\w*])")


def unemphasised(text: str) -> str:
    """A judge's prose without the bold and italic markers it wrote into it.

    A rationale is reprinted inside a bullet whose own label is bold, so a stray
    "**" out of the judge's text closes the report's emphasis rather than the
    judge's and the rest of the line turns bold.
    """
    return _EMPHASIS_RE.sub("", text)


def strip_turn_label(text: str) -> str:
    """Drop a leading "Turn 4:" the judge carried into its own rationale field.

    The label belongs to the transcript, where it says whose turn it is. Copied
    into the ruling it became "The judge ruled this a loss with confidence 0.70.
    Rationale: Turn 4: Final evaluation and judgment. Hypothesis 2 is ...".
    """
    return _TURN_LABEL_RE.sub("", text.strip()).strip()


def strip_rationale_label(text: str) -> str:
    """Drop the judge's own "Rationale:" heading so the report writes one.

    Applied at render time as well as at parse time: sessions recorded before
    this existed carry the label, and doubling it reads as a rendering fault.
    """
    return _RATIONALE_LABEL_RE.sub("", text.strip()).strip()


# The label may trail a bracketed rematch note the report prepends. Judges use the
# whole family of them -- a live report printed "**Conclusion:** Hypothesis 2 ..."
# with the asterisks intact, because only "Rationale" was listed here.
# The colon is required of every label but "Rationale", which is the one the
# prompt itself asks for: "Decision theory favours the coated cell" opens on a
# label word and is a sentence, not a heading.
_RATIONALE_LABEL_RE = re.compile(
    r"(?:^|(?<=\]\s))\**\s*(?:rationale\s*:?|"
    r"(?:conclusion|verdict|judg(?:e)?ment|final (?:judg(?:e)?ment|assessment|"
    r"verdict|answer)|decision|summary)\s*:)\**\s*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Prompt context
# ---------------------------------------------------------------------------


def _title(candidate: Candidate, limit: int = 110) -> str:
    claim = " ".join(candidate.claim.split())
    return claim if len(claim) <= limit else f"{claim[: limit - 1].rstrip()}…"


def _bullets(label: str, items: list[str]) -> list[str]:
    return [f"{label}: {'; '.join(items)}"] if items else []


def render_hypothesis(candidate: Candidate) -> str:
    """The full falsifiable content of a candidate, as the judge sees it."""
    parts = [
        f"Claim: {candidate.claim}",
        f"Rationale: {candidate.rationale}",
        *_bullets("Testable predictions", candidate.predictions),
        *_bullets("Alternative explanations", candidate.alternatives),
        f"Falsifier: {candidate.falsifier}",
        f"Generation strategy: {candidate.generation_strategy}",
        *_bullets("Dependencies", candidate.dependencies),
        *_bullets("Risks", candidate.risks),
        *_bullets("Go/no-go tests", candidate.go_no_go_tests),
    ]
    return "\n".join(parts)


def render_review(reviews: list[CandidateReview]) -> str:
    """Flatten every recorded review of one candidate into the review slot."""
    if not reviews:
        return (
            "No independent review was recorded for this hypothesis. Treat its "
            "claims as unverified."
        )
    blocks: list[str] = []
    for review in reviews:
        lines = [
            f"[{review.criterion} review by {review.reviewer}] "
            f"recommendation: {review.recommendation} "
            f"(reviewer-reported confidence {review.confidence:.2f})",
            *_bullets("Findings", review.findings),
            *_bullets("Fatal flaws", review.fatal_flaws),
            *_bullets("Assumptions", review.assumptions),
            *_bullets("Objections", review.objections),
            *_bullets("Rebuttals", review.rebuttals),
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def reviews_by_candidate(session: Session) -> dict[str, list[CandidateReview]]:
    grouped: dict[str, list[CandidateReview]] = {}
    for artifact in session.artifacts:
        if artifact.schema_name != "ReviewSet" or not artifact.payload:
            continue
        for review in ReviewSet.model_validate(artifact.payload).reviews:
            grouped.setdefault(review.candidate_id, []).append(review)
    return grouped


@dataclass(frozen=True)
class JudgeContext:
    """Everything the two prompts substitute, resolved once per tournament."""

    goal: str
    preferences: str
    idea_attributes: str
    notes: str
    reviews: dict[str, str]
    # The tournament is the one place a specialist prompt is not assembled by
    # ``Specialist.run``, so the working language has to be carried here too.
    # Without it the ranking stage argues in English about hypotheses written
    # in Chinese, and the verdicts quoted in the dossier switch language
    # halfway down the page. Defaulted so a hand-built context in a test still
    # constructs.
    language_preamble: str = ""

    @classmethod
    def build(cls, session: Session, population: CandidatePopulation) -> JudgeContext:
        plan = _research_plan(session)
        goal_parts = [session.question]
        if plan is not None:
            goal_parts.append(f"Declared research mode: {plan.research_mode}.")
            goal_parts.append(f"Intended claim type: {plan.intended_claim}.")
            if plan.success_criteria:
                goal_parts.append(
                    "Success criteria: " + "; ".join(plan.success_criteria) + "."
                )
        else:
            goal_parts.append(f"Declared research mode: {session.research_mode}.")
        preferences = (
            "; ".join(population.comparison_criteria)
            if population.comparison_criteria
            else DEFAULT_PREFERENCES
        )
        notes = [
            "Judge only on the recorded content of each hypothesis and its "
            "reviews; do not credit evidence that is not stated there.",
            "Required scientific-method checks for the declared research mode: "
            + "; ".join(method_requirements(session.research_mode))
            + ".",
            "An unresolved fatal flaw outweighs speculative impact.",
        ]
        if plan is not None and plan.constraints:
            notes.append("Constraints: " + "; ".join(plan.constraints) + ".")
        if plan is not None and plan.governance_requirements:
            notes.append(
                "Governance requirements: "
                + "; ".join(plan.governance_requirements)
                + "."
            )
        rendered_reviews = reviews_by_candidate(session)
        return cls(
            goal=" ".join(goal_parts),
            preferences=preferences,
            idea_attributes=DEFAULT_IDEA_ATTRIBUTES,
            notes=" ".join(notes),
            reviews={
                candidate.id: render_review(rendered_reviews.get(candidate.id, []))
                for candidate in population.candidates
            },
            language_preamble=session_language_clause(session),
        )

    def review_of(self, candidate: Candidate) -> str:
        return self.reviews.get(candidate.id) or render_review([])


def _research_plan(session: Session) -> ResearchPlan | None:
    for artifact in reversed(session.artifacts):
        if artifact.schema_name == "ResearchPlan" and artifact.payload:
            try:
                return ResearchPlan.model_validate(artifact.payload)
            except Exception:
                # A malformed plan degrades the goal text; it must not stop
                # the ranking stage.
                return None
    return None


def comparison_prompt(
    context: JudgeContext, first: Candidate, second: Candidate
) -> str:
    """Section 9.3 single-turn comparison, used for the Swiss rounds."""
    return context.language_preamble + _render(
        COMPARISON_PROMPT,
        idea_attributes=context.idea_attributes,
        goal=context.goal,
        preferences=context.preferences,
        notes=context.notes,
        **{
            "hypothesis 1": render_hypothesis(first),
            "hypothesis 2": render_hypothesis(second),
            "review 1": context.review_of(first),
            "review 2": context.review_of(second),
        },
    )


def debate_prompt(context: JudgeContext, first: Candidate, second: Candidate) -> str:
    """Section 9.3 simulated scientific debate, used for the top-four pairs."""
    return context.language_preamble + _render(
        DEBATE_PROMPT,
        goal=context.goal,
        preferences=context.preferences,
        notes=context.notes,
        **{
            "hypothesis 1": render_hypothesis(first),
            "hypothesis 2": render_hypothesis(second),
            "review 1": context.review_of(first),
            "review 2": context.review_of(second),
        },
    )


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    round_number: int
    order: int
    a: Candidate
    b: Candidate
    debate: bool
    # Set when the finalists already met in a Swiss round. The pair is debated
    # anyway -- the expensive multi-turn judgement is the entire point of the
    # top-four round robin, and skipping it because a cheap single-turn match
    # happened earlier loses exactly the quality the round robin exists to buy.
    rematch_of_round: int | None = None

    @property
    def presented(self) -> tuple[Candidate, Candidate]:
        """Who occupies the "hypothesis 1" slot.

        Alternating by global match order is the position-bias control: an LLM
        judge favours whichever hypothesis it reads first, so half the matches
        must be presented in the reverse order.
        """
        return (self.a, self.b) if self.order % 2 == 0 else (self.b, self.a)


@dataclass(frozen=True)
class MatchRecord:
    match: Match
    verdict: Verdict
    comparison: PairwiseComparison


def _swiss_pairings(
    candidates: list[Candidate],
    ratings: dict[str, float],
    played: set[frozenset[str]],
) -> list[tuple[Candidate, Candidate]]:
    """Rating-ordered Swiss pairing that avoids replaying a pair.

    Mirrors ``parity.tournament_state`` so the deterministic and judged
    tournaments have the same shape, except that an odd candidate out takes a
    bye instead of raising.
    """
    remaining = sorted(candidates, key=lambda item: (-ratings[item.id], item.id))
    pairs: list[tuple[Candidate, Candidate]] = []
    while len(remaining) >= 2:
        first = remaining.pop(0)
        index = next(
            (
                position
                for position, item in enumerate(remaining)
                if frozenset((first.id, item.id)) not in played
            ),
            0,
        )
        pairs.append((first, remaining.pop(index)))
    return pairs


def _judge_match(match: Match, provider: _Completer, context: JudgeContext) -> Verdict:
    first, second = match.presented
    builder = debate_prompt if match.debate else comparison_prompt
    text = provider.complete(role=JUDGE_ROLE, prompt=builder(context, first, second))
    _reject_degenerate(match, text)
    verdict = parse_verdict(text)
    if match.debate and len(verdict.turns) < MIN_DEBATE_TURNS:
        raise DegenerateJudge(
            f"the simulated debate came back with {len(verdict.turns)} turn(s); "
            f"section 9.3 requires at least {MIN_DEBATE_TURNS}",
            text,
        )
    return verdict


def _reject_degenerate(match: Match, text: str) -> None:
    """Refuse to score a response that is not a judgement of this match."""
    reason = _looks_structured(text)
    if reason is None:
        invented = _foreign_ids(text, {match.a.id, match.b.id})
        if invented:
            reason = (
                "the judge referred to candidate identifiers that are not in "
                f"this match ({', '.join(invented[:4])}); the prompts never "
                "show a candidate id, so these were invented"
            )
    if reason is not None:
        raise DegenerateJudge(reason, text)


def _apply_verdict(
    match: Match, verdict: Verdict, ratings: dict[str, float]
) -> PairwiseComparison:
    first, second = match.presented
    left, right = match.a, match.b
    before = {left.id: ratings[left.id], right.id: ratings[right.id]}
    winner_id: str | None = None
    if verdict.winner == 1:
        winner_id = first.id
    elif verdict.winner == 2:
        winner_id = second.id
    if winner_id is not None:
        actual_left = 1.0 if winner_id == left.id else 0.0
        expected_left = 1 / (1 + 10 ** ((ratings[right.id] - ratings[left.id]) / 400))
        delta = ELO_K * (actual_left - expected_left)
        ratings[left.id] += delta
        ratings[right.id] -= delta
    rationale = verdict.rationale
    if match.rematch_of_round is not None:
        # Stated in the record itself so a reader of the comparison list sees
        # why this pair appears twice without reconstructing the pairing.
        rationale = (
            f"[Rematch: this pair also met in Swiss round "
            f"{match.rematch_of_round}. Both judgements are recorded; this "
            f"multi-turn debate is the later, decisive Elo update.] {rationale}"
        )
    return PairwiseComparison(
        round_number=match.round_number,
        candidate_a_id=left.id,
        candidate_b_id=right.id,
        presented_first_id=first.id,
        winner_id=winner_id,
        criterion_scores=verdict.criterion_scores,
        rationale=rationale,
        confidence=verdict.confidence,
        elo_before=before,
        elo_after={left.id: ratings[left.id], right.id: ratings[right.id]},
        # A Swiss comparison is one shot of prose; splitting its paragraphs
        # and calling them turns would dress a single opinion up as an
        # argument. Only a real debate carries a transcript.
        debate_turns=verdict.turns if match.debate else [],
        judge="llm_debate" if match.debate else "llm_comparison",
    )


def _standings(candidates: list[Candidate], ratings: dict[str, float]) -> list[str]:
    return [
        candidate.id
        for candidate in sorted(
            candidates, key=lambda item: (-ratings[item.id], item.id)
        )
    ]


def _affordable_swiss_rounds(field: int, comparisons: int) -> int:
    """How many Swiss rounds the comparison budget leaves room for.

    The budget is one number over the whole tournament, and the finals are the
    matches it should buy last: the top-four round robin is where the
    multi-turn debate happens, so it is held back from the arithmetic and the
    Swiss rounds take the reduction. One round always survives -- a tournament
    that seeded the finals off nothing but the default rating would be picking
    four candidates alphabetically.

    Under the default budget of eighteen and a field of eight this returns the
    full three, so a run inside its budget is unaffected.
    """
    per_round = field // 2
    if per_round < 1:
        return SWISS_ROUNDS
    finals = len(list(combinations(range(min(TOP_ROUND_ROBIN_SIZE, field)), 2)))
    return max(1, min(SWISS_ROUNDS, (comparisons - finals) // per_round))


def run_debate_tournament(
    session: Session,
    provider: _Completer,
    *,
    max_workers: int = 4,
) -> tuple[TournamentState, str]:
    """Play the ranking tournament with the model as judge.

    Three Swiss rounds are decided by the cheap single-turn comparison prompt
    and the top-four round robin by the full simulated debate, matching section
    9.3's split between routine and decisive matches.

    Matches inside a round are independent -- a Swiss round pairs each candidate
    exactly once, and no judgement depends on a rating -- so they are judged
    concurrently and their Elo updates are then applied in match order. That
    keeps the ratings identical to a sequential run while bounding wall-clock
    cost at ``max_workers`` in-flight model calls.

    Returns the tournament state and a Markdown transcript for a report
    appendix. A judge response with no terminator is a draw. A response that is
    not a judgement at all -- a serialized contract, invented candidate ids, or
    a "debate" of fewer than ``MIN_DEBATE_TURNS`` turns -- raises
    ``agents.ContractViolation``, because a degenerate judge that still awards
    Elo produces a tournament that looks healthy and means nothing.
    """
    population = population_from_artifacts(session.artifacts)
    candidates = list(population.candidates)
    context = JudgeContext.build(session, population)
    ratings = {candidate.id: DEFAULT_ELO for candidate in candidates}
    records: list[MatchRecord] = []
    met_in_round: dict[frozenset[str], int] = {}
    order = 0
    standings_history: list[list[str]] = []
    rating_history: list[dict[str, float]] = [dict(ratings)]
    swiss_rounds = _affordable_swiss_rounds(
        len(candidates), session.budget.max_pairwise_comparisons
    )

    def play(matches: list[Match]) -> None:
        if not matches:
            return
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            # ``map`` preserves input order, which is what makes the sequential
            # Elo application below deterministic.
            try:
                verdicts = list(
                    pool.map(
                        lambda item: _judge_match(item, provider, context), matches
                    )
                )
            except DegenerateJudge as broken:
                # Imported here because ``agents`` imports this module; a
                # top-level import would be circular.
                from .agents import ContractViolation

                raise ContractViolation(
                    JUDGE_ROLE,
                    f"the tournament judge did not judge the match: {broken.reason}",
                    broken.text,
                ) from broken
        for match, verdict in zip(matches, verdicts, strict=True):
            records.append(
                MatchRecord(match, verdict, _apply_verdict(match, verdict, ratings))
            )

    for round_number in range(1, swiss_rounds + 1):
        matches: list[Match] = []
        for left, right in _swiss_pairings(candidates, ratings, set(met_in_round)):
            matches.append(Match(round_number, order, left, right, debate=False))
            met_in_round.setdefault(frozenset((left.id, right.id)), round_number)
            order += 1
        play(matches)
        standings_history.append(_standings(candidates, ratings))
        rating_history.append(dict(ratings))

    top = sorted(candidates, key=lambda item: (-ratings[item.id], item.id))[
        :TOP_ROUND_ROBIN_SIZE
    ]
    # Every finalist pair debates, including one that already met in a Swiss
    # round: the round robin exists precisely so the finalists get the
    # multi-turn judgement, and a cheap earlier verdict is not a substitute.
    finals = [
        Match(
            DEBATE_ROUND_NUMBER,
            order + offset,
            left,
            right,
            debate=True,
            rematch_of_round=met_in_round.get(frozenset((left.id, right.id))),
        )
        for offset, (left, right) in enumerate(combinations(top, 2))
    ]
    order += len(finals)
    play(finals)
    if finals:
        standings_history.append(_standings(candidates, ratings))
        rating_history.append(dict(ratings))

    shortlist = _standings(candidates, ratings)[:TOP_ROUND_ROBIN_SIZE]
    stable = stable_rounds(standings_history)
    movement = score_movement(rating_history)
    state = TournamentState(
        ratings=ratings,
        comparisons=[record.comparison for record in records],
        shortlist_ids=shortlist,
        swiss_rounds=swiss_rounds,
        top_round_robin_size=TOP_ROUND_ROBIN_SIZE,
        ranking_stable_rounds=stable,
        score_movement=movement,
        # The workflow's own convergence rule: two stable rounds with under 5%
        # score movement.
        converged=stable >= 2 and movement < SETTLED_MOVEMENT,
    )
    return state, render_transcript(session, records, state, provider)


# ---------------------------------------------------------------------------
# Report transcript
# ---------------------------------------------------------------------------

_ROUND_LABELS = {
    DEBATE_ROUND_NUMBER: "Top-four round robin — simulated scientific debate",
}


def render_transcript(
    session: Session,
    records: list[MatchRecord],
    state: TournamentState,
    provider: _Completer,
) -> str:
    """A Markdown appendix: every match, its verdict, and its debate."""
    titles = {
        candidate.id: _title(candidate)
        for candidate in population_from_artifacts(session.artifacts).candidates
    }
    model_id = getattr(provider, "model_id", "unknown")
    decided = sum(1 for record in records if record.verdict.decided)
    lines = [
        "# Ranking tournament transcript",
        "",
        f"Judge: `{model_id}`, prompted as the Co-Scientist Ranking agent "
        "(supplementary section 9.3).",
        f"Structure: {state.swiss_rounds} Swiss rounds judged by the single-turn "
        "comparison prompt, then a top-"
        f"{TOP_ROUND_ROBIN_SIZE} round robin judged by the simulated "
        "scientific debate prompt."
        + (
            ""
            if state.swiss_rounds >= SWISS_ROUNDS
            # A shortened tournament separates the field less confidently, and a
            # reader comparing Elo ratings across runs needs to know it.
            else f" The design calls for {SWISS_ROUNDS} Swiss rounds; this run "
            f"played {state.swiss_rounds} to stay inside the session's "
            f"{session.budget.max_pairwise_comparisons}-comparison budget over a "
            f"field of {len(state.ratings)}."
        ),
        f"Matches played: {len(records)} ({decided} decided, "
        f"{len(records) - decided} drawn for want of a verdict); one model call "
        "per match.",
        "Presentation order alternates by match to control for position bias.",
        "Confidence is a deliberation-depth proxy unless marked judge-stated; "
        "it is not a calibrated probability and must not be averaged as one.",
        "",
    ]
    current_round: int | None = None
    for index, record in enumerate(records, 1):
        match = record.match
        if match.round_number != current_round:
            current_round = match.round_number
            label = _ROUND_LABELS.get(
                current_round, f"Swiss round {current_round} — single-turn comparison"
            )
            lines.extend(["", f"## Round {current_round}: {label}", ""])
        first, second = match.presented
        comparison = record.comparison
        winner = comparison.winner_id
        verdict_line = (
            f"**Verdict:** `{winner}` — {titles.get(winner, '')}"
            if winner
            else "**Verdict:** no verdict reached; recorded as a draw"
        )
        heading = f"### Match {index} — `{match.a.id}` vs `{match.b.id}`"
        if match.rematch_of_round is not None:
            heading += f" (rematch of Swiss round {match.rematch_of_round})"
        lines.extend(
            [
                heading,
                "",
                f"- Hypothesis 1 (presented first): `{first.id}` — {titles.get(first.id, '')}",
                f"- Hypothesis 2: `{second.id}` — {titles.get(second.id, '')}",
                f"- {verdict_line}",
                f"- Confidence: {comparison.confidence:.2f} "
                f"({record.verdict.confidence_basis})",
                "- Elo: "
                + " · ".join(
                    f"`{key}` {comparison.elo_before[key]:.1f} → "
                    f"{comparison.elo_after[key]:.1f}"
                    for key in (match.a.id, match.b.id)
                ),
            ]
        )
        if comparison.criterion_scores:
            lines.append(
                "- Judge-supplied criterion scores: "
                + ", ".join(
                    f"{name} {value:.2f}"
                    for name, value in sorted(comparison.criterion_scores.items())
                )
            )
        lines.extend(["", comparison.rationale, ""])
        if record.verdict.turns:
            lines.extend(
                [
                    "<details>",
                    f"<summary>Debate transcript ({len(record.verdict.turns)} turns)</summary>",
                    "",
                    *["\n".join((turn, "")) for turn in record.verdict.turns],
                    "</details>",
                    "",
                ]
            )
    lines.extend(
        [
            "",
            "## Final standings",
            "",
            "| Rank | Candidate | Elo |",
            "| ---: | --- | ---: |",
        ]
    )
    ordered = sorted(state.ratings.items(), key=lambda item: (-item[1], item[0]))
    for rank, (candidate_id, rating) in enumerate(ordered, 1):
        lines.append(f"| {rank} | `{candidate_id}` | {rating:.1f} |")
    lines.extend(
        [
            "",
            f"- Shortlist: {', '.join(f'`{item}`' for item in state.shortlist_ids)}",
            f"- Stable rounds: {state.ranking_stable_rounds}",
            f"- Score movement: {state.score_movement:.4f}",
            f"- Converged: {'yes' if state.converged else 'no'}",
        ]
    )
    return "\n".join(lines)
