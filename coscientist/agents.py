"""Specialist roles and optional Google ADK representation of the workflow."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Protocol

from .citations import citation_rule
from .contract_io import repair_prompt, schema_instruction
from .debate import run_debate_tournament
from .disciplines import classify_discipline
from .evidence import (
    ANGLE_SOURCE_TARGET,
    CORPUS_SOURCE_TARGET,
    resolve_grounding_urls,
)
from .methods import method_requirements
from .model_catalog import (
    DEFAULT_MODEL,
    MODEL_CHOICES,
    model_choice,
    session_language_clause,
    specialist_agent_name,
)
from .models import Artifact, ArtifactStatus, Session
from .parity import ROLE_CONTRACTS, TypedPayload, typed_specialist_payload
from .retrieval import fetch_source_document

CRITIC_ROUNDS = int(os.environ.get("COSCIENTIST_CRITIC_ROUNDS", "2"))
"""How many times a domain critic may send a specialist's draft back."""

GEMINI_MODEL = DEFAULT_MODEL
"""Default Vertex AI model ID for the reasoning-backed ADK workflow.

Kept as a name of its own because it is the one every caller outside this
package already imports. :data:`coscientist.model_catalog.DEFAULT_MODEL` is the
definition; this is the alias.
"""

VERTEX_LOCATION = "global"

STRUCTURED_OUTPUT_INSTRUCTIONS = {
    "goal_manager": (
        "Return one ResearchPlan JSON object with question, research_mode, "
        "intended_claim, assumptions, constraints, success_criteria, "
        "stopping_criteria, and governance_requirements."
    ),
    "evidence_discovery": (
        "Use Google Search and return one EvidencePacket JSON object. Every "
        "source and claim stays discovered_unverified. Each claim must be a "
        "distinct finding that could turn out to be wrong -- a measured "
        "quantity, a mechanism, a comparison, a null result -- never a "
        "restatement of the research question, and it must take a side: set "
        "relation to supports or contradicts unless the finding genuinely cuts "
        "both ways. Include the strongest contradicting or null finding the "
        "search returns; a packet that only agrees with the question has not "
        "searched. Give each source the document's own title and, as its url, "
        "the publisher's page or a resolvable identifier (doi.org, PubMed, "
        "arXiv). Where the prompt ends with a resolved Grounding URLs list, "
        "those are the documents your search actually opened -- cite from it "
        "and match each source to the entry it came from. Never copy a "
        "search-grounding redirect link: it expires, names no document, and is "
        "not a citation, and never cite a bare domain, which names no document "
        "either. When you have no locator better than a domain, say that in "
        "limitations rather than inventing a title or a locator. A title is "
        "the document's own title and nothing else: not a sentence from its "
        "abstract, not the finding you took from it, not a volume-and-page "
        "line. A reader scanning the reference list is matching those titles "
        "against papers they already know, and a sentence in that column "
        "matches nothing.\n"
        f"Aim for at least {ANGLE_SOURCE_TARGET} distinct sources and one claim "
        "per source; a pass of this stage is aiming for "
        f"{CORPUS_SOURCE_TARGET} across all its searches. That is a target, not "
        "a quota: if the literature you were asked for does not exist, return "
        "what does and say so in limitations. A padded packet is worse than a "
        "short one, because every entry in it is something a reviewer has to "
        "check before discovering it was never there."
    ),
    "source_verification": (
        "Return one EvidencePacket JSON object.\n"
        "Call fetch_source_document on every source before you give it a status. "
        "Do not assign one from the look of a URL: that is guessing, and a "
        "previous run marked fourteen sources inaccessible without a single "
        "fetch having happened. The tool follows redirects, so DOI links "
        "resolve; it reads PDFs; and it falls back to an open-access copy when "
        "the publisher refuses.\n"
        "The tool reports a tier. Never assign a status above it:\n"
        "- verified: the document was retrieved AND its text states what the "
        "claim attributes to it. Record the exact location -- section, figure, "
        "table or quoted phrase -- where you found it.\n"
        "- metadata_verified: a registry confirms the record but the text could "
        "not be read. Use this for paywalled papers that provably exist. It is "
        "an honest status, not a failure, and the evidence floor counts it.\n"
        "- retracted: a registry records a retraction. Say so in limitations. "
        "Nothing retracted may be recorded as supporting anything.\n"
        "- inaccessible: neither the document nor a registry record was "
        "obtained.\n"
        "If the tool retrieved the text and the text does not say what was "
        "attributed to it, the source is verified and the claim is not. Those "
        "are separate fields and that combination is a real and important "
        "finding.\n"
        "Carry every source and every claim you were given into your packet. "
        "This stage decides a status, never membership: omitting an entry "
        "deletes the record, and the report then shows a smaller literature "
        "than the run actually saw.\n"
        "A locator that names only a website -- a bare domain, a publisher's "
        "front page, a search redirect -- reaches no document, so nothing can "
        "have been checked against it. Mark those inaccessible whatever the "
        "title beside them says.\n"
        "Set verification_note on every source to one sentence saying why it "
        "holds its status, in terms a researcher can act on: which registry "
        "confirmed it, or what the fetch returned."
    ),
    "generation_evidence_first": (
        "Return one CandidatePopulation JSON object containing exactly two candidates "
        "using the evidence_first strategy. For each candidate, generate a reader-facing title, "
        "comprehensive paragraphs for mechanism_model and validation_protocol (>= 50 words each), "
        "categorized evidence_for, evidence_against, and evidence_gaps, and optionally valid Mermaid syntax in workflow_diagram_mermaid."
    ),
    "generation_mechanism_first": (
        "Return one CandidatePopulation JSON object containing exactly two candidates "
        "using the mechanism_first strategy. For each candidate, generate a reader-facing title, "
        "comprehensive paragraphs for mechanism_model and validation_protocol (>= 50 words each), "
        "categorized evidence_for, evidence_against, and evidence_gaps, and optionally valid Mermaid syntax in workflow_diagram_mermaid."
    ),
    "generation_analogy_transfer": (
        "Return one CandidatePopulation JSON object containing exactly two candidates "
        "using the analogy_transfer strategy. For each candidate, generate a reader-facing title, "
        "comprehensive paragraphs for mechanism_model and validation_protocol (>= 50 words each), "
        "categorized evidence_for, evidence_against, and evidence_gaps, and optionally valid Mermaid syntax in workflow_diagram_mermaid."
    ),
    "generation_competing_explanation": (
        "Return one CandidatePopulation JSON object containing exactly two candidates "
        "using the competing_explanation strategy. For each candidate, generate a reader-facing title, "
        "comprehensive paragraphs for mechanism_model and validation_protocol (>= 50 words each), "
        "categorized evidence_for, evidence_against, and evidence_gaps, and optionally valid Mermaid syntax in workflow_diagram_mermaid."
    ),
    "generation": (
        "Return one CandidatePopulation JSON object containing exactly eight candidates. "
        "Each candidate must define a reader-facing title, distinct claim, rationale, mechanism_model, "
        "validation_protocol, predictions, alternatives, falsifier, categorized evidence, and optionally valid Mermaid syntax in workflow_diagram_mermaid."
    ),
    "reflection": "Return one ReviewSet JSON object with one evidence_correctness review per candidate.",
    "novelty_review": "Return one ReviewSet JSON object with one novelty review per candidate.",
    "methods_statistics": "Return one ReviewSet JSON object with one methods_feasibility review per candidate.",
    "impact_review": "Return one ReviewSet JSON object with one impact_safety review per candidate.",
    "ethics_safety_governance": "Return one ReviewSet JSON object with one safety_governance review per candidate.",
    "ranking": (
        "Return one TournamentState JSON object after three randomized Swiss "
        "rounds and a top-four round robin. Include every comparison, Elo before/"
        "after, confidence, shortlist, score movement, and convergence state."
    ),
    "evolution": (
        "Return one EvolutionCycle JSON object. Evolve and independently re-review "
        "the four shortlisted candidates for at most three rounds; stop after two "
        "stable rounds with less than 5% score movement."
    ),
    "proximity": "Return one ResearchLandscape JSON object clustered by mechanism, outcome, evidence overlap, and data needs.",
    "meta_reviewer": (
        "Return one DossierManifest JSON object. Exclude every candidate with an "
        "unresolved fatal flaw from recommendation and state evidence that would change the decision."
    ),
}


RANKING_JUDGE_CONTRACT = (
    "You judge exactly ONE pair of hypotheses per request. You do NOT produce a "
    "tournament, a ranking, Elo ratings, a shortlist, or any JSON object; the "
    "Supervisor plays the tournament and assembles those from your verdicts.\n\n"
    "Follow the comparison or debate procedure in the request exactly. Write "
    "prose. Emit no JSON, no code fence, and no serialized contract of any "
    "kind. Never invent candidate identifiers: the two hypotheses are called "
    "'hypothesis 1' and 'hypothesis 2' and have no other names.\n\n"
    "For a simulated scientific debate, take the full 3 to 5 turns the request "
    "asks for, labelling each one 'Turn N:'. End every response with your "
    "rationale followed by the literal terminator: better idea: <1 or 2>."
)
"""Server-side contract for the live ranking specialist.

This role no longer emits a ``TournamentState``. ``coscientist.debate`` plays
the section 9.3 tournament match by match and builds the state client-side, so
the specialist's job is a single verdict.

Keeping the role name and leaving ``ROLE_CONTRACTS['ranking']`` as
``TournamentState`` is deliberate: that mapping types the *artifact* the stage
produces, which every downstream consumer reads and which the deterministic
offline path still computes arithmetically. Only the prompt-side instruction
changes. A new ``debate_judge`` A2A card was the alternative, but it would have
left the published ``ranking`` card unreachable in live runs and required a
matching server redeploy before the ranking stage could work at all.
"""


def output_contract(role: str) -> str:
    """The role's prose brief plus the exact schema its answer is validated against.

    The prose alone named fields the contracts do not have, so a specialist
    could follow the instruction to the letter and still be rejected. Rendering
    the schema from the Pydantic model makes that drift impossible.
    """
    brief = STRUCTURED_OUTPUT_INSTRUCTIONS.get(role, "")
    model = ROLE_CONTRACTS.get(role)
    if model is None:
        return brief
    return (
        f"{brief}\n\n{schema_instruction(model)}"
        if brief
        else schema_instruction(model)
    )


def configure_vertex_ai_global_endpoint() -> None:
    """Configure ADK's Gemini client to use Vertex AI's global endpoint.

    Authentication and project selection remain the deployer's responsibility:
    set ``GOOGLE_CLOUD_PROJECT`` and Application Default Credentials before
    starting ADK.  The global endpoint is intentional: it optimizes
    availability but does not provide an in-region data-processing guarantee.
    """
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
    os.environ["GOOGLE_CLOUD_LOCATION"] = VERTEX_LOCATION
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        try:
            import google.auth
            from google.auth.exceptions import DefaultCredentialsError
        except ImportError:
            # Importing the offline package must not require cloud credentials.
            return
        try:
            _, project_id = google.auth.default()
        except DefaultCredentialsError:
            return
        if project_id:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id


class Provider(Protocol):
    model_id: str

    def complete(self, *, role: str, prompt: str) -> str: ...


class ContractViolation(RuntimeError):
    """A live specialist's output could not be validated against its contract.

    Raised instead of substituting a template, so a run never presents generated
    boilerplate as though a model had reasoned its way to it.
    """

    def __init__(self, role: str, error: str, content: str):
        super().__init__(
            f"The {role} specialist returned output that does not satisfy its "
            f"contract after a repair attempt: {error}"
        )
        self.role = role
        self.error = error
        self.content = content


class DeterministicProvider:
    """Offline provider used for a transparent demo and tests.

    It purposefully uses conservative templates rather than pretending to have
    searched literature or run experiments.
    """

    model_id = "deterministic-offline"
    # This provider *is* the deterministic path, so its payloads are computed by
    # design rather than substituted for a failed model answer. Only a live
    # provider's unparseable output is a contract violation.
    deterministic = True

    def complete(self, *, role: str, prompt: str) -> str:
        if role.endswith("_critic"):
            return "SATISFIED"
        question = prompt.split("Research question:", 1)[-1].split("\n", 1)[0].strip()
        templates = {
            "goal_manager": f"Research objective: {question}\n\nConstraints to confirm with the researcher:\n- target system, population/material, and available measurements\n- success metric, baseline, resources, timeline, safety/ethics\n\nDeliverable: a falsifiable hypothesis and an auditable experiment plan.",
            "evidence_discovery": "Evidence discovery status: OFFLINE / UNVERIFIED.\n\nSearch questions to run with the live Google Search specialist:\n- What primary studies directly test the proposed mechanism?\n- What negative findings, replications, corrections, or retractions exist?\n- Which official datasets, standards, or registered protocols apply?\n\nNo source has been discovered or verified by the deterministic provider.",
            "source_verification": "Source verification status: NO SOURCES PROVIDED.\n\nA live verifier must open the original source, resolve its DOI/PMID or dataset identifier, inspect the exact supporting passage, and check correction/retraction status. Search snippets cannot satisfy an evidence gate.",
            "generation": "Eight candidate records were created using evidence-first, mechanism-first, analogy/transfer, and competing-explanation strategies. All are proposals pending verified evidence; inspect the typed CandidatePopulation artifact for their predictions, alternatives, dependencies, risks, falsifiers, go/no-go tests, and cross-candidate comparison criteria.",
            "generation_evidence_first": "Eight candidate records were created using evidence-first strategy. All are proposals pending verified evidence; inspect the typed CandidatePopulation artifact for their predictions, alternatives, dependencies, risks, falsifiers, go/no-go tests, and cross-candidate comparison criteria.",
            "generation_mechanism_first": "Eight candidate records were created using mechanism-first strategy. All are proposals pending verified evidence; inspect the typed CandidatePopulation artifact for their predictions, alternatives, dependencies, risks, falsifiers, go/no-go tests, and cross-candidate comparison criteria.",
            "generation_analogy_transfer": "Eight candidate records were created using analogy/transfer strategy. All are proposals pending verified evidence; inspect the typed CandidatePopulation artifact for their predictions, alternatives, dependencies, risks, falsifiers, go/no-go tests, and cross-candidate comparison criteria.",
            "generation_competing_explanation": "Eight candidate records were created using competing-explanation strategy. All are proposals pending verified evidence; inspect the typed CandidatePopulation artifact for their predictions, alternatives, dependencies, risks, falsifiers, go/no-go tests, and cross-candidate comparison criteria.",
            "reflection": "Critical review:\n- Novelty is unverified; search primary literature and negative results before claiming it.\n- Causal claims need controls, randomization/blinding where applicable, and independent replication.\n- Specify effect size, power calculation, preregistered exclusions, and adverse-event monitoring.\n\nRecommendation: advance only candidates that remain feasible after these checks.",
            "novelty_review": "Novelty review:\n- Compare each candidate with verified prior art rather than search-result similarity.\n- Treat missing primary sources and inaccessible dates as insufficient evidence.\n- Preserve potentially valuable minority hypotheses while novelty remains unresolved.",
            "methods_statistics": "Methods and statistics review:\n- Declare the research mode and intended claim before choosing a design.\n- Define constructs, estimand or proof obligation, sampling/precision rationale, controls, missing-data handling, uncertainty, robustness checks, and stopping rules.\n- Separate exploratory analysis from confirmatory tests and require an independent replication or validation path.\n\nStatus: CONDITIONAL; domain-specific calculations require validated inputs.",
            "impact_review": "Impact review:\n- Score expected information gain, scientific importance, feasibility, cost, time, and translational relevance separately.\n- Do not let speculative impact override correctness, safety, or missing evidence.\n- State who benefits, plausible failure modes, and external-validity limits.",
            "ethics_safety_governance": "Governance review:\n- Screen human/animal/environmental/biosafety, privacy, data rights, dual-use, and operational risks.\n- Identify institutional approvals and qualified expert review required before action.\n- Do not access restricted data or execute a real-world protocol.\n\nStatus: HUMAN REVIEW REQUIRED; auto approval cannot waive this requirement.",
            "ranking": "Prioritization (provisional):\n1. Candidate 1 — high tractability; mechanism and metric are directly observable.\n2. Candidate 2 — potentially informative but boundary condition needs operational definition.\n3. Candidate 3 — valuable if interaction is plausible; highest experimental complexity.\n\nScores are decision aids, not evidence: novelty 2/5, feasibility 3/5, expected impact 3/5, risk 3/5.",
            "evolution": "Refined lead hypothesis:\nIf the intervention is applied at a prespecified dose/window, the primary metric will improve versus matched control because the proposed mediator changes first.\n\nExperiment: randomized matched-control pilot; measure baseline, mediator, outcome, and safety endpoints; analyze an effect estimate with uncertainty; then replicate independently.\n\nFalsifier: no prespecified improvement, no mediator shift, or unacceptable safety signal.",
            "proximity": "Related-work map to validate:\n- adjacent mechanism studies\n- intervention and dose/window studies\n- benchmark/control protocols\n- contradictory or null-result reports\n\nDo not infer novelty from this map. Attach DOI/PMID and exact supporting passages after a human literature review.",
            "meta_reviewer": "Meta-review:\nThe lead is testable but not yet evidence-backed. Before execution, verify every factual claim and reference, obtain domain/safety/ethics review, define statistical analysis and data governance, and confirm resources.\n\nDecision: CONDITIONAL ADVANCE — proceed only to protocol drafting, not scientific conclusion.",
        }
        return templates[role]


class A2AProvider:
    """Invoke independently published specialist services through the A2A SDK."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 120,
        model: str = DEFAULT_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # An instance attribute, where this used to be a class attribute set to
        # the one supported model. Two providers pointed at different models
        # would otherwise have shared it, and the second one constructed would
        # have relabelled the first one's artifacts.
        self.model_id = model

    async def _complete(self, *, role: str, prompt: str) -> str:
        import httpx
        from a2a.client import ClientConfig, ClientFactory
        from a2a.types import Message, Part, Role, TextPart

        # The card is published per model, because an LlmAgent's model is fixed
        # when the agent is constructed and the server builds one tree per
        # allowed model at startup. Asking the wrong card would silently run the
        # stage on a model the session did not choose, so the model is in the
        # path rather than in the message.
        # Checked here rather than left to the fetch. Only the roles in
        # ``SPECIALISTS`` are published, and an unpublished one came back as a
        # 404 on a card URL -- which reads like a broken deployment when what is
        # broken is the caller's role string.
        published = {specialist.role for specialist in SPECIALISTS}
        if role not in published:
            raise ValueError(
                f"'{role}' is not a published specialist, so no A2A agent serves "
                "it. A differently prompted turn -- a critique, a debate -- "
                "still addresses the specialist whose work it is about."
            )
        name = specialist_agent_name(role, self.model_id)
        card_path = f"/a2a/specialists/{name}/.well-known/agent-card.json"
        http_client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        try:
            client = await ClientFactory.connect(
                self.base_url,
                client_config=ClientConfig(
                    streaming=True, polling=True, httpx_client=http_client
                ),
                relative_card_path=card_path,
            )
            message = Message(
                message_id=f"msg-{uuid.uuid4()}",
                role=Role.user,
                parts=[Part(root=TextPart(text=prompt))],
            )
            responses: list[str] = []
            grounding_urls: list[str] = []
            async for result in client.send_message(message):
                payload = result[0] if isinstance(result, tuple) else result
                dumped = payload.model_dump(mode="json", exclude_none=True)
                texts = self._texts(dumped)
                if texts:
                    responses = texts
                grounding_urls.extend(self._urls(dumped))
            resolved = await resolve_grounding_urls(
                dict.fromkeys(grounding_urls), client=http_client
            )
        finally:
            await http_client.aclose()
        if not responses:
            raise RuntimeError(f"A2A specialist '{role}' returned no text artifact.")
        content = self._without_prompt_echo(responses, prompt)
        if resolved:
            content += "\n\nGrounding URLs (resolved to the document each one opens):\n"
            content += "\n".join(f"- {url}" for url in resolved)
        return content

    @staticmethod
    def _without_prompt_echo(responses: list[str], prompt: str) -> str:
        """Keep specialist output while removing A2A task-history prompt echoes."""
        normalized_prompt = prompt.strip()
        unique = [
            text.strip()
            for text in dict.fromkeys(responses)
            if text.strip() and text.strip() != normalized_prompt
        ]
        cleaned = "\n".join(unique).strip()
        if cleaned.endswith(normalized_prompt):
            cleaned = cleaned[: -len(normalized_prompt)].rstrip()
        return cleaned

    @classmethod
    def _texts(cls, value) -> list[str]:
        if isinstance(value, dict):
            found = [
                item
                for key, child in value.items()
                if key == "text" and isinstance(child, str)
                for item in [child]
            ]
            for child in value.values():
                found.extend(cls._texts(child))
            return found
        if isinstance(value, list):
            return [text for child in value for text in cls._texts(child)]
        return []

    @classmethod
    def _urls(cls, value) -> list[str]:
        if isinstance(value, dict):
            found = [
                child
                for key, child in value.items()
                if key in {"uri", "url"}
                and isinstance(child, str)
                and child.startswith(("http://", "https://"))
            ]
            for child in value.values():
                found.extend(cls._urls(child))
            return found
        if isinstance(value, list):
            return [url for child in value for url in cls._urls(child)]
        return []

    def complete(self, *, role: str, prompt: str) -> str:
        import asyncio

        return asyncio.run(self._complete(role=role, prompt=prompt))


def bind_provider_model(provider, model: str):
    """Point a provider at the model the loaded session was configured with.

    The web API and the CLI both have to build a provider before they can read
    the session that says which model it should address, so the correction has
    to happen somewhere. Only :class:`A2AProvider` is rebound: the offline
    provider *is* its own model, and a test double's ``model_id`` is usually the
    thing being asserted on, so neither should be quietly rewritten.
    """
    if isinstance(provider, A2AProvider):
        provider.model_id = model
    return provider


@dataclass(frozen=True)
class Specialist:
    stage: str
    role: str
    instruction: str

    def _build_actor_prompt(
        self, session: Session, checklist: str, prior: str, feedback: str
    ) -> str:
        from .disciplines import get_discipline_profile

        discipline = getattr(session, "discipline", None) or "general_interdisciplinary"
        profile = get_discipline_profile(discipline)
        actor_instructions = profile.get_actor_guidance(self.stage)
        discipline_checklist_items = profile.get_stage_checklist(self.stage)
        discipline_checklist = (
            "\n".join(f"- {item}" for item in discipline_checklist_items)
            if discipline_checklist_items
            else ""
        )
        discipline_checklist_section = (
            f"Discipline-specific scientific-method checklist ({profile.name}):\n{discipline_checklist}\n"
            if discipline_checklist
            else ""
        )
        return (
            f"You are the Lead Scientific Specialist for the role '{self.role}' "
            f"at the '{self.stage}' stage of the Co-Scientist system.\n"
            f"Research question: {session.question}\n"
            f"Declared research mode: {session.research_mode}\n"
            f"Scientific discipline: {profile.name}\n"
            f"Required scientific-method checks:\n{checklist}\n"
            f"{discipline_checklist_section}"
            f"Role instruction: {self.instruction}\n"
            f"Prior work:\n{prior}\n"
            f"{citation_rule(session)}\n"
            f"Human feedback: {feedback}\n"
            f"{actor_instructions}\n"
            "Produce a highly professional, scientifically rigorous, and complete result "
            "for this role. Ensure all claims, controls, falsifiers, and mechanisms "
            "are domain-specific and testable. Do not use placeholder text or truncate fields. "
            "When a structured contract is requested, return one JSON object and do not wrap it in prose.\n\n"
            # Immediately above the contract, because what the language clause
            # mostly does is carve out the parts of the contract it must not
            # touch: the field names and the enumerated values. Put it at the
            # top of the prompt instead and several hundred lines of prior work
            # separate the exception from the rule it excepts.
            f"{session_language_clause(session)}"
            f"{output_contract(self.role)}"
        )

    def _build_critic_prompt(
        self, session: Session, content: str, round_num: int, checklist: str
    ) -> str:
        from .disciplines import get_discipline_profile

        discipline = getattr(session, "discipline", None) or "general_interdisciplinary"
        profile = get_discipline_profile(discipline)
        critic_rubric = profile.get_critic_rubric(self.stage)
        discipline_checklist_items = profile.get_stage_checklist(self.stage)
        discipline_checklist = (
            "\n".join(f"- {item}" for item in discipline_checklist_items)
            if discipline_checklist_items
            else ""
        )
        discipline_checklist_section = (
            f"Discipline-specific scientific-method checklist ({profile.name}):\n{discipline_checklist}\n"
            if discipline_checklist
            else ""
        )
        return (
            f"You are the Lead Scientific Critic and Quality Reviewer for the '{self.role}' specialist "
            f"at the '{self.stage}' stage of the Co-Scientist system.\n"
            f"Research Question: {session.question}\n"
            f"Declared Research Mode: {session.research_mode}\n"
            f"Scientific Discipline: {profile.name}\n"
            f"Required Scientific-Method Checklist:\n{checklist}\n"
            f"{discipline_checklist_section}"
            f"Stage Requirements and Role Purpose: {self.instruction}\n\n"
            # The real bound, not a fixed ten. A critic told it has ten rounds
            # left when it has two paces its objections for a conversation that
            # ends after the next one.
            f"--- CURRENT ACTOR DRAFT (Round {round_num}/{CRITIC_ROUNDS}) ---\n{content}\n\n"
            "Evaluate this draft with maximum scientific rigor against the following criteria:\n"
            "1. Completeness & Schema Compliance: Does it provide all required fields, tables, or JSON contracts without omissions or truncation?\n"
            "2. Scientific Rigor & Plausibility: Are mechanisms, controls, falsifiers, or citations domain-specific, plausible, and testable?\n"
            f"3. Domain Quality Rubric & Rigor Pillars ({profile.name}):\n{critic_rubric}\n"
            "4. Epistemic Integrity: Are hypotheses clearly distinguished from verified empirical claims?\n\n"
            "If the draft fully satisfies all scientific and structural requirements, reply with EXACTLY:\n"
            "SATISFIED\n\n"
            "If the draft has ANY deficiencies, omissions, or areas requiring improvement, provide a concise, actionable bulleted critique of what the Actor must change. Do NOT output SATISFIED if changes are needed."
        )

    def _refined_by_critic(
        self,
        session: Session,
        provider: Provider,
        actor_prompt: str,
        content: str,
        checklist: str,
        typed: TypedPayload,
    ) -> tuple[str, TypedPayload]:
        """Let a domain critic send the draft back until it stops objecting.

        Bounded by :data:`CRITIC_ROUNDS` rather than run to exhaustion. Each
        round costs two model calls, and there are seventeen specialists across
        nine stages: at ten rounds a single run can make three hundred calls
        nobody is waiting on productively, and the rounds that change anything
        are the early ones.

        A deterministic provider is skipped entirely. Its answers are fixtures,
        so a critique of one is a critique of a constant, and looping would
        replace the fixture with revision prose every offline test then fails
        to recognise.

        Every revision is re-validated. A round that improves the prose but
        breaks the payload is not an improvement, and without the re-check the
        artifact would carry the last draft's text beside the first draft's
        typed contract.
        """
        if getattr(provider, "deterministic", False) or CRITIC_ROUNDS < 1:
            return content, typed
        model = ROLE_CONTRACTS.get(self.role)
        for round_number in range(1, CRITIC_ROUNDS + 1):
            # Addressed to the specialist's own role, not to a "<role>_critic".
            # A role is the address of a published A2A agent, and only the
            # seventeen specialists are published: a critic role resolved to no
            # agent card, so against the deployment every run died at the first
            # stage with a 404 for
            # ``/a2a/specialists/goal_manager_critic/.well-known/agent-card.json``.
            # The critique is a different prompt to the same specialist, the way
            # the tournament judge is the ranking specialist under a debate
            # prompt.
            critique = provider.complete(
                role=self.role,
                prompt=self._build_critic_prompt(
                    session, content, round_number, checklist
                ),
            ).strip()
            verdict = critique.upper()
            if verdict.startswith("SATISFIED") or "NO MAJOR CRITI" in verdict:
                break
            revision = provider.complete(
                role=self.role,
                prompt=(
                    f"{actor_prompt}\n\n"
                    f"--- PREVIOUS DRAFT (round {round_number}) ---\n{content}\n\n"
                    f"--- SCIENTIFIC REVIEWER CRITIQUE ---\n{critique}\n\n"
                    "Address every point in the critique above and produce a "
                    "revised, superior draft. Return only the complete revised "
                    "result, in the same form the role asked for."
                ),
            )
            retyped = typed_specialist_payload(session, self.role, revision)
            if model is not None and retyped.source == "deterministic_fallback":
                break
            content, typed = revision, retyped
        return content, typed

    def run(self, session: Session, provider: Provider, feedback: str = "") -> Artifact:
        # Before anything is dispatched, because the discipline decides which
        # actor guidance and which critic rubric the prompts below are built
        # from. Scope reclassifies from the catch-all: by then the question has
        # been read once more and the reading is often sharper.
        if not getattr(session, "discipline", None) or (
            self.stage == "scope"
            and getattr(session, "discipline", "general_interdisciplinary")
            == "general_interdisciplinary"
        ):
            classified = classify_discipline(session.question)
            if classified != "general_interdisciplinary" or not getattr(
                session, "discipline", None
            ):
                session.discipline = classified
        if self.role == "ranking" and not getattr(provider, "deterministic", False):
            # Asking a model to report a tournament yields Elo numbers for
            # matches nobody played. Play them instead: the ratings then follow
            # from judgements that are each recorded with their own transcript.
            return self._ranked_by_debate(session, provider, feedback)
        prior_parts = []
        for artifact in session.artifacts:
            if artifact.status != ArtifactStatus.ACCEPTED:
                continue
            body = (
                json.dumps(artifact.payload, ensure_ascii=False)
                if artifact.payload
                else artifact.content
            )
            prior_parts.append(
                f"[{artifact.id} | {artifact.stage} | {artifact.schema_name}]\n{body}"
            )
        prior = "\n\n".join(prior_parts)
        checklist = "\n".join(
            f"- {requirement}"
            for requirement in method_requirements(session.research_mode)
        )
        actor_prompt = self._build_actor_prompt(session, checklist, prior, feedback)
        content = provider.complete(role=self.role, prompt=actor_prompt)
        typed = typed_specialist_payload(session, self.role, content)
        model = ROLE_CONTRACTS.get(self.role)
        live = not getattr(provider, "deterministic", False)
        if typed.source == "deterministic_fallback" and model is not None and live:
            # One targeted retry that names the exact validation failure. The
            # alternative is discarding real reasoning for a template, which is
            # far more expensive than a second call.
            # The repair prompt carries the language clause too. Without it the
            # retry is a bare English instruction to fix a payload, and the
            # model reads that as a change of working language and rewrites the
            # prose it had already produced correctly.
            retry = provider.complete(
                role=self.role,
                prompt=session_language_clause(session)
                + repair_prompt(model, content, typed.error),
            )
            retyped = typed_specialist_payload(session, self.role, retry)
            if retyped.source == "deterministic_fallback":
                raise ContractViolation(self.role, retyped.error or typed.error, retry)
            content, typed = retry, retyped
        # Refinement runs on a draft that already satisfies its contract. A
        # critic asked to improve output the parser rejected is reviewing text
        # that is about to be thrown away, and it spends two model calls a round
        # to do it.
        content, typed = self._refined_by_critic(
            session, provider, actor_prompt, content, checklist, typed
        )
        return Artifact(
            stage=self.stage,
            agent=self.role,
            content=content,
            feedback=feedback,
            producer_model=getattr(provider, "model_id", "unknown"),
            schema_name=typed.schema_name,
            payload=typed.payload,
            payload_source=typed.source,
            payload_repairs=typed.repairs,
            payload_error=typed.error,
        )

    def _ranked_by_debate(
        self, session: Session, provider: Provider, feedback: str
    ) -> Artifact:
        """Rank by running the section 9.3 tournament with the model as judge.

        The state is constructed here rather than parsed, so it never passes
        through ``parsed_tournament_state``'s fallback: there is nothing to fall
        back to, because the tournament is the model's own work.
        """
        tournament, transcript = run_debate_tournament(session, provider)
        return Artifact(
            stage=self.stage,
            agent=self.role,
            content=transcript,
            feedback=feedback,
            producer_model=getattr(provider, "model_id", "unknown"),
            schema_name="TournamentState",
            payload=tournament.model_dump(mode="json"),
            payload_source="specialist",
        )


SPECIALISTS = (
    Specialist(
        "scope",
        "goal_manager",
        "Turn the question into a constrained, falsifiable research objective. "
        "Block residue-specific peptide work without a sequence and observed "
        "single-cell/spatial claims without a dataset or accession; offer only "
        "an explicitly labeled literature-only fallback.",
    ),
    Specialist(
        "evidence",
        "evidence_discovery",
        "Discover literature with Google Search in whichever of two modes the "
        "dispatch names. Enrichment mode: resolve only the unresolved gaps Deep "
        "Research listed, at most six focused queries, and do not repeat the "
        "broad search it already ran. Primary mode, used when the dispatch says "
        "Deep Research was unavailable: nobody has searched yet, so run the "
        "broad search yourself and cover supporting work, contradicting work, "
        "and measurement standards. In both modes every source and claim stays "
        "discovered_unverified -- reading a search result is not verification.",
    ),
    Specialist(
        "evidence",
        "source_verification",
        "Inspect permitted original sources, normalize identifiers, and map exact claim support; never treat snippets as verified evidence.",
    ),
    Specialist(
        "generate",
        "generation",
        "Generate diverse, falsifiable candidate hypotheses.",
    ),
    Specialist(
        "generate",
        "generation_evidence_first",
        "Generate 2 candidates by bridging established empirical findings with unexplained anomalies or gaps in the Knowledge Base.",
    ),
    Specialist(
        "generate",
        "generation_mechanism_first",
        "Generate 2 bottom-up candidates by constructing novel causal pathways, mathematical formulations, or biophysical/computational models.",
    ),
    Specialist(
        "generate",
        "generation_analogy_transfer",
        "Generate 2 candidates by transferring validated control mechanisms, algorithms, or structural motifs from adjacent scientific domains.",
    ),
    Specialist(
        "generate",
        "generation_competing_explanation",
        "Generate 2 rival candidates that challenge the prevailing consensus or dominant interpretation in the Knowledge Base.",
    ),
    Specialist(
        "reflect",
        "reflection",
        "Red-team candidates for evidence, feasibility, safety, and causality.",
    ),
    Specialist(
        "reflect",
        "novelty_review",
        "Independently review prior art, novelty, and incrementalism for every candidate.",
    ),
    Specialist(
        "reflect",
        "methods_statistics",
        "Audit design, measurement, causal assumptions, sampling, analysis, uncertainty, and replication for the declared research mode.",
    ),
    Specialist(
        "reflect",
        "ethics_safety_governance",
        "Audit ethics, safety, privacy, data rights, dual-use, and required institutional approvals; block unsafe promotion.",
    ),
    Specialist(
        "reflect",
        "impact_review",
        "Independently review expected information gain, importance, feasibility, cost, time, and external validity.",
    ),
    Specialist(
        "rank",
        "ranking",
        "Judge one pair of hypotheses per request on novelty, impact, "
        "feasibility, and risk -- by direct comparison, or by simulated "
        "scientific debate when asked -- and name the better idea. The "
        "Supervisor plays the tournament and computes Elo from these verdicts.",
    ),
    Specialist(
        "evolve",
        "evolution",
        "Refine the leading candidate into a testable protocol and falsifier.",
    ),
    Specialist(
        "proximity",
        "proximity",
        "Map related work and contradictions; never fabricate citations.",
    ),
    Specialist(
        "meta_review",
        "meta_reviewer",
        "Audit the entire proposal and issue a conditional go/no-go.",
    ),
)


SPECIALISTS_BY_STAGE = {
    stage: tuple(item for item in SPECIALISTS if item.stage == stage)
    for stage in {item.stage for item in SPECIALISTS}
}


def _generation_config(choice, types):
    """The sampling and thinking settings one model family needs.

    Gemini and Claude disagree about how thinking is asked for, and the
    disagreement is fatal rather than cosmetic: ADK's Anthropic adapter raises
    outright on a ``ThinkingConfig`` that carries a level instead of a budget.
    The ceilings differ too, and for a reason worth writing down -- Anthropic's
    SDK refuses a non-streaming request whose token ceiling implies more than
    ten minutes of work, which caps Claude far below what Gemini allows.
    """
    if choice.family == "claude":
        return types.GenerateContentConfig(
            # Anthropic ignores sampling parameters once thinking is on and ADK
            # warns about it, so the temperature the Gemini agents carry is not
            # repeated here. -1 is adaptive: the model picks its own depth,
            # which is the closest equivalent to Gemini's HIGH level.
            max_output_tokens=choice.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
        )
    return types.GenerateContentConfig(
        temperature=0.2,
        # A full CandidatePopulation or TournamentState runs past 8k tokens; the
        # old ceiling truncated payloads mid-string and the whole stage was lost.
        max_output_tokens=choice.max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
    )


def _adk_model(model_id: str):
    """The value an ``LlmAgent`` wants for ``model=``.

    A Gemini id is passed through as a string. A Claude id cannot be: ADK's
    registry matches ``claude-3-.*`` and ``claude-.*-4.*``, so a Claude 5 id
    resolves to nothing and the agent fails to construct. Handing over the
    model object directly skips the registry, which is the documented escape
    hatch for exactly this case.
    """
    if model_choice(model_id).family != "claude":
        return model_id
    try:
        from google.adk.models.anthropic_llm import Claude
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"Serving {model_id} needs the anthropic package: pip install "
            "'anthropic>=0.43.0'."
        ) from exc
    return Claude(model=model_id)


def build_adk_workflow(model: str = GEMINI_MODEL):
    """Build specialist agents for the A2A surface, for one model.

    The code-level Supervisor remains outside this LLM delegation tree. This
    function intentionally does not use ``SequentialAgent`` because that would
    bypass the persisted approval state machine.

    Called once per allowed model. An ``LlmAgent`` binds its model when it is
    constructed, so a run cannot choose one later; what it chooses instead is
    which of these trees to address, and the agent names carry the model so the
    published A2A cards stay distinct.
    """
    try:
        from google.adk.agents import Agent, LlmAgent
        from google.adk.tools import google_search
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install google-adk to create the live ADK graph.") from exc
    configure_vertex_ai_global_endpoint()
    choice = model_choice(model)
    config = _generation_config(choice, types)
    # Google Search grounding is a Gemini server-side feature, not a function
    # any model could call, and ADK raises rather than degrading when a
    # non-Gemini model is handed the tool. Evidence discovery is the one role
    # that needs it, so on a Claude tree that role alone stays on the default
    # Gemini model and its own config. The alternative -- dropping the tool --
    # would leave discovery inventing sources, which is the single failure this
    # system exists to prevent.
    grounded_choice = choice if choice.search_grounded else model_choice(DEFAULT_MODEL)
    grounded_config = (
        config if choice.search_grounded else _generation_config(grounded_choice, types)
    )
    # Built once and shared. For Gemini this is just the id string, but for
    # Claude it is a client-holding object, and one per specialist would open
    # eleven of them per tree for no benefit.
    chosen_model = _adk_model(model)
    grounded_model = (
        chosen_model if choice.search_grounded else _adk_model(grounded_choice.id)
    )
    agents = []
    for item in SPECIALISTS:
        agent_model, agent_config = chosen_model, config
        if item.role == "evidence_discovery":
            tools = [google_search]
            agent_model = grounded_model
            agent_config = grounded_config
        elif item.role == "source_verification":
            # ADK's load_web_page was here and could not verify anything in this
            # deployment: it imports beautifulsoup4, which this project does not
            # install, so every call raised ImportError; it sends no user agent a
            # publisher will serve; and it passes allow_redirects=False, so every
            # doi.org link -- the only kind that unambiguously names a paper --
            # came back a bare 302. A live run discovered forty-four sources and
            # verified none of them.
            tools = [fetch_source_document]
        else:
            tools = []
        # The ranking specialist is a match judge, not a tournament reporter.
        # It previously carried output_contract("ranking"), which orders a
        # TournamentState; that system instruction beat the section 9.3 user
        # prompt and the model returned a serialized tournament with invented
        # candidate ids instead of debating. Fixing it here fixes the A2A card
        # too, since the card is generated from this agent.
        contract = (
            RANKING_JUDGE_CONTRACT
            if item.role == "ranking"
            else output_contract(item.role)
        )
        agents.append(
            LlmAgent(
                name=specialist_agent_name(item.role, model),
                model=agent_model,
                instruction=(
                    f"{item.instruction}\n\n"
                    "Operate only within this responsibility. Distinguish verified "
                    "evidence, unverified leads, inference, and proposals. Never "
                    "invent sources, results, measurements, or tool output.\n\n"
                    f"{contract}"
                ),
                tools=tools,
                generate_content_config=agent_config,
                description=item.instruction,
            )
        )
    return Agent(
        name=specialist_agent_name("co_scientist_supervisor", model),
        model=chosen_model,
        instruction=(
            "You are the conversational entry point to a rigor-first scientific "
            "workflow. Route the user's requested deliverable directly to the "
            "matching specialist; do not insert a generic planning step when the "
            "required input is already present. In particular, a request for "
            "competing hypotheses with an explicit peptide sequence goes directly "
            "to generation. Before scientific generation, check input sufficiency: "
            "exact peptide fragmentation requires the residue sequence; claims "
            "about observed scRNA-seq clusters, differential expression, trajectories, "
            "or spatial relationships require a supplied dataset or accession. Ask "
            "for the missing input or offer a clearly labeled literature-only plan; "
            "never invent it. The external deterministic Supervisor owns workflow "
            "state, approval, budgets, and promotion. Never claim to approve, execute "
            "real-world research, or advance a research stage."
        ),
        sub_agents=agents,
        generate_content_config=config,
    )


def build_adk_workflows() -> dict:
    """One specialist tree per model on the allowlist, keyed by model id.

    The server publishes every tree at startup so a session can pick its model
    per run. Trees are cheap to build -- no network call, just agent objects --
    but they are not free, so this is the only place that decides how many
    exist, and it decides from :data:`MODEL_CHOICES` rather than from a second
    list somebody has to remember to update.
    """
    return {choice.id: build_adk_workflow(model=choice.id) for choice in MODEL_CHOICES}
