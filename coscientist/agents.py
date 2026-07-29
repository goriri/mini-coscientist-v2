"""Specialist roles and optional Google ADK representation of the workflow."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Protocol

from .methods import method_requirements
from .models import Artifact, ArtifactStatus, Session
from .parity import typed_specialist_payload

GEMINI_MODEL = "gemini-3.1-pro-preview"
"""Current Vertex AI model ID for the reasoning-backed ADK workflow."""

VERTEX_LOCATION = "global"

STRUCTURED_OUTPUT_INSTRUCTIONS = {
    "goal_manager": (
        "Return one ResearchPlan JSON object with question, research_mode, "
        "intended_claim, assumptions, constraints, success_criteria, "
        "stopping_criteria, and governance_requirements."
    ),
    "evidence_discovery": (
        "Use Google Search and return one EvidencePacket JSON object. Every source "
        "and claim must remain discovered_unverified; include URLs and limitations."
    ),
    "source_verification": (
        "Return one EvidencePacket JSON object. A verified claim requires a source "
        "ID, original URL, exact supporting or contradicting location, relation, "
        "and correction/retraction status. Otherwise keep it unverified."
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
    "ethics_safety_governance": "Return one ReviewSet JSON object with one impact_safety governance review per candidate.",
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


class DeterministicProvider:
    """Offline provider used for a transparent demo and tests.

    It purposefully uses conservative templates rather than pretending to have
    searched literature or run experiments.
    """

    model_id = "deterministic-offline"

    def complete(self, *, role: str, prompt: str) -> str:
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

    model_id = GEMINI_MODEL

    def __init__(
        self, base_url: str = "http://127.0.0.1:8000", *, timeout: float = 120
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _complete(self, *, role: str, prompt: str) -> str:
        import httpx
        from a2a.client import ClientConfig, ClientFactory
        from a2a.types import Message, Part, Role, TextPart

        card_path = f"/a2a/specialists/{role}/.well-known/agent-card.json"
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
        finally:
            await http_client.aclose()
        if not responses:
            raise RuntimeError(f"A2A specialist '{role}' returned no text artifact.")
        content = self._without_prompt_echo(responses, prompt)
        if grounding_urls:
            content += "\n\nGrounding URLs:\n" + "\n".join(
                f"- {url}" for url in dict.fromkeys(grounding_urls)
            )
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


@dataclass(frozen=True)
class Specialist:
    stage: str
    role: str
    instruction: str

    def run(self, session: Session, provider: Provider, feedback: str = "") -> Artifact:
        prior_parts = []
        for artifact in session.artifacts:
            if artifact.status != ArtifactStatus.ACCEPTED:
                continue
            # Typed artifacts are immutable collaboration memory. Preserve their
            # complete payload rather than silently clipping source leads,
            # candidate details, reviews, or lineage.
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
        prompt = (
            f"Research question: {session.question}\n"
            f"Declared research mode: {session.research_mode}\n"
            f"Required scientific-method checks:\n{checklist}\n"
            f"Role: {self.instruction}\nPrior work:\n{prior}\n"
            f"Human feedback: {feedback}\n"
            "Return a rigorous result for this role. When a structured contract "
            "is requested, return one JSON object and do not wrap it in prose."
        )
        content = provider.complete(role=self.role, prompt=prompt)
        schema_name, payload = typed_specialist_payload(session, self.role, content)
        return Artifact(
            stage=self.stage,
            agent=self.role,
            content=content,
            feedback=feedback,
            producer_model=getattr(provider, "model_id", "unknown"),
            schema_name=schema_name,
            payload=payload,
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
        "Perform targeted Google Search enrichment only for the unresolved gaps "
        "listed by Deep Research. Use no more than six focused queries; preserve "
        "all results as discovered_unverified and never repeat the broad search.",
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
        "Rank candidates by novelty, impact, feasibility, and risk; state uncertainty.",
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


def build_adk_workflow(model: str = GEMINI_MODEL):
    """Build specialist agents for the A2A surface.

    The code-level Supervisor remains outside this LLM delegation tree. This
    function intentionally does not use ``SequentialAgent`` because that would
    bypass the persisted approval state machine.
    """
    try:
        from google.adk.agents import Agent, LlmAgent
        from google.adk.tools import google_search
        from google.adk.tools.load_web_page import load_web_page
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install google-adk to create the live ADK graph.") from exc
    configure_vertex_ai_global_endpoint()
    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=8192,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
    )
    agents = []
    for item in SPECIALISTS:
        if item.role == "evidence_discovery":
            tools = [google_search]
        elif item.role == "source_verification":
            tools = [load_web_page]
        else:
            tools = []
        agents.append(
            LlmAgent(
                name=item.role,
                model=model,
                instruction=(
                    f"{item.instruction}\n\n"
                    "Operate only within this responsibility. Distinguish verified "
                    "evidence, unverified leads, inference, and proposals. Never "
                    "invent sources, results, measurements, or tool output.\n\n"
                    f"{STRUCTURED_OUTPUT_INSTRUCTIONS[item.role]}"
                ),
                tools=tools,
                generate_content_config=config,
                description=item.instruction,
            )
        )
    return Agent(
        name="co_scientist_supervisor",
        model=model,
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
