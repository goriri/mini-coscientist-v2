# Co-Scientist: Domain-General Scientific Research Improvement Plan

## Purpose and design basis

Co-Scientist must support scientific work across disciplines—not merely produce
biomedical or chemistry proposals. It should help a researcher move from a
well-defined question to a transparent, testable, and ethically reviewable
research plan. It must not claim that a hypothesis is true, fabricate evidence,
or perform laboratory, clinical, field, or safety-critical actions.

This plan uses `coscientist-design.pdf` as the architectural benchmark and
`coscientist-sample1.pdf`, `sample2.pdf`, and `sample3.pdf` as process
benchmarks. The reports are not scientific ground truth. Sample 2 contains 222
pages and about 110,000 extracted words; Sample 3 contains 174 pages and about
86,000 words. They demonstrate broad research-direction mapping, 8–9 candidate
ideas, multiple independent reviews, Elo-style ranking, iteration, and a
polished overview backed by a complete artifact dossier.

They also demonstrate why parity must be rigor-first. Sample 2 selects exact
Gly/Ala peptide junctions without receiving the peptide sequence. Sample 3
describes observed scRNA-seq clusters, trajectories, and spatial relationships
without receiving a dataset. Both reports contain citations explicitly labeled
inaccurate, unsupported, or disputed, and some final recommendations do not
fully reflect fatal flaws identified by reviewers. The local implementation
must match their depth without imitating unsupported specificity.

## Current implementation status

As of 2026-07-29, the project includes the agents-cli runtime and generated A2A
endpoints, a resumable ADK app, purpose-specific specialist Agent Cards,
dedicated Google Search discovery and page-loading verification roles, typed
parity contracts, six method adapters, asynchronous task dispatch, and four
audited interaction profiles: `auto`, `milestone`, `stage`, and `artifact`.
Local development uses SQLite. The public Cloud Run deployment can use
PostgreSQL and GCS-backed persistence when its production storage configuration
is enabled.

The deterministic Supervisor detects missing sequence- and dataset-dependent
inputs, supports an explicit literature-only fallback, creates an eight-item
candidate population, records independent reviews, runs a bounded Swiss/Elo
tournament, evolves the shortlist, builds a research landscape, and exports
Markdown, DOCX, and PDF dossiers. The web UI also has typed stage
presentations.

`result.pdf` shows that these structural capabilities are not yet producing a
research-grade report. In that run, Evidence Discovery returned nine
unverified leads and zero claim records; Source Verification returned zero
sources and zero claims. The workflow nevertheless generated, ranked, evolved,
and recommended candidates. The concise candidate population is generic and
repetitive, while domain-specific material appears later as unvalidated raw
payload. The report then exposes the complete internal ledger by default. These
are promotion-policy, normalization, knowledge-grounding, and report-compilation
defects—not merely visual defects.

The sample-derived behavioral evaluation, repeated quality-flywheel iterations,
claim-support audit, failure-injection campaign, and qualified domain-expert
review remain production-quality gates even though a public development
deployment exists.

## Official-report gap assessment

| Capability | Official samples | Earlier local behavior | Implemented parity behavior |
| --- | --- | --- | --- |
| Input sufficiency | Produced sequence- and dataset-specific claims without required inputs. | Accepted nearly any question and emitted a generic scope. | Blocking typed requirements plus provided-input or explicitly labeled literature-only continuation. |
| Candidate breadth | 8–9 detailed candidates organized into research directions. | Three generic templates. | Eight typed candidates from four independent generation strategies. |
| Review depth | Multiple correctness, novelty, feasibility, impact, and adversarial reviews. | One generic reflection plus broad methods/governance prose. | Five purpose-specific review passes over every candidate, preserving objections, assumptions, confidence, and fatal flaws. |
| Comparison | Candidate tables, ranks, and Elo values. | Static numbered ranking. | Three randomized-order Swiss rounds, a top-four round robin, Elo history, shortlist, and convergence fields. |
| Iteration | Proposals explicitly address earlier objections and revisions. | One untracked refinement. | Versioned top-four evolution records with parents, changes, addressed critiques, new predictions, and mandatory re-review. |
| Evidence | Large search corpus and claim confidence labels, but inconsistent source quality. | Search/verifier agents existed without populated claim contracts. | Search and verifier boundaries exist, but `result.pdf` proves that an empty verified packet can still be accepted and later stages are not grounded in claim records. |
| Reporting | Polished overview, explicit research directions, a knowledge summary, linked references, comparison tables, and a long supporting dossier. | Roughly 70 lines formed by concatenating stage prose. | Exports exist, but the default PDF concatenates human summaries, raw specialist content, typed JSON, and audit records into one document. |

## `result.pdf` versus `sample2.pdf`: report audit

This comparison concerns format, organization, and research process. It does
not treat the official report as scientific ground truth. In particular,
`sample2.pdf` proposes exact Gly15–Ala16 and Gly30–Ala31 fragmentation points
without having received the peptide sequence, and it labels some references
inaccurate or disputed. The local system must preserve its stricter
input-sufficiency rules.

### Measured differences

| Measure | Local `result.pdf` | Official `sample2.pdf` | Interpretation |
| --- | ---: | ---: | --- |
| Pages | 153 | 222 | The local report is not concise despite containing much less research prose. |
| Extracted words | 26,740 | 108,852 | The official report provides roughly four times the explanatory content. |
| PDF link annotations | 0 | 6,495 | The local citations and URLs are not navigable; the official export heavily links its source material. |
| Candidate-ID-like mentions | 1,172 | 2 | Internal identity dominates the local reader experience. |
| Artifact-ID-like mentions | 171 | 0 | Audit-storage details are presented as research content locally. |
| Raw JSON/brace markers | 4,722 | 80 | Most of the local dossier is a serialization dump rather than an edited report. |
| PDF outline/bookmarks | None | None | Both reports need a navigable PDF outline; this is not a feature to copy from the official sample. |

The counts are lexical diagnostics rather than semantic quality scores, but the
difference is unambiguous. Pages 1–9 of the local report form a short summary;
pages 10–153 are primarily complete raw artifacts and JSON. About 94% of the
local pages therefore serve the audit representation rather than the reader.
The official report instead spends pages 2–14 on a structured overview, begins
a dedicated Knowledge Base on page 15, presents a long knowledge synthesis and
research directions before its idea dossier, and uses reader-oriented idea
names, comparison tables, judgments, objections, and references.

### Content-organization findings

| Area | Local report | Official report | Required direction |
| --- | --- | --- | --- |
| Opening | Query, runtime settings, internal candidate IDs, and artifact metadata appear immediately. | Research goal, evaluation criteria, top ideas, directions, comparison, and recommendation are introduced in reader language. | Open with decision context, evidence status, named directions, and plain-language recommendations. Put runtime metadata elsewhere. |
| Knowledge grounding | Nine unverified discovery URLs, zero verified sources, zero claim records, no knowledge synthesis, and no bibliography. | A visible Knowledge Base summarizes concepts, mechanisms, methods, open questions, unexpected connections, and references. | Add a mandatory, versioned evidence baseline before generation. |
| Candidate communication | The concise candidates repeat generic predictions, alternatives, falsifiers, and risks. More specific candidate material appears only in raw payloads. | Ideas have distinct titles, mechanisms, motivation, evidence, weaknesses, and evaluations. | Reject generic fallback populations and require domain-specific, evidence-linked candidate cards. |
| Review and ranking | Repeated `0.45` reviews and `insufficient_evidence` coexist with a numerical ranking and final recommendations. Tables contain opaque IDs. | Reviews, comparative judgments, and rankings are explained using idea names, although consistency is imperfect. | Prevent evidence-poor ranking from becoming a recommendation; reconcile every fatal or insufficient-evidence judgment. |
| Visual rendering | Markdown emphasis, pipe tables, HTML tags, checksums, and JSON are printed literally in 8-point text. There are no links, contents page, headers, or page numbers. | Consistent typography, page headers/footers, conventional tables, numbered citations, links, and readable section hierarchy. | Render structured report blocks directly and add navigation, typography, tables, citations, and page templates. |
| Provenance | Full internal payloads are mixed into the report. | Supporting material remains long but is mostly expressed as readable prose and reference lists. | Separate the scientific report from a machine-readable audit export. |

### Important conclusion

The official report's advantage is not its length. It establishes a reader's
mental model—goal, criteria, evidence landscape, research directions, ideas,
reviews, comparison, and recommendation—before exposing detailed work. The
local report establishes the storage model—artifact, schema, checksum, ID, and
JSON—and expects the reader to reconstruct the science. The redesign must
reverse that priority without weakening provenance.

## Definition of success

For every supported research question, Co-Scientist should produce an auditable
research brief that contains:

1. A clearly scoped question and a declared research mode.
2. A falsifiable hypothesis or a justified alternative when hypothesis testing
   is not appropriate (for example, systematic review, measurement study,
   observational study, model comparison, or exploratory mapping).
3. An explicit causal or theoretical rationale, competing explanations, and
   predictions that differ among them.
4. A feasible method: variables or constructs, measurements, sampling,
   controls/comparators, bias mitigation, analysis, uncertainty, and stopping
   conditions appropriate to the field.
5. A source and data provenance ledger, with evidence strength and verification
   status for every material claim.
6. A plan for replication, reproducibility, ethical/safety review, and human
   domain-expert approval before any real-world action.

## Knowledge base as the evidence substrate

The Knowledge Base must be a first-class workflow artifact, not a list of URLs
inside Generation. Add a visible `evidence` stage between Scope and Generate.
No later stage should receive the whole conversation or raw search output.
Instead, it receives an immutable, versioned evidence baseline plus a
stage-specific retrieval packet.

### Knowledge-building workflow

```text
Research question
      │
      ▼
Question decomposition and research-direction map
      │ terms, entities, mechanisms, outcomes, constraints, rival explanations
      ▼
Evidence discovery ── supporting / contradicting / null / replication /
      │                 methods / safety / correction / retraction searches
      ▼
Source acquisition and identity resolution
      │ DOI, PMID, trial/dataset/standard ID, canonical URL, source type
      ▼
Claim extraction and independent source verification
      │ exact location, relation, confidence, limitations, correction status
      ▼
Deduplication, quality appraisal, contradiction graph, coverage audit
      │
      ▼
KnowledgeBaseManifest v1 ── frozen evidence baseline
      │
      ├── retrieval packet for candidate generation
      ├── independent packet for each reviewer
      ├── evidence-on-demand delta requests during evolution
      └── claim/citation graph for the report compiler
```

1. **Decompose before searching.** The Goal agent creates four to eight
   research directions appropriate to the question. Each direction declares
   its key concepts, causal or theoretical relationships, outcomes, boundary
   conditions, required data, and most important rival explanation.
2. **Search for a balanced evidence set.** Evidence Discovery uses Google
   Search for primary and authoritative leads, but explicitly runs supporting,
   contradictory, null/negative, replication, methods, safety, and
   correction/retraction queries. Search snippets remain leads.
3. **Resolve and verify originals.** Source Verification opens the original
   source where permitted, resolves stable identifiers and bibliographic
   metadata, checks correction/retraction state, and maps each material claim
   to an exact section, page, figure, table, or result location. An inaccessible
   source remains unverified and cannot silently satisfy a gate.
4. **Curate, do not concatenate.** A Knowledge Curator deduplicates sources,
   grades source type and directness, identifies conflicts and evidence gaps,
   and writes a bounded synthesis for each research direction. It does not
   invent consensus when sources disagree.
5. **Freeze a baseline.** The Supervisor creates an immutable
   `KnowledgeBaseManifest` with a version, search coverage, source and claim
   counts, unresolved contradictions, gaps, and a content checksum. Later
   searches create delta versions; they never rewrite the evidence used by an
   earlier decision.
6. **Retrieve by purpose.** Candidate generators receive direction-specific
   evidence packets. Reviewers receive the candidate and an independently
   assembled packet that includes counterevidence. Ranking receives review
   summaries and coverage metrics, not raw papers. The report compiler receives
   the curated claim graph and citations.

### Required typed contracts

Extend the existing evidence models rather than putting another prose field on
`EvidencePacket`:

| Contract | Required content |
| --- | --- |
| `ResearchDirection` | Reader-facing title, scope, mechanism or concept, outcome, competing explanations, required data, and search questions. |
| `SourceRecord` | Canonical identity, title, authors, year, source type, stable identifiers, canonical URL, access status, correction/retraction status, and quality limitations. |
| `EvidenceClaim` | Atomic claim, source, exact location, `supports` / `contradicts` / `neutral`, directness, verification status, confidence, and limitations. |
| `EvidenceGap` | Missing or conflicting evidence, affected directions/candidates, decision impact, and the query or input needed to resolve it. |
| `KnowledgeBaseManifest` | Versioned directions, source and claim references, coverage matrix, contradiction graph, gaps, search cut-off date, and evidence-quality summary. |
| `EvidenceRequest` | Requesting stage/agent, precise claim to verify, priority, budget, status, and resulting delta-manifest reference. |
| `CitationAnchor` | Claim reference plus a stable human citation number and report location; internal IDs never become citation labels. |

Store the complete objects in PostgreSQL/GCS and use hybrid lexical and semantic
retrieval over verified claim text and curated summaries. Retrieval must filter
by verification status, source type, research direction, date, and relation.
Embeddings improve recall but never replace source verification or the exact
location requirement.

### Promotion gates

- A discovery packet with zero claim records is not a knowledge base.
- A verification packet with zero verified claims cannot be `accepted` as
  evidence-complete.
- In normal research mode, Candidate Generation cannot begin until the
  knowledge baseline passes a mode-specific coverage audit. Every direction
  must have either verified grounding or a visible `EvidenceGap`.
- A candidate must link every material rationale, mechanism, and prior-art
  statement to verified claims. Predictions and hypotheses may be novel, but
  must be labeled as proposals rather than findings.
- If no evidence can be verified, the Supervisor offers an explicitly labeled
  exploratory or literature-lead-only workflow. That workflow may generate
  search hypotheses, but it cannot produce a ranked scientific recommendation.
- Ranking is blocked when all candidates are `insufficient_evidence`.
  Candidates may instead be prioritized for evidence acquisition.
- Evolution that adds a new mechanism or factual premise automatically creates
  an `EvidenceRequest` and returns the candidate to evidence/correctness review.
- Meta-review may recommend only candidates whose cited claims resolve, whose
  evidence coverage meets the selected mode's threshold, and whose fatal flaws
  are resolved. Otherwise the correct outcome is “insufficient evidence.”

### Reader-facing Knowledge Base section

The default report must place this material before the candidates:

1. Evidence status and search cut-off date.
2. Research-direction map.
3. Knowledge summary by direction: established findings, contested findings,
   negative/null evidence, methods, safety constraints, and transfer limits.
4. Claim–evidence matrix with confidence and contradiction status.
5. Evidence gaps and questions that would change the decision.
6. Numbered, linked references with source type and verification status.

This section should be broad enough to ground later reasoning but deliberately
curated. `sample2.pdf` demonstrates the value of a substantial knowledge
summary, but its long list of unrelated and sometimes inaccurate sources
demonstrates why source count and page count are poor acceptance metrics.
Coverage, relevance, verification, contradiction handling, and cross-stage
reuse are the correct metrics.

## A domain-general scientific method

The workflow should begin by classifying the request, while allowing the
researcher to override the classification. The scientific-method requirements
are then expressed in the language of the discipline rather than forcing every
project into a wet-lab experiment.

| Research mode | Required scientific-method outputs |
| --- | --- |
| Experimental science and engineering | Intervention, comparator/control, independent and dependent variables, randomization/blinding where applicable, sample/power rationale, calibration, safety limits, analysis plan, and replication experiment. |
| Observational, clinical, ecological, and social science | Target population, sampling frame, inclusion/exclusion criteria, confounders, causal-identification assumptions, missing-data plan, privacy/ethics review, robustness checks, and limits on causal claims. |
| Computational, data science, and AI | Dataset provenance and licenses, task definition, baselines, train/validation/test separation, metrics, ablations, compute/environment record, error analysis, reproducibility package, and misuse assessment. |
| Theory, mathematics, and simulation | Definitions, assumptions, derivation or proof obligations, boundary conditions, numerical verification, limiting cases, falsifiable consequences, and links to observable phenomena where relevant. |
| Systematic review and meta-analysis | Pre-registered protocol, databases and search strategy, eligibility criteria, screening/extraction procedure, risk-of-bias assessment, synthesis model, heterogeneity/sensitivity analysis, and certainty-of-evidence assessment. |
| Measurement, instrumentation, and field research | Construct and measurement model, calibration/traceability, uncertainty budget, sampling and site protocol, quality assurance, data management, and independent validation. |

Every mode uses the same epistemic cycle: define → review prior evidence →
generate explanations or models → derive discriminating predictions → design a
test or analysis → collect or inspect evidence under an approved protocol →
analyze uncertainty → attempt replication or robustness checks → revise or
reject the claim. Exploratory work must be labeled exploratory; confirmatory
claims require pre-specified tests and appropriate controls.

## Target architecture: purpose-built asynchronous research agents

The existing linear, human-gated workflow should evolve into a bounded,
event-driven multi-agent system. Each agent owns one scientific responsibility;
it must not silently perform another agent's decision. The Supervisor is the
only agent permitted to schedule work, merge a completed artefact into the
canonical research record, or request a human gate. It is a coordinator, not a
scientific authority.

```text
Researcher / TUI
        │ human approval, revision, stop
        ▼
Supervisor and research ledger ── A2A task lifecycle ──┐
        │                                               │
        ├── Goal & triage agent                         │
        ├── Evidence discovery agent (Google Search) ──┤
        ├── Source verification agent ─────────────────┤
        ├── Knowledge curator & direction mapper ──────┤
        ├── Generation agent                             │
        ├── Reflection / counterevidence agent           │
        ├── Methods & statistics agent                   │
        ├── Ethics, safety & governance agent            │
        ├── Ranking / tournament agent                   │
        ├── Evolution agent                              │
        ├── Proximity & diversity agent                  │
        ├── Meta-review agent ───────────────────────────┤
        ├── Scientific report architect ────────────────┤
        └── Citation & reader-quality auditor ───────────┘
                         │
                         ▼
       Versioned evidence ledger, candidate graph, artefact store
```

| Agent | Sole purpose | Inputs | Output artefact |
| --- | --- | --- | --- |
| Supervisor | Own the research context, budgets, dependencies, human gates, and final state transitions. | Research goal, approvals, A2A task statuses. | Task graph, audit log, approved research brief. |
| Goal & triage | Translate a request into a structured plan; identify missing information and screen research safety/ethics. | Researcher request and constraints. | `ResearchPlan` and triage decision. |
| Evidence discovery | Use Google Search to find supporting, contrary, negative, replicated, corrected, or retracted source leads; never judge a candidate. | Search questions approved by the Supervisor. | Grounded discovery records, all initially unverified. |
| Source verification | Retrieve allowed original sources, normalize identity, check correction/retraction status, and map claims to exact support; never rank a candidate. | Discovery records and requested claims. | Validated `SourceRecord` and `EvidencePacket` artefacts. |
| Knowledge curator & direction mapper | Build the verified research landscape without proposing or ranking candidates. Deduplicate evidence, synthesize agreements and conflicts, expose gaps, and freeze the evidence baseline. | Plan plus verified claims and sources. | `ResearchDirection` records and a `KnowledgeBaseManifest`. |
| Generation | Produce diverse hypotheses, models, or study designs grounded in approved evidence packets. | Plan, rubric, evidence packets. | Versioned `Candidate` records. |
| Reflection | Attempt to disprove candidates; check novelty, assumptions, competing explanations, and evidence contradictions. | Candidate and evidence packets. | `EvidenceReview` with a promotion/rejection recommendation. |
| Methods & statistics | Check the scientific method for the chosen research mode: design, measurement, controls, causal assumptions, sampling, analysis, uncertainty, and replication. | Candidate, plan, and review. | `MethodReview` and protocol requirements. |
| Ethics, safety & governance | Check human/animal/environmental/biosafety, privacy, dual-use, data rights, and required external approvals. | Plan, candidate, and method review. | `GovernanceReview`; it can block promotion. |
| Ranking | Compare viable candidates with pairwise, randomized-order tournament reviews and researcher-controlled weights. | Reviewed candidates and rubric. | `Scorecard`, comparisons, confidence, and sensitivity analysis. |
| Evolution | Refine a high-value candidate or deliberately create a dissimilar alternative; cannot bypass reflection. | Ranked candidates and critiques. | `EvolvedCandidate` linked to its parents. |
| Proximity & diversity | Maintain a graph of candidate mechanism, predicted outcome, and evidence overlap to find duplicates, gaps, and minority views. | Candidate lineage and reviews. | `ResearchLandscape`. |
| Meta-review | Reconcile the evidence baseline, review disagreements, ranking, unresolved assumptions, and decision conditions. It cannot rewrite earlier evidence or hide a blocking result. | All accepted scientific artefacts. | `DecisionMemo`. |
| Scientific report architect | Transform validated scientific artefacts into a reader-oriented report model. It may summarize but cannot introduce new scientific claims. | Display-safe artefacts, citation anchors, and report profile. | `ScientificReport` block tree. |
| Citation & reader-quality auditor | Check claim coverage, citation resolution, internal-ID leakage, terminology, repetition, navigation, and visual/export integrity. | Draft report plus evidence and display-name registries. | `ReportQualityAudit`; a blocking result prevents final export. |

The Supervisor has explicit budgets for candidates, comparisons, searches, and
iterations. It stops when the budget is spent, convergence is reached, a safety
gate blocks the work, a task cannot be recovered, or the researcher stops the
session. The selected approval profile governs ordinary artifact promotion;
mandatory input, safety, ethics, privacy, and institutional gates always remain
human-controlled. No agent may autonomously run experiments, access restricted
data, submit protocols, or take real-world research actions.

### Implemented architecture and remaining gaps

The repository now enforces the core control architecture:

- `CoScientistWorkflow` is the code-level scheduling and promotion authority;
  the live root agent is not a `SequentialAgent`.
- `App(name="app", ...)` is resumable, the generated FastAPI runtime publishes
  the root and every narrow specialist over A2A, and SQLite persists ADK
  sessions, A2A tasks, research sessions, decisions, and audit events.
- Only Evidence Discovery has `google_search`; Source Verification has
  `load_web_page`; other specialists cannot silently search.
- Specialist completions carry validated schema names and JSON payloads rather
  than relying only on prose. Legacy artifacts migrate as explicitly
  unverified Markdown payloads.
- Missing peptide sequence and single-cell dataset requests create blocking
  `InputRequirement` records. Literature-only continuation requires an
  explicit researcher decision and is printed throughout the dossier.
- The deterministic provider remains an offline demonstration and CI fixture.
  It never marks evidence verified.

The remaining production-quality gaps are behavioral rather than missing
runtime plumbing: live Gemini outputs need repeated evaluation against the new schemas;
grounding metadata must be normalized into claim-level evidence at scale;
evolution requires multiple live re-review cycles to establish convergence;
failure injection and load testing remain incomplete; and no qualified
domain-expert review has approved protocol quality.

`result.pdf` adds a concrete normalization failure mode. A specialist's raw
domain-specific response and its canonical typed artifact can diverge. The
current fallback can replace a malformed live result with generic deterministic
content while retaining the richer raw response in `artifact.content`. The
report then prints both. This creates two apparent truths and allows a generic
artifact to advance.

Replace that behavior with the following boundary:

1. Validate the specialist result against the expected schema and semantic
   completeness rules.
2. If only serialization is malformed, run one bounded schema-repair pass that
   may reorganize the supplied content but may not add claims.
3. Revalidate identifiers, evidence references, domain specificity, candidate
   distinctness, and required fields.
4. If repair fails, store the raw response as a quarantined diagnostic,
   transition the task to `input-required` or `failed-validation`, and retry or
   request human guidance within budget. Do not substitute a publishable
   candidate or review.
5. Only the accepted typed artifact becomes canonical input for later agents
   and the scientific report. Quarantined raw output is available to authorized
   developers, not report readers.

Add semantic validators for template leakage and cross-artifact consistency.
For example, eight candidates that share the same predictions, falsifier,
risks, and go/no-go tests fail candidate diversity even if their IDs differ.
A ranking cannot be accepted when its reviews say every candidate has
insufficient evidence, and a recommendation cannot be accepted when its
candidate cannot be resolved to the exact version that was reviewed.

The project retains the requested Vertex AI global endpoint,
`gemini-3.1-pro-preview`, and high thinking depth for scientific reasoning.
These settings live in one validated configuration module rather than being
duplicated across agents. The Supervisor itself should not need a reasoning
model. Because the model is a preview identifier, startup performs a clear
compatibility check and permits an explicit environment override without
silently falling back to a weaker model.

### A2A protocol contract for asynchronous collaboration

Agents communicate through Agent2Agent (A2A) rather than by relying on one
model's unstructured context window. Each purpose-built agent publishes an
Agent Card declaring its skills, accepted artefact schemas, supported content
types, authentication requirements, streaming and push-notification
capabilities, and safety boundaries. The Supervisor validates the Agent Card
before delegating work.

1. The Supervisor opens a single A2A `contextId` for each research session and
   sends each unit of work as a non-blocking A2A `Task`. Every request includes
   a task type, immutable input artefact references, rubric version, budget,
   deadline, correlation ID, and idempotency key.
2. A receiving agent immediately acknowledges the task, then reports the A2A
   lifecycle state: `submitted`, `working`, `input-required`, `auth-required`,
   `completed`, `failed`, `canceled`, or `rejected`. It returns substantive
   results as typed A2A Artifacts, not as informal status messages.
3. The Supervisor uses streaming status/artifact events when available, and
   otherwise polls `GetTask` with bounded exponential backoff. For long-lived
   sessions it registers authenticated push notifications. Status updates are
   advisory; the durable task record and artefact store are the audit source.
4. Agents may send follow-up tasks only to the Supervisor. The Supervisor
   validates dependencies and schedules them, preventing circular delegation,
   duplicate work, unbounded fan-out, and an evolution agent bypassing review.
5. `input-required` pauses the dependent branch and is surfaced in the TUI with
   the exact requested decision. `failed`, `rejected`, and timed-out tasks
   retain their error artefacts; the Supervisor may retry only idempotent work
   within budget or route it to the researcher.
6. A completed artefact is accepted into the research ledger only after schema,
   provenance, safety, and dependency checks. Human approval remains a separate
   state and cannot be forged by an agent completion event.

This protocol enables evidence search, candidate generation, methods review,
and safety review to run concurrently where their inputs permit, while ranking
and evolution wait for their required reviews. It preserves asynchronous status
visibility without allowing a partial or failed task to be mistaken for a
scientific conclusion.

### Public artefact and state contracts

All service boundaries use Pydantic models with `schema_version`, stable IDs,
timestamps, producing agent/model/prompt/tool versions, input artefact IDs, and
content checksums. A2A artefacts use `application/json` and declare their schema
name and version. The minimum contracts are:

| Contract | Required content |
| --- | --- |
| `ResearchSession` | Question, declared mode, constraints, owner, budgets, status, current gate, and A2A `contextId`. |
| `ResearchPlan` | Claim type, constructs, population/system, assumptions, safety/ethics triage, success and stopping criteria. |
| `TaskEnvelope` | Task type, immutable input references, deadline, budget, correlation/trace ID, idempotency key, and expected output schema. |
| `ResearchDirection` / `KnowledgeBaseManifest` | Direction map, verified claim/source references, coverage, contradictions, gaps, search cut-off, and versioned evidence baseline. |
| `Candidate` | Version and parent IDs, hypothesis/model, rationale, discriminating predictions, alternatives, falsifier, feasibility, and evidence references. |
| `SourceRecord` / `EvidenceClaim` / `EvidencePacket` | Source identity and type, retrieval and verification status, exact claim/location, supports/contradicts/neutral relation, quality limits, and retraction/correction status. |
| `EvidenceReview`, `MethodReview`, `GovernanceReview` | Findings, severity, unresolved requirements, blocking status, and reviewer independence/provenance. |
| `PairwiseComparison` / `Scorecard` | Blinded order, rubric version, criterion scores, evidence references, uncertainty, disagreement, and sensitivity. |
| `ResearchLandscape` | Candidate clusters, lineage, overlap, contradictions, gaps, and protected minority hypotheses. |
| `HumanDecision` | Gate and artefact version, accept/revise/stop action, feedback, actor, timestamp, and optimistic-lock version. |
| `DecisionMemo` | Recommendation, uncertainty, disqualifying flaws, next test, evidence that would change the decision, and required approvals. |
| `DisplayNameRegistry` | Stable reader-facing candidate number, short scientific title, version/lineage label, and internal-object mapping. Internal IDs are excluded from normal views. |
| `ScientificReport` / `ReportQualityAudit` | Ordered semantic report blocks and the results of citation, consistency, leakage, navigation, and layout checks. |

Artefacts are immutable. Revisions create a new version linked to its parent.
Only the Supervisor can append an accepted version to the canonical session.
State transitions are validated against a transition table: a task completion
may create a draft; only a matching, current `HumanDecision` can promote it.
Duplicate idempotency keys return the original task, stale approvals fail, and
cancel/timeout/failure records remain auditable.

## Stage-by-stage requirements

| Stage | Required behaviour | Required artefact and acceptance gate |
| --- | --- | --- |
| Goal manager | Parse the goal into question, research mode, constructs, population/system, constraints, success criteria, resources, time horizon, and ethics/safety/data-governance needs. Detect missing sequence-, dataset-, or measurement-dependent inputs before making empirical claims. | `ResearchPlan` plus blocking `InputRequirement` records. The researcher supplies the input, explicitly chooses a permitted literature-only fallback, or stops. |
| Evidence baseline | Decompose the question into research directions, discover balanced source leads, verify original sources and atomic claims, curate conflicts/gaps, and freeze a versioned baseline. | `ResearchDirection`, `EvidenceClaim`, `EvidenceGap`, and `KnowledgeBaseManifest` records. Zero verified claims blocks normal generation; an explicitly accepted exploratory fallback cannot yield a scientific recommendation. |
| Generation | Use four independent strategies: evidence-first, mechanism-first, analogy/transfer, and competing-explanation generation. Generate domain-specific diversity rather than paraphrases. | Eight `Candidate` records by default, each with a reader-facing title, claim, mechanism/model, predictions, alternatives, dependencies, falsifier, feasibility constraints, and evidence references. Semantic template-leakage and duplicate checks must pass. |
| Reflection | Give every candidate independent evidence/correctness, novelty, methods/feasibility, impact, and safety/governance review. Preserve objections, rebuttals, disagreements, and fatal flaws. | Typed `CandidateReview` sets. Candidates with unresolved fatal flaws cannot become a final recommendation. |
| Ranking | Compare only review-eligible candidates pairwise with blinded ordering where practical. Do not use a single fluent judge as the ground truth. Use evidence reviews and domain-specific criteria. | Tournament outcomes plus a transparent scorecard: validity, evidence strength, novelty, importance, feasibility, risk, cost/time, reproducibility, and expected information gain. All-insufficient evidence blocks recommendation and instead produces an evidence-acquisition priority list. |
| Evolution | Improve only candidates that survive reflection. Combine compatible mechanisms, simplify tests, find alternative operationalizations, and deliberately generate a dissimilar candidate to avoid local optima. Each evolved candidate returns to reflection. | A versioned `EvolvedCandidate` that names exactly what changed, why, the new discriminating prediction, and the tests required before promotion. |
| Proximity | Cluster candidates by mechanism, data requirement, intervention, and predicted outcome—not wording alone. Use the clusters to detect duplicates, blind spots, and underexplored but plausible directions. | A `ResearchLandscape` with clusters, relationships, contradictions, coverage gaps, and a diversity recommendation. The researcher may protect a minority hypothesis from premature elimination. |
| Meta-review | Synthesize recurrent reviewer concerns, audit evidence lineage, revisit the highest-impact assumptions, and monitor safety/ethics drift. | A `DecisionMemo`: strongest directions, disqualifying flaws, unresolved uncertainty, minimum next experiment/analysis, evidence that would change the recommendation, and go / conditional go / no-go. |
| Report | Preserve uncertainty while translating accepted typed artifacts into a coherent document for a scientific reader. Never concatenate raw agent output or JSON into the default report. | `ScientificReport` plus a passing `ReportQualityAudit`. The machine audit ledger is a separate export. |

## Reader-oriented report design

### Separate the scientific report from the audit record

The current `compile_dossier()` deliberately appends every artifact, its raw
content, metadata, checksum, and validated JSON. That is useful for debugging
but unsuitable as a default report. Replace the single mixed document with
three explicit products:

- **Research brief:** a 10–20 page decision-focused report for fast review.
- **Scientific dossier (default):** a curated 25–45 page report with the full
  evidence landscape, candidate portfolio, review synthesis, shortlist,
  validation protocols, uncertainty, and references.
- **Machine audit archive:** JSON/JSONL plus a manifest containing internal IDs,
  checksums, tasks, decisions, schema versions, model/tool provenance, and raw
  quarantined diagnostics where authorized. This is never embedded in the
  normal PDF or DOCX.

Length is a budget, not a target. A section may grow when the question demands
it, but duplicated raw artifacts never count as scientific depth.

### Default scientific-dossier organization

1. Cover: query-derived title, full question, date, research mode, evidence
   cut-off, report status, and integrity notice.
2. Contents and “How to read this report.”
3. Executive synthesis: answer, confidence, evidence sufficiency, recommended
   directions, critical uncertainties, and immediate next decision.
4. Scope and evaluation criteria.
5. Knowledge Base and evidence landscape.
6. Main research directions.
7. Candidate portfolio.
8. Independent review synthesis and disagreements.
9. Comparative ranking and shortlist.
10. Evolved candidates and what changed.
11. Proximity map, coverage gaps, and minority hypotheses.
12. Final recommendation or explicit insufficient-evidence outcome.
13. Discriminating validation protocol with go/no-go conditions.
14. Limitations, safety, ethics, governance, and reproducibility.
15. Numbered linked references.
16. Optional human-readable methods appendix. Machine provenance is a separate
    download.

The executive section must not name a candidate only by an opaque identifier.
It should say, for example, “Candidate 2 — SHP2 co-inhibition for adaptive MAPK
rebound,” followed by a one-sentence rationale and evidence status.

### Candidate and review presentation

Create a stable `DisplayNameRegistry` as soon as the initial population is
accepted. Each candidate receives:

- a stable ordinal for the session;
- a short, domain-specific scientific title;
- a one-sentence claim;
- an optional version label such as “revised after methods review”; and
- an internal-ID mapping stored only in the machine audit archive.

The report uses the title everywhere. Evolved lineage is written in reader
language—“Derived from Candidate 2; added a resistant-organoid validation
arm”—rather than as an ID chain. Review and ranking tables join through the
registry before rendering. Any unresolved lookup is a compilation error, not
permission to print the internal ID.

Each candidate card contains, in this order:

1. Claim and mechanism.
2. Why it is plausible.
3. Evidence for, evidence against, and evidence gaps.
4. Discriminating predictions and competing explanations.
5. Falsifier.
6. Minimal validation design and go/no-go threshold.
7. Feasibility, safety, and translation risks.
8. Review consensus, disagreement, confidence, and current disposition.

The ranking section first uses a compact table with title, mechanism,
evidence-strength band, review disposition, score interval, fatal-flaw status,
and shortlist status. Elo and pairwise details belong in a collapsible web view
or methods appendix, not the main recommendation narrative.

### Deterministic report compilation

The report architect may help select and summarize already accepted content,
but the final compiler must be deterministic:

1. Build a schema-versioned `ScientificReport` block tree from accepted typed
   artifacts, the display-name registry, and citation anchors.
2. Resolve every cross-reference and material-claim citation.
3. Run consistency checks across evidence, reviews, ranking, evolution, and
   meta-review.
4. Run an ID-leakage and serialization-leakage linter.
5. Render the same semantic blocks into the web view, Markdown, DOCX, and PDF.
6. Run format-specific layout and accessibility checks before publishing.

Do not feed the Markdown dossier line by line into escaped PDF paragraphs. The
current renderer consequently prints `**bold**`, backticks, pipe-table syntax,
`<details>` tags, and JSON literally. Use native report components: paragraphs
with inline emphasis, real tables, callout boxes, figures, numbered lists,
cross-references, citations, and `KeepTogether`/page-break rules.

The PDF requires a clickable contents page, outline bookmarks, page numbers,
running title, section headers, working hyperlinks, readable body type, table
header repetition, overflow protection, and Unicode font coverage. The DOCX
requires equivalent heading styles, real tables, references, editable content,
and compatibility with Google Docs. The web report uses the same block model
with expandable technical methods, never raw JSON by default.

### Report-quality gate

The Citation & Reader-Quality Auditor blocks export when any of the following
is true:

- a material factual claim lacks a resolved citation or uncertainty label;
- an internal `candidate_`, `artifact_`, `task_`, `src_`, `claim_`, checksum,
  schema field, raw JSON object, Markdown pipe table, or HTML tag appears
  outside an explicitly requested developer audit;
- a recommended candidate is missing, has insufficient evidence, or has an
  unresolved fatal flaw;
- candidate names, versions, ranks, or dispositions disagree across sections;
- the Knowledge Base is absent or is only a list of URLs;
- content is clipped, silently truncated, unreadably small, or lacks a
  navigable reference target.

## Evidence, search, and data standards

Evidence handling is deliberately split into discovery and verification:

1. A dedicated Evidence Discovery service uses Gemini's `google_search` tool to
   find leads, including negative findings, replications, retractions,
   conflicting results, and publication-bias signals. Other scientific agents
   request evidence through the Supervisor instead of searching independently.
2. A separate verification/normalization step opens the original primary source
   where access and policy permit, resolves DOI/PMID or dataset identifiers,
   checks title/authors/date and correction/retraction status, and maps each
   material claim to an exact supporting or contradicting location. Snippets
   remain `discovered_unverified`.
3. Tool-using search and typed normalization are separate agent steps. ADK
   structured `output_schema` disables tool calling/delegation, and mixing
   Google Search with custom function tools can disable automatic function
   calling. Therefore the search agent returns grounded discovery material; a
   schema-only normalizer or deterministic boundary validator produces the
   typed `EvidencePacket`.
4. Prefer primary literature, official datasets/standards, registered
   protocols, and authoritative repositories. Clearly label reviews, preprints,
   news, snippets, and non-peer-reviewed material. Citation metadata and
   grounding links are persisted, not reconstructed from prose.
5. Keep data provenance separately from literature provenance: origin, consent
   or authorization, license, transformations, quality checks, privacy risks,
   and version/checksum where applicable.
6. Never invent citations, measurements, experimental results, effect sizes, or
   tool outputs. If the source cannot be inspected, report the limitation and
   prevent it from satisfying an evidence gate.

Retrieved content is untrusted input. The verifier rejects unsupported schemes
and private-network targets, limits redirects/size/type, and treats instructions
inside sources as data rather than agent commands. Restricted or paywalled
material is never bypassed.

## Rigor, reproducibility, and governance gates

Before a candidate can receive a conditional go, the relevant gates must pass:

- Construct validity and measurement plan; calibration and uncertainty where
  relevant.
- Design validity: comparator, confounders, bias controls, and causal limits.
- Statistical or analytical plan: estimand, sample/power or precision rationale,
  multiple-testing policy, missing-data handling, robustness checks, and
  pre-specified exclusion/stopping rules where applicable.
- Reproducibility: protocol version, software/environment or instrument setup,
  data/code/material availability, and independent replication or validation.
- Ethics and safety: human/animal/environmental/biosafety/privacy review,
  applicable approvals, dual-use screening, and escalation to qualified
  institutional review when needed.
- Reporting: uncertainty intervals or equivalent, negative results, deviations
  from plan, limitations, and conflicts of interest.

## Persistence, security, and operations

The development profile uses a SQLite-backed ADK session service and research
ledger plus a project-local filesystem artefact store. The append-only event
log is the recovery source for sessions, A2A tasks, human decisions, candidate
lineage, and evidence provenance. Schema migrations, checksums, transactions,
optimistic locking, and restart recovery are required. ADK in-memory sessions
remain test-only. A production database/object store is selected only with the
deployment target and its backup, retention, residency, and deletion policy.

Local A2A services bind to loopback and may use development credentials.
Production requires HTTPS, service identity/OIDC, Agent Card allowlisting,
least-privilege tool credentials, authenticated push callbacks, timestamped
replay protection, and secret rotation. Secrets, credentials, private source
content, and chain-of-thought are never written to artefacts or logs; store
concise decision rationales instead. Each session has rate, token, search,
candidate, iteration, wall-clock, and storage budgets with timeouts,
backpressure, bounded concurrency, and circuit breakers.

Structured logs and OpenTelemetry traces correlate
`research_session_id/contextId`, A2A `taskId`, correlation/trace ID, agent,
artefact/candidate ID, schema/model/prompt/tool versions, task state, retry,
latency, token/cost estimate, and verified-source counts. Content capture is
off/redacted by default. Development starts with local logs and traces; Cloud
Trace, dashboards, and alerts for stuck tasks, failure rate, verification rate,
budget exhaustion, and safety blocks are enabled only as part of an approved
deployment/observability tier.

## Implementation roadmap

### Completed foundation

- The agents-cli scaffold, generated FastAPI/A2A transport, resumable ADK app,
  global Gemini configuration, dependency lock, local SQLite services, and
  deployed Cloud Run surface exist.
- The deterministic Supervisor, typed contracts, checksums, legacy migration,
  six method adapters, idempotent task records, optimistic locking, and
  restart-recovery tests exist.
- Input-sufficiency gates, literature-only fallback, eight-candidate generation,
  five independent review passes, three Swiss tournament rounds plus a top-four
  round robin, shortlist evolution, proximity analysis, and dossier export are
  implemented, but `result.pdf` proves their live semantic gates and report
  presentation need correction.
- The web, TUI, and CLI expose `auto`, `milestone`, `stage`, and `artifact`
  profiles; legacy `--approval-mode human` maps to `stage`.

### Phase 0 — freeze the regression fixture

- Preserve `result.pdf` as the failing format/grounding benchmark and
  `sample2.pdf` as a presentation/process benchmark.
- Create deterministic fixtures matching the observed failure: nine unverified
  discovery leads, zero verified claims, a malformed but domain-specific raw
  candidate response, a generic fallback population, all reviews marked
  insufficient, and a tournament that attempts to recommend.
- Add snapshot diagnostics for page structure, ID leakage, raw serialization,
  citation links, and report sections. Do not assert exact LLM wording.

### Phase 1 — enforce evidence and normalization gates

- Add `ResearchDirection`, `KnowledgeBaseManifest`, `EvidenceGap`,
  `EvidenceRequest`, and evidence-baseline versioning to `coscientist/models.py`.
- Add the visible Evidence stage and transition rules to
  `coscientist/orchestration.py`. A zero-claim verification result must end in
  evidence-required, exploratory-fallback, or stopped—not Generate.
- Replace silent generic fallback in `coscientist/parity.py` and the live
  provider boundary with bounded schema repair plus `failed-validation`.
- Implement template-leakage, candidate-distinctness, evidence-link, review /
  ranking consistency, and exact-version validators.
- Ensure asynchronous A2A evidence requests can be submitted during review and
  evolution, with budgeted deltas to the frozen baseline.

**Phase 1 exit:** the `result.pdf` evidence state cannot advance to normal
Generation; no malformed live response can become an accepted generic
substitute; and an all-insufficient review set cannot produce a recommendation.

### Phase 2 — build the Knowledge Base

- Expand Evidence Discovery queries using the research-direction coverage
  matrix and preserve Google grounding metadata.
- Normalize original sources, stable identifiers, exact locations,
  corrections/retractions, source type, directness, and limitations.
- Implement deduplication, contradiction detection, coverage scoring, and
  mode-specific evidence thresholds.
- Add a Knowledge Curator specialist and deterministic manifest builder.
- Add filtered hybrid retrieval and stage-specific evidence packets. Measure
  whether later candidates and reviews actually cite the baseline rather than
  merely receiving it.
- Expose the Knowledge Base in current-stage, historical-stage, and dossier
  views with readable source titles and numbered citations.

**Phase 2 exit:** every normal-mode run has a versioned evidence baseline before
candidate generation; every candidate rationale resolves to verified claims or
is visibly marked as an evidence gap; searches cover counterevidence and
corrections, not only supporting leads.

### Phase 3 — replace the dossier compiler

- Add `DisplayNameRegistry`, `ScientificReport`, and `ReportQualityAudit`.
- Refactor `coscientist/dossier.py` so the default report consumes only accepted
  typed artifacts and never appends raw content or JSON.
- Reuse and extend `coscientist/presentation.py` as the semantic report block
  source so web, Markdown, DOCX, and PDF communicate the same science.
- Add the brief, scientific-dossier, and machine-audit export profiles.
- Implement native PDF/DOCX headings, tables, citation links, page templates,
  bookmarks/contents, cross-references, Unicode, and overflow handling.
- Add deterministic ID/JSON/Markdown/HTML leakage linting and cross-section
  scientific consistency checks.

**Phase 3 exit:** the default PDF contains zero internal IDs or raw
serialization, a readable Knowledge Base, stable scientific candidate titles,
working references and navigation, and no silently clipped content.

### Phase 4 — improve scientific depth

- Make the four generation strategies operate independently on different
  evidence packets and require mechanism-level diversity.
- Have each reviewer assemble or request independent counterevidence rather
  than reuse the generator's rationale.
- Rank with evidence-strength bands and uncertainty intervals in addition to
  bounded Elo; expose sensitivity to criterion weights.
- Run repeated evolution and re-review cycles until the top ranking is stable
  for two rounds with less than 5% score movement, or three evolution rounds
  are exhausted.
- Add a Report Architect that may summarize accepted artifacts but cannot add
  facts, and a separate Citation & Reader-Quality Auditor with blocking
  authority.

### Phase 5 — evaluate and roll out

- Use `pytest` for deterministic code, schema, ledger, state-machine, citation
  validator, report compiler, A2A contract, failure-injection, and security
  tests. Use `agents-cli eval` for model behavior and trace quality; do not
  encode probabilistic output wording as brittle unit tests.
- Expand the versioned evaluation set with both sample-derived missing-input
  cases, complete synthetic inputs, at least two cases per research mode, and
  adversarial false-citation, retraction, confounding, leakage, unsafe,
  privacy, prompt-injection, insufficient-evidence, and search-outage cases.
- Run the quality flywheel: dataset → generate → grade → inspect failures → fix
  → compare for at least five iterations. Include a report-reader rubric and
  compare the new report against `result.pdf` on the same query.
- Run desktop/mobile web checks and PDF/DOCX visual regression checks on cover,
  contents, Knowledge Base, candidate, ranking, validation, references, and
  long-table pages.
- Deploy through the existing approved Cloud Run workflow only after
  deterministic, behavioral, and export gates pass; retain rollback to the
  prior revision and verify public persistence/export behavior.

**Exit thresholds:** deterministic tests pass; 100% of cited claim and source
references resolve; at least 95% of sampled material claims are supported or
correctly labeled uncertain; overall multi-turn task success is at least 0.85
with no research mode below 0.75; trajectory and tool-use quality are each at
least 0.80; groundedness is at least 0.90; all safety and mandatory human-gate
cases pass. A qualified domain reviewer must approve protocol quality and a
scientific reader must approve report usability before production promotion.

## Acceptance criteria

- A researcher can select or confirm a research mode for any scientific domain
  represented by the six adapters, without the system assuming a biomedical
  experiment.
- The workflow cannot enter normal candidate generation with zero verified
  claims. An explicitly approved exploratory fallback cannot end in a ranked
  scientific recommendation.
- Every run freezes a versioned Knowledge Base before generation and reports
  its search cut-off, coverage, contradictions, gaps, and verification status.
- At least 95% of sampled material claims in the Knowledge Base, candidate
  rationales, reviews, and recommendation resolve to an exact verified source
  location; the remainder are explicitly labeled uncertain or proposed.
- Every candidate uses verified knowledge by reference. Cross-stage
  evidence-reuse coverage is reported, and receiving an evidence packet without
  citing it does not count as grounding.
- Every promoted candidate has a falsifier or an explicit reason why a
  hypothesis framework is inappropriate.
- Every material factual claim has a provenance status; unsupported claims are
  visibly marked as proposals or unknowns.
- Ranking is traceable to independent reviews and a researcher-controlled
  rubric, with disagreements and sensitivity exposed.
- A review set in which all candidates are insufficiently evidenced cannot
  produce a scientific recommendation. No unresolved fatal flaw can disappear
  between Reflection, Rank, Evolution, and Meta-review.
- Every final brief includes method, limitations, uncertainty, reproducibility,
  ethics/safety status, and the human approvals required for action.
- The default report contains a cover, contents, executive synthesis, Knowledge
  Base, research directions, distinct candidate cards, review synthesis,
  comparison/shortlist, validation protocol, limitations/governance, and
  numbered linked references in that order.
- The default PDF, DOCX, Markdown, and web report contain zero raw internal
  IDs, checksums, schema names, raw JSON, unrendered Markdown tables, or HTML
  tags. Internal provenance remains downloadable as a separate machine audit.
- Candidate titles and numbering remain stable across Generate, Reflect, Rank,
  Evolve, Proximity, Meta-review, and all export formats. Unresolvable display
  names block compilation.
- PDF and DOCX exports have working navigation and references, readable type,
  real tables, Unicode coverage, repeated table headers, and no clipping or
  silent truncation at desktop-print and Google Docs review.
- A2A disconnects, duplicate requests, task cancellation, process restart, and
  stale human decisions cannot corrupt or incorrectly advance the session.
- Only the Evidence Discovery service has Google Search access; a material claim
  cannot satisfy an evidence gate until the original source is verified or the
  limitation is explicitly accepted by the researcher.
- Logs and traces correlate the complete task lineage without recording secrets,
  private source content, or hidden model reasoning.
- The system remains a scientific decision-support tool: qualified humans retain
  responsibility for literature validation, approvals, research conduct, and
  conclusions.

Track these custom evaluation metrics in addition to the existing task,
groundedness, safety, and tool-use metrics:

- `knowledge_base_coverage`
- `claim_location_resolution`
- `counterevidence_search_coverage`
- `cross_stage_evidence_reuse`
- `candidate_domain_specificity`
- `candidate_mechanism_diversity`
- `review_ranking_consistency`
- `fatal_flaw_consistency`
- `internal_id_leakage`
- `raw_serialization_leakage`
- `report_section_completeness`
- `citation_link_integrity`
- `report_navigation_and_layout`
- `reader_decision_comprehension`
