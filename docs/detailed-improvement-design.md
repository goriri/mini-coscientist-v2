# Co-Scientist: Section-to-Section Detailed Implementation Design

## 1. Executive Overview & Architecture Mapping

This document is the section-to-section detailed technical design for implementing the improvements defined in [`docs/improvement-plan.md`](file:///usr/local/google/home/jush/mini-coscientist-v2/docs/improvement-plan.md) across the `mini-coscientist-v2` codebase.

The goal is to transition `mini-coscientist-v2` from a structural demonstration into an auditable, domain-general scientific research assistant that enforces rigor-first input gates, verifiable knowledge grounding, bounded schema normalization without generic fallbacks, multi-turn evolutionary convergence, and reader-oriented scientific report compilation.

### 1.1 Architecture & Roadmap Phase Mapping Table

| Subsystem | Primary Implementation Targets | Implementation Plan Phases | Addressed Plan Sections |
| --- | --- | --- | --- |
| **1. Evidence & Knowledge Base Substrate** | [`coscientist/models.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/models.py), [`coscientist/evidence.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/evidence.py), new `coscientist/curation.py` | **Phase 1** (Evidence Gates)<br>**Phase 2** (Knowledge Base) | Section 6 (Knowledge base as evidence substrate)<br>Section 11 (Evidence, search, and data standards) |
| **2. Normalization, Schema Repair & Quarantining** | New `coscientist/normalization.py`, [`coscientist/parity.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/parity.py), [`app/research_api.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/app/research_api.py) | **Phase 1** (Normalization Gates) | Section 3 (Official-report gap assessment)<br>Section 4 (`result.pdf` vs `sample2.pdf` audit) |
| **3. Reader-Oriented Report Architecture** | [`coscientist/dossier.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/dossier.py), [`coscientist/presentation.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/presentation.py), new `coscientist/registry.py` | **Phase 3** (Dossier Compiler Replacement) | Section 4 (Content-organization findings)<br>Section 10 (Reader-oriented report design) |
| **4. Scientific Depth & Evolution Convergence** | [`coscientist/orchestration.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/orchestration.py), [`coscientist/agents.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/agents.py), [`coscientist/collaboration.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/collaboration.py) | **Phase 4** (Scientific Depth) | Section 7 (Domain-general scientific method)<br>Section 8 (Target architecture)<br>Section 9 (Stage-by-stage requirements)<br>Section 12 (Rigor and governance gates) |
| **5. Evaluation Suite & Quality Flywheel** | `tests/unit/`, `tests/integration/`, new `evals/datasets/`, `agents-cli eval` | **Phase 0** (Regression Fixture)<br>**Phase 5** (Evaluate & Roll Out) | Section 5 (Definition of success)<br>Section 13 (Persistence, security, operations)<br>Section 15 (Acceptance criteria) |

---

## 2. Subsystem 1: Evidence & Knowledge Base Substrate

### 2.1 Extended Typed Contracts ([`coscientist/models.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/models.py))

To elevate the Knowledge Base from an unverified list of discovery URLs into an immutable, versioned evidence baseline, the following Pydantic models will be added or updated in [`coscientist/models.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/models.py):

```python
class ResearchDirection(Contract):
    id: str = Field(default_factory=lambda: new_id("dir"))
    title: str
    scope: str
    mechanism_or_concept: str
    outcome: str
    competing_explanations: list[str] = Field(default_factory=list)
    required_data: list[str] = Field(default_factory=list)
    search_questions: list[str] = Field(default_factory=list)

class EvidenceGap(Contract):
    id: str = Field(default_factory=lambda: new_id("gap"))
    direction_id: str
    description: str
    decision_impact: Literal["low", "medium", "high", "blocking"] = "medium"
    resolution_query: str

class EvidenceRequest(Contract):
    id: str = Field(default_factory=lambda: new_id("evreq"))
    requesting_stage: str
    requesting_agent: str
    claim_to_verify: str
    priority: int = Field(default=1, ge=1, le=5)
    budget_usd: float = Field(default=1.0, ge=0.0)
    status: Literal["submitted", "working", "completed", "failed", "rejected"] = "submitted"
    resulting_manifest_version: int | None = None

class CitationAnchor(Contract):
    id: str = Field(default_factory=lambda: new_id("cite"))
    claim_id: str
    human_citation_number: int
    report_location: str
    display_text: str

class KnowledgeBaseManifest(Contract):
    id: str = Field(default_factory=lambda: new_id("kb"))
    version: int = 1
    parent_version: int | None = None
    directions: list[ResearchDirection] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    coverage_matrix: dict[str, float] = Field(default_factory=dict)
    contradiction_graph: list[tuple[str, str]] = Field(default_factory=list)
    unresolved_gaps: list[EvidenceGap] = Field(default_factory=list)
    search_cutoff_date: str = Field(default_factory=utc_now)
    checksum: str = ""
```

### 2.2 Knowledge Curator Agent (`coscientist/curation.py`)

A dedicated `KnowledgeCuratorService` class will be created in `coscientist/curation.py` to consume the unverified `DiscoveryManifest` from Evidence Discovery and the verified `EvidencePacket` from Source Verification:
1. **Deduplication:** Merges identical canonical URLs and DOIs/PMIDs across discovery passes.
2. **Quality & Directness Appraisal:** Assigns source directness (`primary_experimental`, `primary_observational`, `meta_analysis`, `review`, `preprint`) and flags limitations.
3. **Contradiction Detection:** Constructs a bipartite graph linking opposing claims (`supports` vs. `contradicts` for the same atomic proposition).
4. **Baseline Freezing:** Computes the SHA-256 checksum of the serialized claims, directions, and sources, producing `KnowledgeBaseManifest` v1.

### 2.3 Immutable Versioning & Delta Requests

When downstream stages (Reflection, Evolution) encounter new mechanisms or require counterevidence:
1. The specialist emits an `EvidenceRequest` task to the Supervisor.
2. The Supervisor verifies that the request is within the session's search budget (`ResearchBudget.max_searches`).
3. Source Verification runs a bounded check on the requested claim.
4. Instead of mutating `KnowledgeBaseManifest` v1 in-place, the Supervisor calls `KnowledgeCuratorService.create_delta_manifest(parent=kb_v1, new_claims=..., new_sources=...)`, producing **`KnowledgeBaseManifest` v2**.
5. Earlier candidate generations remain traceably linked to v1; evolved candidates explicitly reference v2.

### 2.4 Mode-Specific Evidence Thresholds & Exploratory Fallback

In [`coscientist/orchestration.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/orchestration.py), the `evidence` stage acceptance gate will enforce:
- **Blocking Condition:** If `len(kb_manifest.claim_ids) == 0` or all sources have `verification_status == "discovered_unverified"`, normal transition to `generate` is blocked.
- **Exploratory Fallback:** If the researcher explicitly accepts `exploratory_evidence_accepted = True` via TUI/web prompt, the Supervisor marks the session mode as exploratory.
- **Recommendation Prohibition:** Any session operating under exploratory fallback is programmatically prohibited from outputting a ranked scientific recommendation in `DecisionMemo`; it may only recommend "Prioritized Hypotheses for Evidence Acquisition."

---

## 3. Subsystem 2: A2A Normalization, Schema Repair & Quarantining

### 3.1 Dedicated Normalization Service (`coscientist/normalization.py`)

To eliminate the two-truth problem where a malformed specialist output is silently replaced by generic fallback text while the raw response is printed in the appendix, all specialist completions will pass through `NormalizationService` in `coscientist/normalization.py`:

```
+--------------------+      Raw JSON / Markdown      +-----------------------+
|  Specialist Agent  | -----------------------------> |  NormalizationService |
+--------------------+                               +-----------------------+
                                                                 |
                                        +------------------------+------------------------+
                                        | Valid                                           | Invalid JSON
                                        v                                                 v
                             +--------------------+                            +---------------------+
                             | Semantic Validator |                            | Bounded 1-Pass      |
                             +--------------------+                            | Schema Repair       |
                                        |                                      +---------------------+
                         +--------------+--------------+                                  |
                         | Pass                        | Fail (or Repair Fail)            |
                         v                             v                                  v
                +-----------------+          +-------------------+             +--------------------+
                | Accepted Typed  |          | Quarantined Raw   |             | Re-validate Schema |
                | Artifact        |          | Diagnostic        |             +--------------------+
                +-----------------+          +-------------------+
```

### 3.2 Bounded One-Pass Schema Repair Algorithm

When a model response fails initial Pydantic JSON parsing:
1. `NormalizationService.repair_schema(raw_text, target_model)` runs a single deterministic regex/AST cleanup pass (stripping markdown fences, repairing unclosed JSON brackets, trailing commas, and encoding issues).
2. If structural JSON errors persist, it invokes a bounded, 0-temperature LLM repair prompt (`gemini-2.5-flash`) with strict instructions: **"Reorganize the provided text into valid JSON matching schema X. Do NOT add new claims, citations, or metrics not present in the original text."**
3. If the repaired JSON still fails validation, repair aborts. **No generic template fallback is ever substituted.**

### 3.3 Semantic Validators

Before an artifact is accepted into the research ledger, `NormalizationService.validate_semantics()` executes four mandatory checks:
1. **Template-Leakage Detector:** Scans strings for placeholder patterns (`TODO`, `TBD`, `[insert ...]`, default fallback strings from legacy parity models).
2. **Internal ID-Leakage Linter:** Ensures reader-facing title and rationale fields do not contain raw UUIDs matching `candidate_[0-9a-f]{16}`, `artifact_[0-9a-f]{16}`, or `src_[0-9a-f]{16}`.
3. **Candidate Distinctness (Diversity) Validator:** For `CandidatePopulation`, computes pairwise Jaccard similarity across candidate `predictions`, `falsifier`, and `go_no_go_tests`. If similarity exceeds `0.75` across any pair, the population fails diversity validation.
4. **Evidence Grounding Validator:** Verifies that every `evidence_ids` reference in a candidate or review resolves to an active `EvidenceClaim` in the current `KnowledgeBaseManifest`.

### 3.4 Quarantine & Task Lifecycle Integration

When an artifact fails repair or semantic validation:
1. The raw payload is persisted as an `Artifact` with `status = ArtifactStatus.REJECTED` and `artifact_type = "quarantined_diagnostic"`.
2. The corresponding A2A `TaskRecord` state is updated to `TaskState.INPUT_REQUIRED` (or `FAILED_VALIDATION`).
3. In [`coscientist/dossier.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/dossier.py), quarantined diagnostics are **excluded** from default reader reports and are only accessible in the developer audit archive.

---

## 4. Subsystem 3: Reader-Oriented Report Architecture

### 4.1 Display Name Registry (`coscientist/registry.py`)

To decouple human-readable report narratives from internal storage identifiers, a `DisplayNameRegistry` service will be added:

```python
class DisplayNameEntry(Contract):
    internal_id: str
    reader_ordinal: int
    scientific_title: str
    short_label: str  # e.g., "Candidate 2 — SHP2 co-inhibition"
    version_badge: str | None = None

class DisplayNameRegistry(Contract):
    session_id: str
    entries: dict[str, DisplayNameEntry] = Field(default_factory=dict)
```

- When `CandidatePopulation` is accepted, `DisplayNameRegistry.register_candidates()` assigns stable 1-indexed ordinals and extracts short scientific titles.
- During Evolution, derived candidates receive lineage labels in reader language (e.g., *"Derived from Candidate 2; added resistant-organoid validation arm"*).
- All tables in reflection, ranking, and meta-review join through `DisplayNameRegistry`. Any unresolved internal ID during compilation raises a `ReportCompilationError`.

### 4.2 Report Compiler Refactor ([`coscientist/dossier.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/dossier.py))

`compile_dossier()` will be refactored into a multi-product API:

```python
def compile_dossier(
    session: Session,
    *,
    mode: Literal["default", "brief"] = "default",
    registry: DisplayNameRegistry | None = None,
) -> str:
    """Compile the reader-oriented Scientific Dossier or Research Brief."""
    ...

def compile_audit_archive(session: Session) -> dict[str, Any]:
    """Export the complete machine-readable audit archive (JSONL ledger + manifest)."""
    ...
```

#### Product Specification
1. **Scientific Dossier (`mode="default"`, 25–45 pages):**
   - **Section Order:**
     1. Cover Page & Integrity Notice
     2. Contents & How to Read This Report
     3. Executive Synthesis
     4. Scope & Evaluation Criteria
     5. Knowledge Base & Evidence Landscape (with verified citations)
     6. Main Research Directions
     7. Candidate Portfolio (using stable scientific titles)
     8. Independent Review Synthesis & Disagreements
     9. Comparative Ranking & Shortlist (compact tables)
     10. Evolved Candidates & Lineage Summary
     11. Proximity Map, Coverage Gaps & Minority Hypotheses
     12. Final Recommendation / Disposition
     13. Discriminating Validation Protocol & Go/No-Go Conditions
     14. Limitations, Safety, Ethics, Governance & Reproducibility
     15. Numbered Linked References
   - **Exclusions:** Zero raw JSON blocks, zero checksum strings, zero `candidate_xxx` or `artifact_xxx` IDs.
2. **Research Brief (`mode="brief"`, 10–20 pages):**
   - Condensed decision summary containing Executive Synthesis, Shortlist Comparison Table, Top Recommended Candidate Card, Validation Protocol, and Governance/Safety risks.
3. **Machine Audit Archive (`compile_audit_archive`):**
   - Returns a structured dictionary containing `transcript.jsonl`, all raw artifacts (including quarantined diagnostics), full A2A task envelopes, model/tool version metadata, and cryptographic SHA-256 checksums.

### 4.3 Semantic Block Tree & Multi-Format Rendering ([`coscientist/presentation.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/presentation.py))

`presentation.py` will define an abstract `ScientificReport` block tree (`ReportHeader`, `ReportSection`, `CandidateCardBlock`, `ComparisonTableBlock`, `CalloutBlock`, `ReferenceListBlock`) that compiles deterministically into:
- **Markdown:** Clean GitHub-flavored markdown with standard pipe tables and numbered reference anchors `[1]`, `[2]`.
- **HTML / Web View:** Expandable `<details>` sections for statistical methods, interactive sortable ranking tables.
- **PDF & DOCX:** Clean typography, running headers/footers, page numbers ("Page X of Y"), clickable table of contents, overflow wrapping for tables, and Unicode symbol support.

### 4.4 Citation & Reader-Quality Auditor (`ReportQualityAudit`)

A final blocking gate before report publication:

```python
class ReportQualityAudit(Contract):
    passed: bool
    unresolved_citations: list[str] = Field(default_factory=list)
    id_leakage_violations: list[str] = Field(default_factory=list)
    json_leakage_violations: list[str] = Field(default_factory=list)
    section_completeness_errors: list[str] = Field(default_factory=list)
```

If `ReportQualityAudit.passed` is `False`, the TUI/Web UI prevents PDF/DOCX download and displays exact compilation violations for automated or developer correction.

---

## 5. Subsystem 4: Scientific Depth & Evolutionary Convergence

### 5.1 Four Independent Candidate Generation Strategies

In [`coscientist/orchestration.py`](file:///usr/local/google/home/jush/mini-coscientist-v2/coscientist/orchestration.py), the `generate` stage will invoke four parallel generation tasks, each with distinct prompt directives and partitioned evidence packets:
1. `evidence_first`: Derives candidates by bridging verified supporting literature with unexplained experimental anomalies.
2. `mechanism_first`: Constructs bottom-up causal models or mathematical formulations of the target phenomenon.
3. `analogy_transfer`: Translocates established control mechanisms or algorithms from adjacent scientific domains.
4. `competing_explanation`: Deliberately generates rival hypotheses that challenge the prevailing consensus in the Knowledge Base.

### 5.2 Independent Counterevidence in Reflection

In the `reflect` stage, the Reflection specialist will:
- Receive the candidate card **without** the generator's internal reasoning summary.
- Query the `KnowledgeBaseManifest` specifically for claims where `relation == "contradicts"` or `limitations` are present.
- If counterevidence is missing from the baseline, submit an `EvidenceRequest` for targeted counterevidence search before rendering its `CandidateReview`.

### 5.3 Multi-Criteria Ranking & Sensitivity Analysis

In the `rank` stage, `Scorecard` evaluations will record:
- **Evidence-Strength Bands:** `strong_verified`, `moderate_verified`, `preliminary`, `insufficient_evidence`.
- **Uncertainty Intervals:** Score confidence intervals `[low, high]` for each criterion (validity, novelty, feasibility, impact, reproducibility).
- **Weight Sensitivity:** Computes rank perturbation under 3 rubric weighting profiles (`rigor_first`, `novelty_seeking`, `rapid_feasibility`) and highlights candidates whose shortlist inclusion is sensitive to weighting.

### 5.4 Automated Multi-Turn Evolution & Re-review Convergence Loop

To ensure evolutionary refinement stabilizes before final meta-review, `CoScientistWorkflow.advance()` will implement an automatic convergence sub-loop during the `evolve` stage:

```python
async def execute_evolution_loop(self) -> EvolutionCycle:
    cycle = EvolutionCycle()
    while len(cycle.records) < self.session.budget.max_evolution_rounds:
        # 1. Evolve shortlisted candidates
        evolved_candidates = await self._run_evolution(cycle)
        
        # 2. Independent Re-review of evolved candidates
        rereviews = await self._run_rereviews(evolved_candidates)
        cycle.rereviews.extend(rereviews)
        
        # 3. Re-run Swiss Tournament ranking on active + evolved pool
        new_ranking = await self._run_tournament_ranking(rereviews)
        cycle.ranking_history.append(new_ranking)
        
        # 4. Check convergence across last 2 rounds
        if len(cycle.ranking_history) >= 2:
            prev_rank = cycle.ranking_history[-2]
            curr_rank = cycle.ranking_history[-1]
            movement = calculate_score_movement(prev_rank, curr_rank)
            if movement < 0.05 and prev_rank.shortlist_ids == curr_rank.shortlist_ids:
                cycle.converged = True
                cycle.stop_reason = "ranking_converged_below_5_percent_movement"
                break
                
    if not cycle.converged:
        cycle.stop_reason = "max_evolution_rounds_exhausted"
    return cycle
```

### 5.5 Proximity Analysis & Minority Hypothesis Protection

During `proximity`, `ResearchLandscape` clusters candidates by shared causal mechanism and operational outcome:
- **Duplicate Detection:** Flags candidate pairs with `evidence_overlap > 0.80` and identical interventions.
- **Minority Hypothesis Protection:** If a candidate has high `novelty` but lower `feasibility` and represents a unique mechanism cluster, it is tagged in `protected_minority_ids` to prevent premature elimination during meta-review.

---

## 6. Subsystem 5: Evaluation Suite, Quality Flywheel & Document Delivery

### 6.1 Test Matrix & Verification Strategy

1. **Deterministic Code & Schema Tests (`tests/unit/`):**
   - Verify Pydantic validation, schema repair bounds, ID-leakage linting, checksum calculation, A2A task idempotency, and sqlite state-machine transitions.
2. **Behavioral Trace & Report Evals (`agents-cli eval`):**
   - Execute multi-turn trajectories across 12 canonical benchmark scenarios spanning all 6 research modes (`experimental`, `observational`, `computational`, `theory_simulation`, `systematic_review`, `measurement_field`).

### 6.2 Custom Evaluation Metrics

In addition to standard ADK metrics (`trajectory_quality`, `tool_use_quality`, `groundedness`, `safety`), the evaluation suite will implement the 14 custom metrics specified in `improvement-plan.md`:
1. `knowledge_base_coverage`: Proportion of required research directions with verified claim grounding (`>= 0.90`).
2. `claim_location_resolution`: Percentage of material claims resolving to exact source locations (`>= 0.95`).
3. `counterevidence_search_coverage`: Ratio of contradictory/null searches to positive searches (`>= 0.30`).
4. `cross_stage_evidence_reuse`: Percentage of verified baseline claims cited in downstream reviews/rankings.
5. `candidate_domain_specificity`: Semantic density of domain terms vs. generic template language.
6. `candidate_mechanism_diversity`: Inverse pairwise Jaccard similarity across candidate population (`>= 0.75`).
7. `review_ranking_consistency`: Alignment correlation between `CandidateReview` recommendations and Elo rank.
8. `fatal_flaw_consistency`: Verifies 100% exclusion of candidates with unresolved fatal flaws from final recommendations.
9. `internal_id_leakage`: Zero count of raw UUIDs in default report outputs.
10. `raw_serialization_leakage`: Zero count of unparsed JSON blocks in default report outputs.
11. `report_section_completeness`: Presence of all 15 required sections in `mode="default"`.
12. `citation_link_integrity`: 100% resolution of `[n]` citation markers to bibliography entries.
13. `report_navigation_and_layout`: Validation of TOC links and table formatting without truncation.
14. `reader_decision_comprehension`: LLM-as-judge scoring of executive synthesis clarity and decision readiness (`>= 0.85`).

### 6.3 Quality Flywheel Execution Plan

```bash
# 1. Synthesize multi-turn scientific evaluation scenarios
agents-cli eval dataset synthesize --project mini-coscientist-v2 --out evals/datasets/v2_benchmark.json

# 2. Run agent execution traces against the benchmark dataset
agents-cli eval generate --dataset evals/datasets/v2_benchmark.json --out evals/traces/v2_run1.json

# 3. Grade traces using custom scientific & report metrics
agents-cli eval grade --traces evals/traces/v2_run1.json --config evals/config/judge_config.yaml --out evals/results/grade_v2_run1.json

# 4. Cluster failure modes and inspect regression deltas
agents-cli eval analyze --results evals/results/grade_v2_run1.json
agents-cli eval compare --baseline evals/results/grade_baseline.json --candidate evals/results/grade_v2_run1.json
```

### 6.4 Rollout & Exit Criteria Checklist

Before promoting `mini-coscientist-v2` to production dev/staging:
- [ ] All deterministic unit & integration tests pass in `pytest`.
- [ ] 100% of cited claim and source references resolve to verified locations.
- [ ] `>= 95%` of sampled material claims are supported or explicitly labeled uncertain.
- [ ] Overall multi-turn task success is `>= 0.85` (with no individual research mode below `0.75`).
- [ ] Groundedness score is `>= 0.90`.
- [ ] Zero internal UUIDs or JSON serialization markers leak into `result.pdf` or `report.md`.
- [ ] Qualified domain-expert review approves protocol rigor and report usability.

---
*End of Detailed Design Document.*
