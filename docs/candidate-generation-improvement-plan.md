# Candidate Generation & Report Presentation: V2 Improvement Plan

**Author:** Google AI CE / Co-Scientist Pair Programming  
**Date:** 2026-07-29  
**Reference Document:** `sample2.pdf` (45-mer Hydrophobic Therapeutic Peptide Synthesis Report)  
**Status:** PROPOSED FOR FUTURE PHASES  

---

## 1. Executive Summary & Comparative Analysis

Following the implementation of **Phase 1 (Evidence & Normalization Gates)** and the **V2 4x2 Parallel Strategy Candidate Generation Architecture**, we generated a local end-to-end report (`generated_report_comparison.md`) and conducted a section-by-section comparison against the reference benchmark `sample2.pdf`.

### 1.1 Comparative Scorecard: V1 (`sample2.pdf`) vs. V2 (Current Implementation)

| Architectural Dimension | V1 Benchmark (`sample2.pdf`) | V2 Current Implementation (`generated_report_comparison.md`) | Proposed Next Advancement (V3) |
| :--- | :--- | :--- | :--- |
| **Candidate Card Structure** | Unstructured narrative paragraphs in Section 4; key experimental parameters buried in prose. | **8-Part Structured Scientific Cards** (`Title`, `Strategy`, `Claim`, `Mechanism & Model`, `Plausibility`, `Categorized Evidence`, `Predictions/Alternatives`, `Falsifier`, `Validation Protocol`, `Risks`). | Interactive UI folding & automated **Mermaid pathway/ligation diagrams** for each candidate card. |
| **Generation Strategy & Diversity** | Single generalist prompt call generating all candidates; risk of semantic convergence. | **4 Parallel Strategy Specialists** (`evidence_first`, `mechanism_first`, `analogy_transfer`, `competing_explanation`) generating 2 cards each (`4x2 Fan-Out`). | Dynamic allocation of generation budget based on unresolved gaps in `KnowledgeBaseManifest`. |
| **Normalization & Quality Gates** | No automated word-count or Jaccard diversity enforcement. | **Automated Normalization Service** enforcing Jaccard distinctness (`<= 0.85`) and minimum word counts (`>= 20` words) for mechanism and protocol. | Automated domain-specific construct checking (e.g., verifying sequence fragmentation points for chemistry/peptide tasks). |
| **Evidence Lineage & Grounding** | Uncategorized literature citations mixed in narrative text. | **Categorized Evidence Grounding** (`evidence_for`, `evidence_against`, `evidence_gaps`) explicitly listed per candidate. | Clickable DOI/PMID `CitationAnchor` links verified against `source_verification` packets. |
| **Cross-Candidate Synthesis** | Tabular summary in Section 5 comparing Epimerization, Yield, and 2026 Plausibility. | Sequential presentation of candidate cards in Section 4 and rankings in Section 5. | Executive **Cross-Candidate Evaluation Matrix Table** automatically synthesized from `ReviewSet` and `TournamentState`. |

---

## 2. Key Findings from `sample2.pdf` Comparison

1. **Rich Domain Context vs. Structured Rigor:**
   - `sample2.pdf` excels at domain-specific technical prose (e.g., detailing achiral `Gly15-Ala16` and `Gly30-Ala31` fragmentation points, depsipeptide switches, and fluorophilic `Butelase-GL` ligation in 30% HFIP).
   - Our **V2 Implementation** ensures that *every* candidate—regardless of domain—adheres to an auditable **8-part scientific schema** that prevents thin ideas from bypassing peer review.

2. **Visual Synthesis of Complex Mechanisms:**
   - In `sample2.pdf`, multi-step chemical assembly and ligation tracks are described purely in text. A major upgrade for reader comprehension is automatically rendering **visual reaction/workflow diagrams** using Mermaid notation inside the Candidate card.

---

## 3. Recommended Improvement Plan (Phases A – D)

### Phase A: Executive Cross-Candidate Comparison Table (Report Dossier Enhancement)
- **Objective:** Bridge the narrative style of `sample2.pdf` with V2's typed artifacts by generating an upfront summary table in `coscientist/dossier.py` before listing full candidate cards.
- **Implementation:**
  - In `compile_dossier()`, extract `TournamentState.rankings`, `ReviewSet` scores, and `CandidatePopulation` metadata.
  - Render a comparative Markdown table with columns:
    - `Rank` | `Candidate Title` | `Strategy` | `Mechanism Summary` | `Evidence Strength` | `Fatal Flaws` | `Shortlist Status`
- **Acceptance Gate:** Unit test verifying that `compile_dossier()` emits a well-formed Markdown table joining candidate titles with ranking scorecards.

### Phase B: Visual Pathway & Ligation Diagram Generation (Mermaid Integration)
- **Objective:** Enable specialists to emit structured Mermaid diagrams illustrating candidate mechanisms, experimental workflows, or molecular fragmentation schemes.
- **Implementation:**
  - Add an optional field `workflow_diagram_mermaid: str = ""` to `Candidate(Contract)` in `coscientist/models.py`.
  - Update `SPECIALISTS` instructions in `coscientist/agents.py` to prompt models to generate a `graph TD` or `sequenceDiagram` for complex experimental protocols.
  - Format the diagram inside `coscientist/dossier.py` and `coscientist/presentation.py` using fenced ````mermaid` blocks.
- **Acceptance Gate:** Verify that Mermaid diagrams render cleanly without unquoted special characters or syntax errors.

### Phase C: Verified Citation Anchoring & Epistemic Badging
- **Objective:** Ensure all references in `evidence_for`, `evidence_against`, and `evidence_gaps` map directly to verified items in the `KnowledgeBaseManifest`.
- **Implementation:**
  - Create a citation resolver in `coscientist/normalization.py` that checks candidate reference strings against `CitationAnchor` records.
  - Automatically attach epistemic status badges (`[Verified Source]`, `[Unverified Lead]`, `[Literature-Only Fallback]`) to each bullet point in the rendered report.
- **Acceptance Gate:** Any unverified empirical claim in `evidence_for` triggers a warning in the report quality audit.

### Phase D: Interactive Candidate Section Refinement in Web UI / TUI
- **Objective:** Provide granular HITL control over individual candidate sections.
- **Implementation:**
  - Extend `/api/research/sessions/{id}/decisions` in `app/fast_api_app.py` to accept targeted section edits (e.g., `{"action": "refine_section", "candidate_id": "cand_1", "section": "validation_protocol", "feedback": "Add positive control arm"}`).
  - Invoke `SpecialistAgent` with a focused prompt to re-elucidate only the targeted field while preserving the remainder of the 8-part card.
- **Acceptance Gate:** End-to-end integration test confirming that refining `validation_protocol` updates the candidate card without altering `claim` or `falsifier`.

---

## 4. Immediate Action Items
1. Review and approve this Improvement Plan.
2. Schedule **Phase A (Executive Cross-Candidate Comparison Table)** for implementation in the next engineering sprint.
