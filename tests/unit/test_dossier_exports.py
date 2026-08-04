from io import BytesIO
from zipfile import ZipFile

from coscientist.dossier import render_docx, render_pdf

DOSSIER = """# Co-Scientist Research Dossier

**Question:** Does the intervention improve the measured outcome?

## Executive synthesis

- Candidate 1 remains testable.
- Independent replication is required.

## Decision audit

| Candidate | Decision |
| --- | --- |
| Candidate 1 | Conditional advance |
"""


def test_pdf_export_is_a_valid_pdf():
    exported = render_pdf(DOSSIER)

    assert exported.startswith(b"%PDF-")
    assert len(exported) > 1000


def test_docx_export_is_google_docs_compatible():
    exported = render_docx(DOSSIER)

    assert exported.startswith(b"PK")
    with ZipFile(BytesIO(exported)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "Co-Scientist Research Dossier" in document_xml
    assert "Candidate 1 remains testable" in document_xml


def test_pdf_export_renders_tables_as_reportlab_table_objects():
    table_dossier = """# Title
    
| Col A | Col B |
| --- | --- |
| Val 1 | Val 2 |
"""
    exported = render_pdf(table_dossier)
    assert exported.startswith(b"%PDF-")
    assert len(exported) > 1000


def test_compile_dossier_toc_alignment_and_evidence_discovery():
    import re
    from coscientist.dossier import compile_dossier
    from coscientist.models import ApprovalProfile
    from coscientist.orchestration import CoScientistWorkflow

    flow = CoScientistWorkflow(
        "Design a 2026-State-of-the-Art Synthesis Strategy for a 45-mer Hydrophobic Therapeutic Peptide",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=1,
    )
    flow.accept_literature_only()
    flow.run_auto()
    from coscientist.models import Artifact, DiscoveryManifest, DeepResearchRun
    manifest = DiscoveryManifest(
        question=flow.session.question,
        runs=[DeepResearchRun(pass_number=1, status="completed")],
    )
    flow.session.artifacts.append(
        Artifact(
            stage="evidence",
            agent="deep_research_discovery",
            artifact_type="specialist_output",
            content="Evidence discovered.",
            schema_name="DiscoveryManifest",
            payload=manifest.model_dump(mode="json"),
        )
    )
    report_md = compile_dossier(flow.session)

    toc_titles = [
        m.group(1).strip()
        for m in re.finditer(r"^\s*-\s*\[([^\]]+)\]\(#", report_md, re.MULTILINE)
    ]
    heading_titles = [
        m.group(1).strip()
        for m in re.finditer(r"^##\s+(.+)$", report_md, re.MULTILINE)
        if m.group(1).strip() != "Table of Contents"
    ]
    assert toc_titles == heading_titles, f"TOC mismatch: {toc_titles} != {heading_titles}"

    assert "## Evidence Discovery" in report_md
    assert "- Deep Research passes:" in report_md
    assert "- Distinct source leads:" in report_md

    main_section = report_md.split("## Complete artifact appendix")[0]
    assert not re.search(r"\bcandidate_[a-z0-9_-]+\b", main_section, re.IGNORECASE)


def test_render_pdf_clickable_toc_mermaid_and_index():
    md = """# Co-Scientist Research Dossier

## Table of Contents

- [Executive synthesis (Executive Summary)](#executive-synthesis-executive-summary)
- [Evidence Discovery](#evidence-discovery)

## Executive synthesis (Executive Summary)

Summary content here.
| Col A | Col B |
| --- | --- |
| Val 1 | Val 2 |

## Evidence Discovery

```mermaid
graph TD
    A[Baseline Strategy] --> B(Evaluation Stage)
```
"""
    pdf_bytes = render_pdf(md)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2500

    assert b"/Link" in pdf_bytes or b"/Annot" in pdf_bytes
    assert b"/Dest" in pdf_bytes or b"/Dests" in pdf_bytes or b"/Names" in pdf_bytes

    assert b"Index of Figures and Tables" in pdf_bytes
    assert b"Figure 1" in pdf_bytes or "Figure 1".encode("utf-16-be") in pdf_bytes
    assert b"Table 1" in pdf_bytes or "Table 1".encode("utf-16-be") in pdf_bytes
    assert b"Page" in pdf_bytes or "Page".encode("utf-16-be") in pdf_bytes


def test_flywheel_evaluators_detect_defects():
    from coscientist.dossier import compile_dossier, render_pdf
    from coscientist.models import ApprovalProfile
    from coscientist.orchestration import CoScientistWorkflow
    from run_10_iteration_5_agent_flywheel import (
        evaluate_epistemic_evidence,
        evaluate_executive_synthesis_toc,
        evaluate_pdf_formatting_typography,
        evaluate_scientific_content,
        evaluate_structural_completeness,
    )

    flow = CoScientistWorkflow(
        "Design a 2026-State-of-the-Art Synthesis Strategy for a 45-mer Hydrophobic Therapeutic Peptide",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=1,
    )
    flow.accept_literature_only()
    flow.run_auto()
    clean_md = compile_dossier(flow.session)
    clean_pdf = render_pdf(clean_md)

    sc_score, _ = evaluate_scientific_content(flow.session, clean_md)
    pdf_score, _ = evaluate_pdf_formatting_typography(clean_pdf, clean_md)
    epi_score, _ = evaluate_epistemic_evidence(flow.session, clean_md)
    str_score, _ = evaluate_structural_completeness(flow.session, clean_md)
    toc_score, _ = evaluate_executive_synthesis_toc(flow.session, clean_md)
    total_clean = sc_score + pdf_score + epi_score + str_score + toc_score
    assert total_clean == 100, f"Clean total was {total_clean} != 100"

    defective_md = (
        clean_md.replace("## Evidence Discovery", "## Omitted Discovery")
        + "\nTODO: fix candidate_xxxx\n"
    )
    sc_score2, _ = evaluate_scientific_content(flow.session, defective_md)
    epi_score2, _ = evaluate_epistemic_evidence(flow.session, defective_md)
    str_score2, _ = evaluate_structural_completeness(flow.session, defective_md)
    assert epi_score2 < 20, "Epistemic evaluator failed to deduct points for missing Evidence Discovery"
    assert str_score2 < 20, "Structural evaluator failed to deduct points for TODO/candidate_xxxx"
    assert (sc_score2 + epi_score2 + str_score2) < 60
