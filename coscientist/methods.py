"""Domain adapters for six families of scientific research methods."""

from __future__ import annotations

from .models import RESEARCH_MODES
from .disciplines import classify_discipline, DISCIPLINES, DISCIPLINE_TAXONOMY

METHOD_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "experimental": (
        "intervention and comparator/control",
        "independent and dependent variables",
        "randomization/blinding where applicable",
        "sample size or power rationale",
        "calibration, safety limits, analysis, and replication",
    ),
    "observational": (
        "target population and sampling frame",
        "inclusion/exclusion criteria",
        "confounders and causal-identification assumptions",
        "missing-data, privacy, and ethics plan",
        "robustness checks and limits on causal claims",
    ),
    "computational": (
        "dataset provenance, licenses, and leakage controls",
        "task definition, baselines, and train/validation/test separation",
        "metrics, ablations, compute, and environment record",
        "error analysis and reproducibility package",
        "misuse and dual-use assessment",
    ),
    "theory_simulation": (
        "definitions, assumptions, and proof obligations",
        "boundary conditions and limiting cases",
        "analytical or numerical verification",
        "falsifiable consequences",
        "links to observable phenomena where relevant",
    ),
    "systematic_review": (
        "pre-registered protocol and search strategy",
        "eligibility, screening, and extraction procedures",
        "risk-of-bias assessment",
        "synthesis model, heterogeneity, and sensitivity analysis",
        "certainty-of-evidence assessment",
    ),
    "measurement_field": (
        "construct and measurement model",
        "calibration, traceability, and uncertainty budget",
        "sampling/site protocol and quality assurance",
        "data management",
        "independent validation",
    ),
}


def classify_research_mode(question: str) -> str:
    """Conservative keyword triage that a researcher may override."""
    text = question.lower()
    rules = (
        (
            "systematic_review",
            ("systematic review", "meta-analysis", "evidence synthesis"),
        ),
        (
            "computational",
            (
                "machine learning",
                "artificial intelligence",
                "dataset",
                "benchmark",
                "algorithm",
                "software",
            ),
        ),
        (
            "theory_simulation",
            ("theorem", "proof", "mathematical", "simulation", "theoretical model"),
        ),
        (
            "measurement_field",
            ("instrument", "sensor", "calibration", "field study", "measurement"),
        ),
        (
            "observational",
            ("observational", "cohort", "survey", "case-control", "registry"),
        ),
    )
    for mode, terms in rules:
        if any(term in text for term in terms):
            return mode
    return "experimental"


def method_requirements(mode: str) -> tuple[str, ...]:
    if mode not in RESEARCH_MODES:
        raise ValueError(
            f"Unsupported research mode '{mode}'. Choose one of: "
            f"{', '.join(RESEARCH_MODES)}"
        )
    return METHOD_REQUIREMENTS[mode]
