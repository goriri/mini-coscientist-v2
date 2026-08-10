"""Typed Python Discipline Registry for multi-agent domain-specific scientific research."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import DISCIPLINES

DISCIPLINE_TAXONOMY = DISCIPLINES


class DisciplineProfile(BaseModel):
    """Typed contract specifying domain-specific prompts, rubrics, and checklists."""

    name: str
    default_actor_guidance: str = ""
    actor_guidance: dict[str, str] = Field(default_factory=dict)
    default_critic_rubric: str = ""
    """What a critic must reject at every stage, which is nothing any profile records.

    Each of the twelve profiles set this to a verbatim copy of its generate rubric,
    so every stage's critic was handed the generate stage's demands. The reflect
    critics were told to reject a review that omits "a structured Evaluation Table
    and an explicit Critical Scientific Judgment section" -- artefacts of the
    hypothesis contract, asked of a reviewer -- and the reviewers complied: fourteen
    review findings on a live run were a Markdown table in a prose field. What is
    domain-specific and stage-specific is already carried by ``stage_checklists``.
    """
    critic_rubrics: dict[str, str] = Field(default_factory=dict)
    stage_checklists: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    def get_actor_guidance(self, stage: str) -> str:
        parts = []
        if self.default_actor_guidance:
            parts.append(self.default_actor_guidance)
        if stage in self.actor_guidance:
            parts.append(self.actor_guidance[stage])
        return "\n\n".join(parts) if parts else ""

    def get_critic_rubric(self, stage: str) -> str:
        parts = []
        if self.default_critic_rubric:
            parts.append(self.default_critic_rubric)
        if stage in self.critic_rubrics:
            parts.append(self.critic_rubrics[stage])
        return "\n\n".join(parts) if parts else ""

    def get_stage_checklist(self, stage: str) -> tuple[str, ...]:
        return self.stage_checklists.get(stage, ())


def _build_default_stage_checklists(domain_label: str) -> dict[str, tuple[str, ...]]:
    return {
        "scope": (
            f"Define target system and research objective in {domain_label}.",
            f"Specify explicit domain boundaries, variables, and feasibility constraints for {domain_label}.",
            f"Identify ethical, safety, and governance requirements relevant to {domain_label}.",
        ),
        "evidence": (
            f"Query authoritative {domain_label} literature databases and repositories.",
            f"Verify primary sources and supporting claims in {domain_label} without relying on search snippets.",
            f"Document empirical limitations, negative results, or retractions in {domain_label}.",
        ),
        "generate": (
            f"Formulate falsifiable hypotheses with domain-specific mechanisms in {domain_label}.",
            f"Specify concrete experimental, observational, or analytical protocols for {domain_label}.",
            f"Include domain-appropriate controls, calibration, and validation checks in {domain_label}.",
        ),
        "reflect": (
            f"Critique mechanistic plausibility against established {domain_label} principles.",
            f"Examine methodological feasibility and potential confounding factors in {domain_label}.",
            f"Verify that novelty claims are grounded in primary {domain_label} literature.",
        ),
        "rank": (
            f"Compare candidates using multi-criteria tournament scoring for {domain_label}.",
            f"Balance novelty and expected scientific impact against methodological feasibility in {domain_label}.",
            f"Provide explicit comparative rationales grounded in {domain_label} evidence.",
        ),
        "evolve": (
            f"Refine shortlisted {domain_label} candidates by addressing specific critic feedback.",
            f"Enhance experimental or analytical rigor with improved controls and falsifiers in {domain_label}.",
            f"Independently re-review evolved {domain_label} hypotheses before promotion.",
        ),
        "proximity": (
            f"Map adjacent {domain_label} research clusters and competing explanations.",
            f"Identify shared mechanisms, outcome overlaps, and required data sources in {domain_label}.",
            f"Highlight under-explored or minority hypotheses in {domain_label}.",
        ),
        "meta_review": (
            f"Conduct final audit of factual claims, citations, and fatal flaws in {domain_label}.",
            f"Verify safety, ethics, and governance readiness for {domain_label} research.",
            f"Synthesize definitive recommendation and conditions for advancing {domain_label} proposals.",
        ),
        "report": (
            f"Compile comprehensive, auditable research dossier tailored to {domain_label} standards.",
            f"Present structured tables, diagrams, and citations for {domain_label} peer review.",
            f"Ensure transparent documentation of assumptions, limitations, and verification status in {domain_label}.",
        ),
    }


def _create_registry() -> dict[str, DisciplineProfile]:
    registry: dict[str, DisciplineProfile] = {}

    # 1. chemistry_materials
    #
    # This profile covers a peptide synthesis and an inorganic thin film alike, so the
    # pillars it mandates have to be ones both systems have. They were written off the
    # peptide benchmark and named its constructs outright -- epimerization controls,
    # Stereochemical Integrity, Ligation Strategy, exact fragmentation points -- and
    # the generate critic rejected any draft that left them out. A live run on ALD
    # coatings for NMC811 cathodes duly supplied them, and the words reached the
    # reader: "Strict epimerization controls -- in this solid-state context, precise
    # precursor pulsing to prevent uneven film growth" (an amorphous oxide film has no
    # stereocentre to invert), "Ligation Strategy | ALD coating adhesion under
    # mechanical stress" (nothing is being ligated; the row means does it stick), "a
    # >50% reduction in exact fragmentation points for m/z 44", and "the primary
    # biophysical risk is a potential increase in initial Li-ion charge-transfer
    # resistance" in a coin cell. So the categories that are properties of a specific
    # chemistry are asked for by what they measure, with the peptide terms kept as the
    # example they are, and the critic now rejects a term used outside its domain
    # instead of demanding it.
    registry["chemistry_materials"] = DisciplineProfile(
        name="chemistry_materials",
        default_actor_guidance=(
            "Apply Chemistry and Materials Science standards: specify chemical structures or phases, reagents and precursors, "
            "reaction pathways, synthetic or deposition routes, thermodynamic/kinetic parameters, and the controls that suppress "
            "the side reactions the system at hand can actually undergo."
        ),
        actor_guidance={
            "generate": (
                "For candidate hypothesis generation ('generate' stage), EVERY candidate must strictly follow "
                "the gold-standard scientific structure of chemistry/materials benchmarks:\n"
                "1. Rich Technical Narrative: Detailed chemical, structural or physical mechanisms, specific state-of-the-art reagents, precursors or additives, synthetic or deposition routes, the characterization endpoints that resolve them, and quantitative parameters.\n"
                "2. Motivation and Supporting Evidence: A dedicated subsection explaining the scientific rationale and citing empirical evidence from the knowledge base.\n"
                "3. Evaluation of Idea Table: A structured Markdown table ('Category | Description | Judgment') of five categories. Technological Leap and Purity Potential are fixed; the other three must name properties the proposed system actually has. For a molecular synthesis those are typically Aggregation Control, Stereochemical Integrity and Ligation Strategy; for a solid-state, thin-film or bulk-materials system name that system's equivalents instead -- phase stability, film conformality, interfacial adhesion, defect control -- and never carry over a category the system has no instance of.\n"
                "4. Critical Scientific Judgment: An explicit judgment paragraph balancing strengths against the specific risks of this chemistry -- epimerization or racemization where there are stereocentres, parasitic CVD, phase reconstruction or dissolution where there are not -- and the failure modes that follow from them."
            )
        },
        critic_rubrics={
            "generate": (
                "Reject drafts that omit mandatory chemistry/materials rigor pillars: specific state-of-the-art reagents, precursors or additives, "
                "the synthetic or deposition route, the controls on the side reactions this system can undergo, a structured Evaluation Table whose five categories "
                "are properties the proposed system has, and an explicit Critical Scientific Judgment section evaluating chemical risks. "
                "Reject as a rigor failure, not accept as the pillar it imitates, any construct applied outside its domain: epimerization controls or "
                "stereochemical integrity claimed for a system with no stereocentre, a ligation strategy for one with nothing being joined, "
                "fragmentation points where no species is being fragmented, or a biophysical risk in a system with no biology in it."
            )
        },
        stage_checklists=_build_default_stage_checklists(
            "chemistry and materials science"
        ),
    )

    # 2. biology_medicine
    registry["biology_medicine"] = DisciplineProfile(
        name="biology_medicine",
        default_actor_guidance=(
            "Apply Biology and Medicine standards: specify biological pathways, experimental models (in vitro/in vivo), "
            "cohort designs, inclusion/exclusion criteria, sample sizes, statistical power, and safety/ethics controls."
        ),
        actor_guidance={
            "generate": (
                "For candidate hypothesis generation ('generate' stage), EVERY candidate must strictly follow "
                "the gold-standard scientific structure of biology/medicine benchmarks:\n"
                "1. Rich Technical Narrative: Detailed biological mechanisms, cohort designs, experimental models (in vitro/in vivo), sample sizes, and statistical power calculations.\n"
                "2. Motivation and Supporting Evidence: A dedicated subsection explaining the scientific rationale and citing empirical evidence from the knowledge base.\n"
                "3. Evaluation of Idea Table: A structured Markdown table ('Category | Description | Judgment') evaluating Cohort Validity, Statistical Power, Biomarker Specificity, Translational Leap, and Clinical Feasibility.\n"
                "4. Critical Scientific Judgment: An explicit judgment paragraph balancing strengths against specific biological confounders, toxicity, or clinical failure modes."
            )
        },
        critic_rubrics={
            "generate": (
                "Reject drafts that omit mandatory biology/medicine rigor pillars: specific biological mechanisms, "
                "cohort designs or experimental models (in vitro/in vivo), sample sizes and statistical power calculations, confounder controls, "
                "biomarker specificity, a structured Evaluation Table, and an explicit Critical Scientific Judgment section evaluating biological risks or toxicity."
            )
        },
        stage_checklists=_build_default_stage_checklists("biology and medicine"),
    )

    # 3. physics_engineering
    registry["physics_engineering"] = DisciplineProfile(
        name="physics_engineering",
        default_actor_guidance=(
            "Apply Physics and Engineering standards: specify physical principles, conservation laws, equations of motion, "
            "experimental apparatus, calibration traceable to standards, and uncertainty budget."
        ),
        actor_guidance={
            "generate": (
                "For candidate hypothesis generation ('generate' stage), EVERY candidate must strictly follow "
                "the gold-standard scientific structure of physics/engineering benchmarks:\n"
                "1. Rich Technical Narrative: Detailed physical principles, conservation laws, equations of motion, experimental setups, calibration protocols, and quantitative parameters.\n"
                "2. Motivation and Supporting Evidence: A dedicated subsection explaining the scientific rationale and citing empirical evidence from the knowledge base.\n"
                "3. Evaluation of Idea Table: A structured Markdown table ('Category | Description | Judgment') evaluating Physical Rigor, Calibration Traceability, Experimental Feasibility, Technological Leap, and Uncertainty Quantification.\n"
                "4. Critical Scientific Judgment: An explicit judgment paragraph balancing strengths against specific physical constraints, instrument noise, or failure modes."
            )
        },
        critic_rubrics={
            "generate": (
                "Reject drafts that omit mandatory physics/engineering rigor pillars: physical principles, conservation laws, "
                "equations of motion, calibration protocols, uncertainty quantification, a structured Evaluation Table, "
                "and an explicit Critical Scientific Judgment section evaluating physical limits."
            )
        },
        stage_checklists=_build_default_stage_checklists("physics and engineering"),
    )

    # 4. computer_science_ai
    registry["computer_science_ai"] = DisciplineProfile(
        name="computer_science_ai",
        default_actor_guidance=(
            "Apply Computer Science and AI standards: specify algorithmic complexity, formal architecture definitions, "
            "training/inference compute requirements, benchmark datasets, baseline comparisons, and data leakage controls."
        ),
        actor_guidance={
            "generate": (
                "For candidate hypothesis generation ('generate' stage), EVERY candidate must strictly follow "
                "the gold-standard scientific structure of computer science/AI benchmarks:\n"
                "1. Rich Technical Narrative: Detailed algorithmic complexity, formal architecture specifications, training/inference compute requirements, benchmark dataset specifications, and data leakage controls.\n"
                "2. Motivation and Supporting Evidence: A dedicated subsection explaining the scientific rationale and citing empirical evidence from the knowledge base.\n"
                "3. Evaluation of Idea Table: A structured Markdown table ('Category | Description | Judgment') evaluating Algorithmic Complexity, Dataset Diversity, Baseline Superiority, Compute Efficiency, and Robustness.\n"
                "4. Critical Scientific Judgment: An explicit judgment paragraph balancing strengths against specific computational risks, overfitting, data leakage, or failure modes."
            )
        },
        critic_rubrics={
            "generate": (
                "Reject drafts that omit mandatory computer science/AI rigor pillars: algorithmic complexity, "
                "formal problem/architecture definitions, training/inference compute requirements, benchmark dataset specifications, "
                "data leakage controls, a structured Evaluation Table, and an explicit Critical Scientific Judgment section evaluating computational limits."
            )
        },
        stage_checklists=_build_default_stage_checklists("computer science and AI"),
    )

    # 5. mathematics_statistics
    registry["mathematics_statistics"] = DisciplineProfile(
        name="mathematics_statistics",
        default_actor_guidance=(
            "Apply Mathematics and Statistics standards: specify axiomatic rigor, mathematical axioms, proof sketches, formal definitions, "
            "boundary conditions, analytical or statistical derivations, and limiting cases."
        ),
        actor_guidance={
            "generate": (
                "For candidate hypothesis generation ('generate' stage), EVERY candidate must strictly follow "
                "the gold-standard scientific structure of mathematics/statistics benchmarks:\n"
                "1. Rich Technical Narrative: Detailed mathematical axioms, proof sketches, formal definitions, boundary conditions, and analytical/statistical derivations.\n"
                "2. Motivation and Supporting Evidence: A dedicated subsection explaining the scientific rationale and citing theorems or empirical evidence from the knowledge base.\n"
                "3. Evaluation of Idea Table: A structured Markdown table ('Category | Description | Judgment') evaluating Axiomatic Rigor, Proof Sketch Completeness, Generalizability, Analytical Leap, and Falsifiability.\n"
                "4. Critical Scientific Judgment: An explicit judgment paragraph balancing strengths against specific proof gaps, degenerate cases, or analytical failure modes."
            )
        },
        critic_rubrics={
            "generate": (
                "Reject drafts that omit mandatory mathematics/statistics rigor pillars: axiomatic rigor, proof sketch completeness, "
                "formal definitions, boundary conditions, analytical or statistical derivations, a structured Evaluation Table, "
                "and an explicit Critical Scientific Judgment section evaluating proof gaps or degenerate cases."
            )
        },
        stage_checklists=_build_default_stage_checklists("mathematics and statistics"),
    )

    # 6. earth_climate_sciences
    registry["earth_climate_sciences"] = DisciplineProfile(
        name="earth_climate_sciences",
        default_actor_guidance=(
            "Apply Earth and Climate Sciences standards: specify geophysical mechanisms, spatiotemporal scales, "
            "boundary conditions, observational/reanalysis datasets, and uncertainty quantification."
        ),
        actor_guidance={
            "generate": (
                "For candidate hypothesis generation ('generate' stage), EVERY candidate must strictly follow "
                "the gold-standard scientific structure of earth/climate sciences benchmarks:\n"
                "1. Rich Technical Narrative: Detailed geophysical mechanisms, spatiotemporal scales, boundary conditions, observational datasets, and uncertainty quantification.\n"
                "2. Motivation and Supporting Evidence: A dedicated subsection explaining the scientific rationale and citing empirical evidence from the knowledge base.\n"
                "3. Evaluation of Idea Table: A structured Markdown table ('Category | Description | Judgment') evaluating Spatiotemporal Resolution, Observational Support, Model Parameterization, Geoscience Leap, and Uncertainty Management.\n"
                "4. Critical Scientific Judgment: An explicit judgment paragraph balancing strengths against specific observational biases, chaotic dynamics, or climate model limitations."
            )
        },
        critic_rubrics={
            "generate": (
                "Reject drafts that omit mandatory earth/climate sciences rigor pillars: geophysical mechanisms, "
                "spatiotemporal scales, boundary conditions, observational datasets, uncertainty quantification, a structured Evaluation Table, "
                "and an explicit Critical Scientific Judgment section evaluating geophysical limitations."
            )
        },
        stage_checklists=_build_default_stage_checklists("earth and climate sciences"),
    )

    # 7. neuroscience_cognitive
    registry["neuroscience_cognitive"] = DisciplineProfile(
        name="neuroscience_cognitive",
        default_actor_guidance=(
            "Apply Neuroscience and Cognitive Science standards: specify neural mechanisms, behavioral paradigms, "
            "neuroimaging/electrophysiological controls, sample sizes, and statistical power."
        ),
        actor_guidance={
            "generate": (
                "For candidate hypothesis generation ('generate' stage), EVERY candidate must strictly follow "
                "the gold-standard scientific structure of neuroscience/cognitive benchmarks:\n"
                "1. Rich Technical Narrative: Detailed neural mechanisms, behavioral paradigms, neuroimaging/electrophysiological controls, sample sizes, and statistical power calculations.\n"
                "2. Motivation and Supporting Evidence: A dedicated subsection explaining the scientific rationale and citing empirical evidence from the knowledge base.\n"
                "3. Evaluation of Idea Table: A structured Markdown table ('Category | Description | Judgment') evaluating Neural Plausibility, Behavioral Control, Measurement Resolution, Cognitive Leap, and Statistical Robustness.\n"
                "4. Critical Scientific Judgment: An explicit judgment paragraph balancing strengths against specific experimental confounders, measurement noise, or cognitive task limitations."
            )
        },
        critic_rubrics={
            "generate": (
                "Reject drafts that omit mandatory neuroscience/cognitive rigor pillars: neural mechanisms, "
                "behavioral paradigms, neuroimaging/electrophysiological controls, sample sizes and statistical power calculations, a structured Evaluation Table, "
                "and an explicit Critical Scientific Judgment section evaluating experimental confounders."
            )
        },
        stage_checklists=_build_default_stage_checklists(
            "neuroscience and cognitive science"
        ),
    )

    # 8. astronomy_astrophysics
    registry["astronomy_astrophysics"] = DisciplineProfile(
        name="astronomy_astrophysics",
        default_actor_guidance=(
            "Apply Astronomy and Astrophysics standards: specify astrophysical mechanisms, observational wavelengths/instruments, "
            "signal-to-noise ratios, calibration controls, and systematic error quantification."
        ),
        actor_guidance={
            "generate": (
                "For candidate hypothesis generation ('generate' stage), EVERY candidate must strictly follow "
                "the gold-standard scientific structure of astronomy/astrophysics benchmarks:\n"
                "1. Rich Technical Narrative: Detailed astrophysical mechanisms, observational wavelengths/instruments, signal-to-noise ratios, calibration controls, and quantitative parameters.\n"
                "2. Motivation and Supporting Evidence: A dedicated subsection explaining the scientific rationale and citing empirical evidence from the knowledge base.\n"
                "3. Evaluation of Idea Table: A structured Markdown table ('Category | Description | Judgment') evaluating Astrophysical Plausibility, Instrumental Resolution, Signal-to-Noise Ratio, Discovery Leap, and Systematic Error Control.\n"
                "4. Critical Scientific Judgment: An explicit judgment paragraph balancing strengths against specific instrumental limitations, systematic errors, or astrophysical confounds."
            )
        },
        critic_rubrics={
            "generate": (
                "Reject drafts that omit mandatory astronomy/astrophysics rigor pillars: astrophysical mechanisms, "
                "observational wavelengths/instruments, signal-to-noise ratios, calibration controls, a structured Evaluation Table, "
                "and an explicit Critical Scientific Judgment section evaluating systematic errors."
            )
        },
        stage_checklists=_build_default_stage_checklists("astronomy and astrophysics"),
    )

    # 9. social_science_economics
    registry["social_science_economics"] = DisciplineProfile(
        name="social_science_economics",
        default_actor_guidance=(
            "Apply Social Science and Economics standards: specify theoretical constructs, econometric/causal identification strategies, "
            "survey/census dataset provenance, confounder controls, and robustness checks."
        ),
        actor_guidance={
            "generate": (
                "For candidate hypothesis generation ('generate' stage), EVERY candidate must strictly follow "
                "the gold-standard scientific structure of social science/economics benchmarks:\n"
                "1. Rich Technical Narrative: Detailed theoretical constructs, econometric/causal identification strategies, survey/census datasets, confounder controls, and robustness checks.\n"
                "2. Motivation and Supporting Evidence: A dedicated subsection explaining the scientific rationale and citing empirical evidence from the knowledge base.\n"
                "3. Evaluation of Idea Table: A structured Markdown table ('Category | Description | Judgment') evaluating Causal Identification, Construct Validity, Sample Representativeness, Theoretical Leap, and Policy Relevance.\n"
                "4. Critical Scientific Judgment: An explicit judgment paragraph balancing strengths against specific endogeneity, selection bias, or socio-economic confounders."
            )
        },
        critic_rubrics={
            "generate": (
                "Reject drafts that omit mandatory social science/economics rigor pillars: theoretical constructs, "
                "econometric/causal identification strategies, dataset provenance, confounder controls, robustness checks, a structured Evaluation Table, "
                "and an explicit Critical Scientific Judgment section evaluating endogeneity or selection bias."
            )
        },
        stage_checklists=_build_default_stage_checklists(
            "social science and economics"
        ),
    )

    # 10. environmental_ecology
    registry["environmental_ecology"] = DisciplineProfile(
        name="environmental_ecology",
        default_actor_guidance=(
            "Apply Environmental and Ecology Science standards: specify ecological mechanisms, biodiversity metrics, "
            "field sampling protocols, spatiotemporal controls, and environmental impact assessments."
        ),
        actor_guidance={
            "generate": (
                "For candidate hypothesis generation ('generate' stage), EVERY candidate must strictly follow "
                "the gold-standard scientific structure of environmental/ecology benchmarks:\n"
                "1. Rich Technical Narrative: Detailed ecological mechanisms, biodiversity metrics, field sampling protocols, spatiotemporal controls, and environmental impact assessments.\n"
                "2. Motivation and Supporting Evidence: A dedicated subsection explaining the scientific rationale and citing empirical evidence from the knowledge base.\n"
                "3. Evaluation of Idea Table: A structured Markdown table ('Category | Description | Judgment') evaluating Ecological Relevance, Sampling Rigor, Biodiversity Metric Validity, Environmental Leap, and Field Feasibility.\n"
                "4. Critical Scientific Judgment: An explicit judgment paragraph balancing strengths against specific ecological complexity, seasonal variability, or sampling biases."
            )
        },
        critic_rubrics={
            "generate": (
                "Reject drafts that omit mandatory environmental/ecology rigor pillars: ecological mechanisms, "
                "biodiversity metrics, field sampling protocols, spatiotemporal controls, a structured Evaluation Table, "
                "and an explicit Critical Scientific Judgment section evaluating sampling biases."
            )
        },
        stage_checklists=_build_default_stage_checklists(
            "environmental and ecology science"
        ),
    )

    # 11. pharmacology_toxicology
    registry["pharmacology_toxicology"] = DisciplineProfile(
        name="pharmacology_toxicology",
        default_actor_guidance=(
            "Apply Pharmacology and Toxicology standards: specify pharmacokinetic/pharmacodynamic (PK/PD) models, "
            "dosing regimens, toxicity endpoints, assays, therapeutic index, and adverse reaction controls."
        ),
        actor_guidance={
            "generate": (
                "For candidate hypothesis generation ('generate' stage), EVERY candidate must strictly follow "
                "the gold-standard scientific structure of pharmacology/toxicology benchmarks:\n"
                "1. Rich Technical Narrative: Detailed pharmacokinetic/pharmacodynamic models, dosing regimens, toxicity endpoints, assays, and therapeutic index calculations.\n"
                "2. Motivation and Supporting Evidence: A dedicated subsection explaining the scientific rationale and citing empirical evidence from the knowledge base.\n"
                "3. Evaluation of Idea Table: A structured Markdown table ('Category | Description | Judgment') evaluating Pharmacodynamic Specificity, Dosing Feasibility, Toxicity Profile, Therapeutic Leap, and Safety Margin.\n"
                "4. Critical Scientific Judgment: An explicit judgment paragraph balancing strengths against specific metabolite toxicity, adverse drug reactions, or pharmacokinetic limitations."
            )
        },
        critic_rubrics={
            "generate": (
                "Reject drafts that omit mandatory pharmacology/toxicology rigor pillars: pharmacokinetic/pharmacodynamic models, "
                "dosing regimens, toxicity endpoints, assays, therapeutic index calculations, a structured Evaluation Table, "
                "and an explicit Critical Scientific Judgment section evaluating toxicity risks."
            )
        },
        stage_checklists=_build_default_stage_checklists("pharmacology and toxicology"),
    )

    # 12. general_interdisciplinary
    registry["general_interdisciplinary"] = DisciplineProfile(
        name="general_interdisciplinary",
        default_actor_guidance=(
            "Apply General Interdisciplinary Scientific standards: specify cross-disciplinary mechanisms, clear independent/dependent variables, "
            "domain-appropriate controls, calibration, and falsifiable validation protocols."
        ),
        actor_guidance={
            "generate": (
                "For candidate hypothesis generation ('generate' stage), EVERY candidate must strictly follow "
                "the gold-standard scientific structure of interdisciplinary scientific benchmarks:\n"
                "1. Rich Technical Narrative: Detailed cross-disciplinary mechanisms, clear independent/dependent variables, domain-appropriate controls, and falsifiable validation protocols.\n"
                "2. Motivation and Supporting Evidence: A dedicated subsection explaining the scientific rationale and citing empirical evidence from the knowledge base.\n"
                "3. Evaluation of Idea Table: A structured Markdown table ('Category | Description | Judgment') evaluating Mechanistic Clarity, Empirical Testability, Methodological Rigor, Interdisciplinary Leap, and Falsifiability.\n"
                "4. Critical Scientific Judgment: An explicit judgment paragraph balancing strengths against specific cross-disciplinary confounds, measurement limitations, or failure modes."
            )
        },
        critic_rubrics={
            "generate": (
                "Reject drafts that omit mandatory general interdisciplinary rigor pillars: clear mechanistic hypotheses, "
                "independent/dependent variables, domain-appropriate controls, falsifiable validation protocols, a structured Evaluation Table, "
                "and an explicit Critical Scientific Judgment section evaluating cross-disciplinary confounds."
            )
        },
        stage_checklists=_build_default_stage_checklists(
            "general interdisciplinary science"
        ),
    )

    return registry


DISCIPLINE_REGISTRY = _create_registry()


def get_discipline_profile(discipline: str) -> DisciplineProfile:
    """Retrieve the DisciplineProfile for a canonical discipline."""
    return DISCIPLINE_REGISTRY.get(
        discipline, DISCIPLINE_REGISTRY["general_interdisciplinary"]
    )


def classify_discipline(question: str) -> str:
    """Classify a research question into the canonical 12-discipline taxonomy."""
    text = question.lower()
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "pharmacology_toxicology",
            (
                "pharmacokinetic",
                "pharmacodynamic",
                "toxicity",
                "toxicology",
                "dosage",
                "adverse drug",
                "drug delivery",
                "drug resistance",
                "metabolite toxicity",
                "therapeutic index",
                "pharmacology",
                "toxic",
            ),
        ),
        (
            "neuroscience_cognitive",
            (
                "neuro",
                "brain",
                "synaptic",
                "cortex",
                "cognitive",
                "perception",
                "fmri",
                "eeg",
                "memory",
                "neural pathway",
                "cognition",
                "neuroscience",
                "neurological",
            ),
        ),
        (
            "astronomy_astrophysics",
            (
                "exoplanet",
                "galaxy",
                "supernova",
                "cosmology",
                "black hole",
                "stellar",
                "telescope",
                "astronomy",
                "astrophysics",
                "pulsar",
                "redshift",
                "interstellar",
                "planetary",
            ),
        ),
        (
            "earth_climate_sciences",
            (
                "climate modeling",
                "climate model",
                "climate",
                "oceanography",
                "geology",
                "seismic",
                "tectonic",
                "atmospheric",
                "meteorology",
                "glacier",
                "paleoclimate",
                "earthquake",
                "geophysical",
                "earth science",
            ),
        ),
        (
            "environmental_ecology",
            (
                "ecosystem",
                "biodiversity",
                "conservation",
                "pollution",
                "habitat",
                "extinction",
                "ecological",
                "biome",
                "food web",
                "deforestation",
                "ecology",
                "environmental",
            ),
        ),
        (
            "chemistry_materials",
            (
                "peptide synthesis",
                "peptide",
                "chemical synthesis",
                "organic synthesis",
                "inorganic synthesis",
                "polymer synthesis",
                "catalys",
                "polymer",
                "alloy",
                "crystal",
                "battery",
                "coating",
                "reagent",
                "chemical",
                "electrochem",
                "molecular doping",
                "ligand",
                "stereochem",
                "epimerization",
                "chemistry",
                "materials science",
            ),
        ),
        (
            "biology_medicine",
            (
                "crispr",
                "scrna-seq",
                "nsclc",
                "pd-1",
                "gene editing",
                "gene therapy",
                "gene expression",
                "gene regulation",
                "genes",
                "genetics",
                "genomic",
                "protein expression",
                "proteins",
                "protein structure",
                "cell ",
                "cells ",
                "cellular",
                "clinical",
                "disease",
                "tumor",
                "cancer",
                "immun",
                "vaccin",
                "pathogen",
                "observational cohort",
                "cohort",
                "biology",
                "medical",
                "biomarker",
                "in vitro",
                "in vivo",
            ),
        ),
        (
            "computer_science_ai",
            (
                "machine learning",
                "optimizer",
                "artificial intelligence",
                "deep learning",
                "neural network",
                "algorithm",
                "llm",
                "transformer",
                "dataset",
                "benchmark",
                "nlp",
                "computer vision",
                "reinforcement learning",
                "software",
                "ai",
                "computer science",
                "algorithmic",
            ),
        ),
        (
            "physics_engineering",
            (
                "quantum",
                "thermodynamics",
                "optics",
                "laser",
                "fluid dynamics",
                "aerodynamics",
                "semiconductor",
                "superconductor",
                "mechanical engineering",
                "circuit",
                "electromagnetic",
                "particle physics",
                "physics",
                "engineering",
            ),
        ),
        (
            "mathematics_statistics",
            (
                "theorem",
                "proof",
                "lemma",
                "topology",
                "algebra",
                "number theory",
                "combinatorics",
                "differential equation",
                "stochastic",
                "statistical inference",
                "axiom",
                "manifold",
                "mathematical",
                "statistics",
            ),
        ),
        (
            "social_science_economics",
            (
                "macroeconomic",
                "policy",
                "economics",
                "inflation",
                "monetary",
                "fiscal",
                "sociology",
                "voting",
                "labor market",
                "demographic",
                "economic",
                "social science",
                "econometric",
            ),
        ),
    )
    for disc, terms in rules:
        if any(term in text for term in terms):
            return disc
    return "general_interdisciplinary"
