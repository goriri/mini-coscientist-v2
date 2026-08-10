import pytest

from coscientist.agents import DeterministicProvider, Specialist
from coscientist.disciplines import (
    DISCIPLINE_REGISTRY,
    DISCIPLINE_TAXONOMY,
    DISCIPLINES,
    DisciplineProfile,
    classify_discipline,
    get_discipline_profile,
)
from coscientist.models import STAGES, Session
from coscientist.orchestration import CoScientistWorkflow


@pytest.mark.parametrize(
    ("question", "expected_discipline"),
    [
        (
            "Design exact fragmentation points for a hydrophobic 45-mer peptide synthesis.",
            "chemistry_materials",
        ),
        (
            "Identify scRNA-seq clusters and CRISPR editing efficacy in cancer cells.",
            "biology_medicine",
        ),
        (
            "Benchmark a machine learning optimizer algorithm on a large dataset.",
            "computer_science_ai",
        ),
        (
            "Develop a climate modeling simulation of oceanography and atmospheric dynamics.",
            "earth_climate_sciences",
        ),
        (
            "Study macroeconomic policy effects on monetary inflation and fiscal labor markets.",
            "social_science_economics",
        ),
        (
            "Prove a theorem in differential geometry and topology.",
            "mathematics_statistics",
        ),
        (
            "Study quantum superconductivity in optoelectronic circuits.",
            "physics_engineering",
        ),
        (
            "Investigate synaptic plasticity and EEG signals in visual cortex.",
            "neuroscience_cognitive",
        ),
        (
            "Measure redshift of distant exoplanetary atmosphere with a space telescope.",
            "astronomy_astrophysics",
        ),
        (
            "Assess ecosystem biodiversity conservation and pollution in rainforests.",
            "environmental_ecology",
        ),
        (
            "Evaluate pharmacokinetic toxicity assay and dosing for drug delivery.",
            "pharmacology_toxicology",
        ),
        (
            "Conduct a general interdisciplinary study of scientific methodologies.",
            "general_interdisciplinary",
        ),
    ],
)
def test_discipline_classification(question: str, expected_discipline: str):
    assert classify_discipline(question) == expected_discipline
    session = Session(question=question)
    assert session.discipline == expected_discipline
    flow = CoScientistWorkflow(question)
    assert flow.session.discipline == expected_discipline


def test_taxonomy_and_registry_completeness():
    assert len(DISCIPLINE_TAXONOMY) == 12
    assert DISCIPLINES == DISCIPLINE_TAXONOMY
    for discipline in DISCIPLINE_TAXONOMY:
        assert discipline in DISCIPLINE_REGISTRY
        profile = get_discipline_profile(discipline)
        assert isinstance(profile, DisciplineProfile)
        assert profile.name == discipline
        for stage in STAGES:
            checklist = profile.get_stage_checklist(stage)
            assert checklist, f"Missing checklist for {discipline} in stage {stage}"


def test_discipline_dynamic_prompt_loading():
    # 1. Test Computer Science / AI (non-chemical/non-biological discipline)
    cs_question = (
        "Benchmark a machine learning optimizer algorithm on an image dataset."
    )
    cs_session = Session(question=cs_question, discipline="computer_science_ai")
    specialist = Specialist("generate", "generation", "Generate candidate hypotheses")

    actor_prompt = specialist._build_actor_prompt(
        cs_session, "- general check", "prior work", "feedback"
    )
    assert "Scientific discipline: computer_science_ai" in actor_prompt
    assert "algorithmic complexity" in actor_prompt
    assert "benchmark dataset specifications" in actor_prompt
    assert (
        "Algorithmic Complexity, Dataset Diversity, Baseline Superiority"
        in actor_prompt
    )
    # Verify no hardcoded chemistry/biology bias
    assert "precursors or additives" not in actor_prompt
    assert "epimerization" not in actor_prompt
    assert "stereochemical integrity" not in actor_prompt
    assert "ligation strategy" not in actor_prompt
    assert "fragmentation points" not in actor_prompt
    assert "cohort designs" not in actor_prompt

    critic_prompt = specialist._build_critic_prompt(
        cs_session, "draft content", 1, "- general check"
    )
    assert "Scientific Discipline: computer_science_ai" in critic_prompt
    assert "data leakage controls" in critic_prompt
    assert "computational limits" in critic_prompt
    # Verify no hardcoded chemistry/biology bias in critic rubric
    assert "precursors or additives" not in critic_prompt
    assert "epimerization" not in critic_prompt
    assert "stereochemical integrity" not in critic_prompt
    assert "fragmentation points" not in critic_prompt
    assert "cohort designs" not in critic_prompt

    # 2. Test Mathematics / Statistics (non-chemical/non-biological discipline)
    math_session = Session(
        question="Prove a theorem in differential geometry.",
        discipline="mathematics_statistics",
    )
    math_actor_prompt = specialist._build_actor_prompt(
        math_session, "- check", "prior", ""
    )
    assert "Scientific discipline: mathematics_statistics" in math_actor_prompt
    assert "axiomatic rigor" in math_actor_prompt
    assert "proof sketches" in math_actor_prompt
    assert "precursors or additives" not in math_actor_prompt
    assert "cohort designs" not in math_actor_prompt

    math_critic_prompt = specialist._build_critic_prompt(
        math_session, "content", 1, "- check"
    )
    assert "Scientific Discipline: mathematics_statistics" in math_critic_prompt
    assert "proof gaps" in math_critic_prompt
    assert "precursors or additives" not in math_critic_prompt

    # 3. Test Chemistry / Materials (verifying domain-specific chemistry prompt is preserved for Chemistry)
    chem_session = Session(
        question="Design exact fragmentation points for peptide synthesis.",
        discipline="chemistry_materials",
    )
    chem_actor_prompt = specialist._build_actor_prompt(
        chem_session, "- check", "prior", ""
    )
    assert "Scientific discipline: chemistry_materials" in chem_actor_prompt
    assert "precursors or additives" in chem_actor_prompt
    assert "epimerization or racemization where there are stereocentres" in (
        chem_actor_prompt
    )

    # 4. Test Specialist.run() during scope stage automatically classifies a general session
    unclassified_session = Session(question=cs_question)
    unclassified_session.discipline = "general_interdisciplinary"
    scope_specialist = Specialist("scope", "goal_manager", "Manage goals")
    provider = DeterministicProvider()
    scope_specialist.run(unclassified_session, provider)
    assert unclassified_session.discipline == "computer_science_ai"


def test_a_peptide_construct_is_an_example_and_not_a_pillar_the_critic_demands():
    """The chemistry pillars have to fit a thin film as well as a peptide.

    They were written off the peptide benchmark and named its constructs outright,
    and the generate critic rejected any draft that left them out. A live run on ALD
    coatings for NMC811 cathodes duly supplied them: "Strict epimerization controls
    -- in this solid-state context, precise precursor pulsing to prevent uneven film
    growth", for an amorphous oxide with no stereocentre to invert, and a row reading
    "Ligation Strategy | ALD coating adhesion under mechanical stress", for a surface
    with nothing being joined.
    """
    profile = get_discipline_profile("chemistry_materials")
    actor = profile.get_actor_guidance("generate")
    critic = profile.get_critic_rubric("generate")

    # The five-row table is still mandatory, and the peptide categories are still
    # named -- as what a molecular synthesis typically fills them with.
    assert "structured Markdown table" in actor
    assert "Aggregation Control, Stereochemical Integrity and Ligation Strategy" in (
        actor
    )
    assert "For a molecular synthesis those are typically" in actor
    assert "film conformality" in actor
    assert "never carry over a category the system has no instance of" in actor

    # Nothing demands them of a system that has no instance of them, and the critic
    # now rejects each one used outside its domain rather than requiring it.
    assert "epimerization controls, stereochemical integrity" not in critic
    assert "Reject as a rigor failure" in critic
    assert "stereochemical integrity claimed for a system with no stereocentre" in (
        critic
    )
    assert "a ligation strategy for one with nothing being joined" in critic
    assert "fragmentation points where no species is being fragmented" in critic
    assert "a biophysical risk in a system with no biology in it" in critic


def test_the_generate_rubric_does_not_reach_a_reviewer_of_reviews():
    """Each stage's critic gets its own stage's rubric, and none where none is set.

    Every profile used to carry ``default_critic_rubric`` as a verbatim copy of its
    generate rubric, so the reflect critic was handed the hypothesis contract and
    told to reject a review that omits "a structured Evaluation Table". The
    reviewers complied: fourteen review findings on a live run arrived as a Markdown
    table inside a prose field.
    """
    session = Session(
        question="Design exact fragmentation points for peptide synthesis.",
        discipline="chemistry_materials",
    )
    profile = get_discipline_profile("chemistry_materials")
    assert profile.default_critic_rubric == ""
    assert "Evaluation Table" in profile.get_critic_rubric("generate")
    assert profile.get_critic_rubric("reflect") == ""

    generating = Specialist("generate", "generation", "Generate candidate hypotheses")
    generate_prompt = generating._build_critic_prompt(session, "draft", 1, "- check")
    assert "Domain Quality Rubric & Rigor Pillars (chemistry_materials)" in (
        generate_prompt
    )
    assert "structured Evaluation Table" in generate_prompt
    assert "4. Epistemic Integrity" in generate_prompt

    reflecting = Specialist("reflect", "reflection", "Review candidate hypotheses")
    reflect_prompt = reflecting._build_critic_prompt(session, "draft", 1, "- check")
    assert "Evaluation Table" not in reflect_prompt
    assert "Domain Quality Rubric" not in reflect_prompt
    # Renumbered, not left as an empty heading over a blank line.
    assert "3. Epistemic Integrity" in reflect_prompt
    assert "4." not in reflect_prompt
    # The discipline still reaches this critic, by the checklist that is scoped to
    # its stage rather than by a rubric copied off another one.
    assert "chemistry_materials" in reflect_prompt
