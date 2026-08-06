"""The bounded dispatcher that fans one stage out across its specialists.

The bound is the interesting part. It exists so a stage that asks for ten
searches does not open ten sockets at once, and for most of this project's life
no stage asked for more than the bound allowed -- so the code that waits was
never once executed, and the defect it carried never surfaced.

Offline specialists answer without awaiting anything, so each one runs to
completion before the next is scheduled and nothing ever waits either. Both
tests here therefore make the specialists yield: without that they exercise the
uncontended path, which is the path that always worked.
"""

from __future__ import annotations

import asyncio

from coscientist.agents import SPECIALISTS_BY_STAGE, DeterministicProvider
from coscientist.collaboration import LocalA2ATaskBus
from coscientist.models import (
    Artifact,
    ArtifactStatus,
    Candidate,
    CandidatePopulation,
    Session,
)

# The reflect stage runs five reviewers, which is the widest fan-out available
# offline; the bound is lowered to two so the waiting path is actually taken.
REVIEWERS = SPECIALISTS_BY_STAGE["reflect"]
BOUND = 2


def _session() -> Session:
    """A session the reviewers have something to review."""
    session = Session(question="Can a coating extend cycle life?")
    population = CandidatePopulation(
        candidates=[
            Candidate(
                title="A conformal alumina coating suppresses decomposition.",
                claim="A conformal alumina coating suppresses electrolyte decomposition.",
                rationale="Surface passivation limits electrolyte reduction.",
                mechanism_model="Surface passivation limits electrolyte reduction.",
                validation_protocol="Coin cells against an uncoated control.",
                predictions=["Capacity fade halves over 500 cycles."],
                falsifier="Fade is unchanged at matched C-rate.",
            )
        ],
        comparison_criteria=["Novelty", "Testability"],
    )
    session.artifacts.append(
        Artifact(
            stage="generate",
            agent="generation",
            artifact_type="specialist_output",
            content="One candidate.",
            schema_name="CandidatePopulation",
            payload=population.model_dump(mode="json"),
            status=ArtifactStatus.ACCEPTED,
        )
    )
    return session


def _slow_bus() -> tuple[LocalA2ATaskBus, list[int]]:
    """A bus whose specialists yield, and a record of how many were ever in flight."""
    bus = LocalA2ATaskBus(REVIEWERS, DeterministicProvider(), max_concurrency=BOUND)
    live = [0, 0]  # in flight now, and the most ever in flight at once

    for service in bus.services.values():
        execute = service.execute

        async def yielding(session, *, _execute=execute, **kwargs):
            live[0] += 1
            live[1] = max(live[1], live[0])
            try:
                await asyncio.sleep(0)
                return await _execute(session, **kwargs)
            finally:
                live[0] -= 1

        service.execute = yielding

    return bus, live


def test_a_contended_dispatch_can_be_repeated_on_a_second_event_loop():
    """The workflow drives each stage through its own ``asyncio.run``.

    An ``asyncio.Semaphore`` binds itself to a loop the first time it has to make
    a waiter, and that binding is lazy: a bus whose stages never filled the bound
    never made one and never noticed. Discovery now fans out to ten against a
    bound of four, so the fifth caller waits, the semaphore binds to the loop
    that stage ran on, and the next stage died on "bound to a different event
    loop" -- one stage after the one that actually broke it.
    """
    bus, live = _slow_bus()
    session = _session()

    first = asyncio.run(bus.dispatch_stage(session, REVIEWERS, feedback="one"))
    second = asyncio.run(bus.dispatch_stage(session, REVIEWERS, feedback="two"))

    assert live[1] == BOUND, "nothing ever waited, so the binding was never made"
    assert len(first) == len(second) == len(REVIEWERS)
    assert {result.artifact.id for result in first}.isdisjoint(
        result.artifact.id for result in second
    )


def test_the_fan_out_never_exceeds_the_bound():
    """Ten concurrent grounded searches against one endpoint is what the bound is
    for, so a per-dispatch semaphore has to be one per dispatch and not one per
    specialist, which would be a bound in name only."""
    assert len(REVIEWERS) > BOUND, "the bound is not being exercised"
    bus, live = _slow_bus()

    asyncio.run(bus.dispatch_stage(_session(), REVIEWERS))

    assert live[1] == BOUND
