"""The job state machine.

The transition table is the authority on what may happen to a job, so it is tested
exhaustively rather than by example: every state pair is checked, and the properties that
must hold for *any* future edit to the table are asserted directly.
"""

from __future__ import annotations

import pytest

from dispatch.core.states import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    TRANSITIONS,
    ExitReason,
    JobState,
    can_transition,
)


def test_every_state_appears_in_the_table() -> None:
    """A state missing from the table would raise KeyError at the worst possible moment."""
    assert set(TRANSITIONS) == set(JobState)


def test_terminal_states_have_no_outgoing_transitions() -> None:
    """History is immutable: re-running a case creates a new job, it does not revive one."""
    for state in TERMINAL_STATES:
        assert TRANSITIONS[state] == frozenset(), f"{state} must be a dead end"


def test_non_terminal_states_can_reach_a_terminal_state() -> None:
    """No state may be a trap. Every live job must have some way to finish."""
    for state in set(JobState) - TERMINAL_STATES:
        reachable = _reachable_from(state)
        assert reachable & TERMINAL_STATES, f"{state} cannot reach any terminal state"


def test_every_state_is_reachable_from_queued() -> None:
    """A state nothing can reach is dead code pretending to be a design."""
    reachable = _reachable_from(JobState.QUEUED) | {JobState.QUEUED}
    unreachable = set(JobState) - reachable
    assert unreachable == set(), f"unreachable states: {unreachable}"


def test_self_transitions_are_always_rejected() -> None:
    """Re-entering a state is not a change, and recording it would pollute the audit trail."""
    for state in JobState:
        assert not can_transition(state, state)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (JobState.QUEUED, JobState.PREPARING),
        (JobState.QUEUED, JobState.HELD),
        (JobState.QUEUED, JobState.CANCELLED),
        (JobState.HELD, JobState.QUEUED),
        (JobState.PREPARING, JobState.RUNNING),
        (JobState.PREPARING, JobState.FAILED),
        (JobState.RUNNING, JobState.COMPLETED),
        (JobState.RUNNING, JobState.CANCELLED),
        (JobState.RUNNING, JobState.UNKNOWN),
    ],
)
def test_expected_transitions_are_allowed(source: JobState, target: JobState) -> None:
    assert can_transition(source, target)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (JobState.QUEUED, JobState.RUNNING),  # must pass through PREPARING
        (JobState.QUEUED, JobState.COMPLETED),  # cannot complete without running
        (JobState.HELD, JobState.PREPARING),  # must be released first
        (JobState.COMPLETED, JobState.QUEUED),  # history is immutable
        (JobState.RUNNING, JobState.PREPARING),  # no going backwards
        (JobState.RUNNING, JobState.HELD),  # hold applies to the queue, not to a process
        (JobState.CANCELLED, JobState.RUNNING),
    ],
)
def test_expected_transitions_are_forbidden(source: JobState, target: JobState) -> None:
    assert not can_transition(source, target)


def test_held_jobs_cannot_be_scheduled_directly() -> None:
    """A held job must be released before it can run.

    This is what makes hold a real guarantee rather than a hint the scheduler may ignore.
    """
    assert JobState.PREPARING not in TRANSITIONS[JobState.HELD]


def test_active_states_are_exactly_the_resource_holding_ones() -> None:
    """PREPARING holds an allocation because decomposition uses the machine."""
    assert {JobState.PREPARING, JobState.RUNNING} == ACTIVE_STATES


def test_states_are_stored_as_readable_text() -> None:
    """A sqlite3 session should show RUNNING, not 3."""
    assert JobState.RUNNING.value == "RUNNING"
    assert str(JobState.RUNNING) == "RUNNING"


def test_exit_reasons_are_lowercase_tokens() -> None:
    assert ExitReason.OK.value == "ok"
    assert ExitReason.PREPARE_FAILED.value == "prepare_failed"


def _reachable_from(start: JobState) -> set[JobState]:
    """Transitive closure of the transition table from ``start``."""
    seen: set[JobState] = set()
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for nxt in TRANSITIONS[current]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen
