from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from biblio.config import Config
from biblio.practice.srs import FSRS, due_today, schedule

T0 = datetime(2026, 10, 1, 10, 0).astimezone()


@pytest.fixture
def fsrs(config: Config) -> FSRS:
    p = config.practice
    return FSRS(p.fsrs_weights, p.desired_retention, p.max_interval_days)


def test_recall_is_ninety_percent_after_one_stability(fsrs: FSRS) -> None:
    assert fsrs.retrievability(10, 10) == pytest.approx(0.9)
    assert fsrs.retrievability(0, 10) == 1.0
    assert fsrs.interval_days(10) == pytest.approx(10)  # desired retention 0.9
    assert fsrs.interval_days(0.01) == 1.0  # at least a day


def test_intervals_grow_with_success_and_shrink_after_a_lapse(fsrs: FSRS) -> None:
    answers: list[tuple[datetime, str]] = []
    when = T0
    intervals = []
    for result in ["correct", "correct", "correct", "wrong"]:
        answers.append((when, result))
        state = fsrs.replay(answers)
        assert state is not None
        intervals.append(fsrs.interval_days(state.stability))
        when += timedelta(days=intervals[-1])
    assert intervals[0] < intervals[1] < intervals[2]  # ~4, ~14, ~47 days
    assert intervals[3] < intervals[2] / 4  # a lapse: a much shorter interval, though
    # FSRS keeps part of a strong memory (~5 days after 47, not back to the first ~4)
    assert state.lapses == 1 and state.reviews == 4
    partial = fsrs.replay([(T0, "partial")])
    good = fsrs.replay([(T0, "correct")])
    assert partial and good and partial.stability < good.stability
    assert partial.difficulty > good.difficulty


def test_due_today_puts_the_most_forgotten_first(fsrs: FSRS) -> None:
    now = T0 + timedelta(days=30)
    histories = {
        ("a.md", "[1]"): [(T0, "correct")],  # due after ~4 days: long overdue
        ("b.md", "[2]"): [(now - timedelta(days=5), "wrong")],  # due after 1 day
        ("c.md", "[3]"): [(now - timedelta(hours=2), "correct")],  # just answered
    }
    schedules = schedule(histories, fsrs, now)
    due = due_today(schedules, now)
    assert [s.key[0] for s in due] == ["b.md", "a.md"] or [s.key[0] for s in due] == [
        "a.md",
        "b.md",
    ]
    assert due[0].recall_now <= due[1].recall_now  # most at risk first
    assert schedules[("c.md", "[3]")].recall_now > 0.95
    assert ("c.md", "[3]") not in {s.key for s in due}
