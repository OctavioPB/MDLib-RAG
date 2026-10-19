"""Spaced repetition with FSRS-4.5 (ADR-025) over practised library sections.

A section's memory state (stability S in days, difficulty D in 1..10) is not stored: it is rebuilt
by replaying its answers in order, so every past answer counts and nothing can drift. Answers map
to FSRS ratings: wrong → Again (1), partial → Hard (2), correct → Good (3).

Formulas from the FSRS-4.5 algorithm (open-spaced-repetition):
  R(t, S) = (1 + F·t/S)^C            with C = -0.5, F = 19/81, so R(S, S) = 0.9
  I(r, S) = S/F · (r^(1/C) - 1)      the interval at which recall falls to r
  S0(G) = w[G-1];  D0(G) = w4 - (G-3)·w5
  D' = w7·D0(4) + (1-w7)·(D - w6·(G-3))                       (clamped to 1..10)
  S' success = S·(e^w8 · (11-D) · S^-w9 · (e^(w10·(1-R)) - 1) · hard · easy + 1)
  S' lapse   = w11 · D^-w12 · ((S+1)^w13 - 1) · e^(w14·(1-R))
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

DECAY = -0.5
FACTOR = 19 / 81
RATINGS = {"wrong": 1, "partial": 2, "correct": 3}  # Easy (4) is never given


@dataclass(frozen=True)
class MemoryState:
    stability: float  # days until recall falls to 90%
    difficulty: float  # 1 (easy) .. 10 (hard)
    last_review: datetime
    reviews: int
    lapses: int  # answers rated Again


class FSRS:
    def __init__(
        self, weights: Sequence[float], desired_retention: float, max_interval_days: int
    ) -> None:
        if len(weights) != 17:
            raise ValueError("FSRS-4.5 needs 17 parameters")
        self.w = list(weights)
        self.retention = desired_retention
        self.max_interval = max_interval_days

    def retrievability(self, elapsed_days: float, stability: float) -> float:
        return float((1 + FACTOR * max(elapsed_days, 0.0) / stability) ** DECAY)

    def interval_days(self, stability: float) -> float:
        days = float(stability / FACTOR * (self.retention ** (1 / DECAY) - 1))
        return min(max(days, 1.0), float(self.max_interval))

    def _initial_difficulty(self, rating: int) -> float:
        return self.w[4] - (rating - 3) * self.w[5]

    def first(self, rating: int) -> tuple[float, float]:
        stability = max(self.w[rating - 1], 0.1)
        return stability, min(max(self._initial_difficulty(rating), 1.0), 10.0)

    def next(
        self, stability: float, difficulty: float, elapsed_days: float, rating: int
    ) -> tuple[float, float]:
        w = self.w
        r = self.retrievability(elapsed_days, stability)
        moved = difficulty - w[6] * (rating - 3)
        new_d = w[7] * self._initial_difficulty(4) + (1 - w[7]) * moved
        new_d = min(max(new_d, 1.0), 10.0)
        if rating == 1:
            new_s = (
                w[11]
                * difficulty ** (-w[12])
                * ((stability + 1) ** w[13] - 1)
                * math.exp(w[14] * (1 - r))
            )
            new_s = min(new_s, stability)  # forgetting never makes memory stronger
        else:
            hard = w[15] if rating == 2 else 1.0
            easy = w[16] if rating == 4 else 1.0
            growth = (
                math.exp(w[8])
                * (11 - difficulty)
                * stability ** (-w[9])
                * (math.exp(w[10] * (1 - r)) - 1)
                * hard
                * easy
            )
            new_s = stability * (growth + 1)
        return max(new_s, 0.1), new_d

    def replay(self, answers: Sequence[tuple[datetime, str]]) -> MemoryState | None:
        """The memory state after these (when, result) answers, oldest first."""
        state: MemoryState | None = None
        for when, result in answers:
            rating = RATINGS[result]
            if state is None:
                s, d = self.first(rating)
            else:
                elapsed = (when - state.last_review).total_seconds() / 86400
                s, d = self.next(state.stability, state.difficulty, elapsed, rating)
            lapses = (state.lapses if state else 0) + (rating == 1)
            state = MemoryState(s, d, when, (state.reviews if state else 0) + 1, lapses)
        return state


@dataclass(frozen=True)
class SectionSchedule:
    key: tuple[str, str]  # (doc_path, heading_path JSON), as in the practice history
    state: MemoryState
    due: datetime
    recall_now: float  # estimated probability of recall at the time of the computation


def schedule(
    histories: dict[tuple[str, str], list[tuple[datetime, str]]], fsrs: FSRS, now: datetime
) -> dict[tuple[str, str], SectionSchedule]:
    out = {}
    for key, answers in histories.items():
        state = fsrs.replay(answers)
        if state is None:
            continue
        due = state.last_review + timedelta(days=fsrs.interval_days(state.stability))
        elapsed = (now - state.last_review).total_seconds() / 86400
        out[key] = SectionSchedule(key, state, due, fsrs.retrievability(elapsed, state.stability))
    return out


def due_today(
    schedules: dict[tuple[str, str], SectionSchedule], now: datetime
) -> list[SectionSchedule]:
    """Sections whose review date is today or past, the most likely to be forgotten first."""
    end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=0)
    due = [s for s in schedules.values() if s.due <= end_of_day]
    return sorted(due, key=lambda s: (s.recall_now, s.due))
