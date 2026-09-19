"""victory — victory-point ticket drain and annihilation checks (task 15).

Every `ticket_interval_s`, each team drains tickets proportional to how far
behind it is on victory points (owned regardless of supply connection): if an
enemy team's VP count exceeds this team's by `lead > 0`, this team loses
`tickets_per_vp_lead * lead` tickets. A team is eliminated when its tickets
drop to zero or below, or when every HQ building belonging to one of its
players is destroyed (checked first, so annihilation always wins over a
tickets result computed the same tick). At the time limit the team with more
tickets wins; a tie at zero tickets or at the time limit is a draw
(`winner = -1`). Exactly two teams are supported, matching the ticket rule in
the design brief.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coh.sim.constants import DT, TICKS_PER_SECOND
from coh.sim.state import Event

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim

__all__ = ["run"]


def _teams(sim: "Sim") -> list[int]:
    return sorted(sim.state.tickets)


def _vp_count(sim: "Sim", team: int) -> int:
    return sum(
        1
        for pid, pdef in sim.map.points.items()
        if pdef.type == "victory" and sim.state.points[pid].owner_team == team
    )


def _team_hqs_alive(sim: "Sim", team: int) -> bool:
    hq_ids = sorted({p.hq_id for p in sim.state.players.values() if p.team == team})
    if not hq_ids:
        return True
    return any(hq_id in sim.state.buildings for hq_id in hq_ids)


def _end_game(sim: "Sim", winner: int, reason: str) -> None:
    sim.state.winner = winner
    sim.state.events.append(Event(kind="game_over", tick=sim.state.tick, data={"winner": winner, "reason": reason}))


def _drain_tickets(sim: "Sim", teams: list[int]) -> None:
    econ = sim.data.economy
    interval = round(econ.ticket_interval_s * TICKS_PER_SECOND)
    if interval <= 0 or (sim.state.tick + 1) % interval != 0:
        return
    vp_counts = {team: _vp_count(sim, team) for team in teams}
    for team in teams:
        enemy_max = max((vp_counts[t] for t in teams if t != team), default=0)
        lead = enemy_max - vp_counts[team]
        if lead > 0:
            sim.state.tickets[team] -= econ.tickets_per_vp_lead * lead


def _winner_by_tickets(sim: "Sim", team_a: int, team_b: int) -> int:
    tickets_a, tickets_b = sim.state.tickets[team_a], sim.state.tickets[team_b]
    if tickets_a > tickets_b:
        return team_a
    if tickets_b > tickets_a:
        return team_b
    return -1


def run(sim: "Sim") -> None:
    """Advance the victory system by one tick."""
    if sim.state.winner is not None:
        return

    teams = _teams(sim)
    if len(teams) != 2:
        return  # annihilation / ticket win rules are defined for two teams only
    team_a, team_b = teams

    # Annihilation is checked before tickets, so it always takes precedence
    # over a tickets result computed later the same tick.
    a_alive, b_alive = _team_hqs_alive(sim, team_a), _team_hqs_alive(sim, team_b)
    if not a_alive or not b_alive:
        if not a_alive and not b_alive:
            _end_game(sim, -1, "annihilation")
        else:
            _end_game(sim, team_b if not a_alive else team_a, "annihilation")
        return

    _drain_tickets(sim, teams)

    if sim.state.tickets[team_a] <= 0 or sim.state.tickets[team_b] <= 0:
        _end_game(sim, _winner_by_tickets(sim, team_a, team_b), "tickets")
        return

    ticks_elapsed = sim.state.tick + 1
    if ticks_elapsed * DT >= sim.config.time_limit_s:
        _end_game(sim, _winner_by_tickets(sim, team_a, team_b), "time_limit")
