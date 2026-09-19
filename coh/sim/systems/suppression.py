"""suppression — decay, suppressed/pinned thresholds, and `Reinforce` progress (task 9).

Nearby-suppression *spill* happens at bullet time, inside
`coh/sim/systems/combat.py`'s `add_suppression` (it needs the weapon's
nearby-suppression mult/radius and the shooter's team, both only known
there). This module owns everything that happens once a tick rather than
once a bullet:

- decaying `squad.suppression` toward 0 and flipping `suppressed` / `pinned`
  with hysteresis (`SuppressionDef.suppress_at` / `suppress_recover` /
  `pin_at` / `pin_recover`);
- recovery rate: `recovery_per_s * cover_recovery_mult[cover vs the last
  attacker's direction]`, further multiplied by `noncombat_recovery_mult`
  once `noncombat_recovery_delay_s` has passed since `last_hit_tick`;
- advancing an in-progress `Reinforce` order: paying for and timing one
  model at a time (`orders.apply_reinforce` starts the first), adding the
  member when its timer elapses, and either continuing automatically (still
  missing members, still in range, still affordable) or stopping.

`Retreat` (`coh/sim/orders.py`) clears suppression outright and
`combat.add_suppression` refuses to accumulate more on a `RETREATING`
squad, so a retreating squad never becomes suppressed or pinned again while
it is running for its HQ.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from coh.maps.cover import cover_at
from coh.maps.format import cell_of
from coh.sim.constants import CELL_M, DT, TICKS_PER_SECOND
from coh.sim.state import Event, Member

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim
    from coh.sim.state import Squad


def run(sim: "Sim") -> None:
    """Advance the suppression system by one tick."""
    for sid in sorted(sim.state.squads):
        _update_squad(sim, sim.state.squads[sid])
    _advance_reinforce(sim)


# ---------------------------------------------------------------------------
# Decay + suppressed/pinned thresholds
# ---------------------------------------------------------------------------


def _update_squad(sim: "Sim", squad: "Squad") -> None:
    sdef = sim.data.squads.get(squad.def_id)
    if sdef is None or sdef.suppression is None or not squad.alive_members:
        return
    sup = sdef.suppression

    # Thresholds are evaluated against this tick's value (as combat left
    # it) *before* recovery ticks it down, so a bullet that lands exactly on
    # `pin_at` this tick actually registers as pinned this tick.
    if squad.pinned:
        if squad.suppression < sup.pin_recover:
            squad.pinned = False
    elif squad.suppression >= sup.pin_at:
        squad.pinned = True

    if squad.suppressed:
        if squad.suppression < sup.suppress_recover:
            squad.suppressed = False
    elif squad.suppression >= sup.suppress_at:
        squad.suppressed = True

    recovery = sup.recovery_per_s * _recovery_mult(sim, squad)
    squad.suppression = min(1.0, max(0.0, squad.suppression - recovery * DT))


def _recovery_mult(sim: "Sim", squad: "Squad") -> float:
    econ = sim.data.economy
    mult = econ.cover_recovery_mult.get(_recovery_cover(sim, squad), 1.0)
    if squad.last_hit_tick < 0 or (sim.state.tick - squad.last_hit_tick) * DT >= econ.noncombat_recovery_delay_s:
        mult *= econ.noncombat_recovery_mult
    return mult


def _recovery_cover(sim: "Sim", squad: "Squad") -> str:
    """The cover type recovery is computed against: `cover_at` toward the
    last attacker if the squad has been hit before, else `open`; a
    garrisoned squad uses `garrison` (falling back to `heavy` if the economy
    table has no `garrison` entry) regardless of any attacker direction."""
    if squad.garrison_in is not None:
        return "garrison" if "garrison" in sim.data.economy.cover_recovery_mult else "heavy"
    if squad.last_attacker_pos is None:
        return "open"
    from_pos = np.array(squad.last_attacker_pos, dtype=float)
    return cover_at(sim.map, cell_of(squad.pos, CELL_M), from_pos)


# ---------------------------------------------------------------------------
# Reinforce
# ---------------------------------------------------------------------------


def _advance_reinforce(sim: "Sim") -> None:
    from coh.sim import orders as orders_mod

    for sid in sorted(sim.state.squads):
        squad = sim.state.squads[sid]
        if not squad.reinforcing:
            continue
        if not squad.alive_members:
            squad.reinforcing = False
            continue

        sdef = sim.data.squads.get(squad.def_id)
        is_reinforce_order = isinstance(squad.order, orders_mod.Reinforce)
        if sdef is None or not is_reinforce_order or orders_mod.reinforce_building(sim, squad) is None:
            # Moving out of range, or a different order overwriting this
            # one, cancels further models: no refund for one in progress.
            squad.reinforcing = False
            if is_reinforce_order:
                squad.order = None
            continue

        if sim.state.tick < squad.reinforce_done_tick:
            continue
        _complete_reinforce_model(sim, squad, sdef)


def _complete_reinforce_model(sim: "Sim", squad: "Squad", sdef) -> None:
    from coh.sim import orders as orders_mod
    from coh.sim.systems import production

    slot = len(squad.members)
    weapon = sdef.loadout[slot] if slot < len(sdef.loadout) else ""
    squad.members.append(Member(hp=sdef.member_hp, weapon=weapon))
    sim.state.events.append(
        Event(
            kind="reinforced",
            tick=sim.state.tick,
            data={"squad": squad.id, "owner": squad.owner, "def_id": squad.def_id},
        )
    )

    if len(squad.members) >= sdef.members:
        squad.reinforcing = False
        squad.order = None
        return

    player = sim.state.players[squad.owner]
    cost = orders_mod.reinforce_model_cost(sim, sdef)
    if orders_mod.reinforce_building(sim, squad) is None or not production.can_afford(player, cost):
        squad.reinforcing = False
        squad.order = None
        return

    production.pay(player, cost)
    seconds = orders_mod.reinforce_model_time_s(sdef)
    squad.reinforce_done_tick = sim.state.tick + max(1, math.ceil(seconds * TICKS_PER_SECOND))
