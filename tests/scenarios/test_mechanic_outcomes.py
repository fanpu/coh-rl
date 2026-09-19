"""Does the sim reproduce CoH's signature engagements?

Six set-piece fights, each on a small purpose-built inline map, each run over
30 seeds, each asserting that the side CoH would favour wins at least 70% of
the time. These are outcome tests, not unit tests: they do not care *which*
rule produced the result, only that suppression, firing arcs, cover,
garrisons and armour add up to the fight a player would expect.

**These run on the fixture tables, not the real CoH1 ones.** The real tables
(`coh/data/tables/`) do not exist yet, so the numbers here are the
hand-written fixture numbers. Re-pointing this file at real data is meant to
be two edits: swap the def ids in `SCENARIO_UNITS`, and point the `data`
fixture at the real loader. Everything below goes through `SCENARIO_UNITS`
and `data` and never names a fixture id directly.

Two scenarios needed units the fixture tables do not contain at all. Both
are pre-existing *additive* test-only fixtures from earlier tasks rather than
edits to any shipped number:

- the AT-gun duel (scenario 3) uses `at_team_duel` / `duel_tank`, which task
  11 added precisely so that an AT-gun-vs-tank duel has a decidable winner in
  each starting geometry;
- the garrison duel (scenario 4) uses `garrison_rifles`, which task 12 added
  so that the fight is squad-vs-squad rather than both sides shooting at the
  house (the plain fixture rifle would rather knock the building down).

No sourced value was changed to make any of these pass.
"""

from __future__ import annotations

import math

import pytest

from coh.maps.format import center_of
from coh.sim.constants import CELL_M, TICKS_PER_SECOND
from coh.sim.orders import AttackMove
from coh.sim.sim import PlayerSetup, SimConfig
from coh.sim.state import SquadState
from coh.sim.systems import garrison as garrison_sys
from tests.helpers import fixture_data, make_sim

# --- the one table to swap when the real stat tables land -------------------

SCENARIO_UNITS = {
    "rifles": "rifles",  # the baseline rifle squad
    "hmg": "hmg_team",  # the heavy machine gun team (90-degree arc, setup time)
    "at_gun": "at_team_duel",  # the AT gun of scenario 3 (additive fixture)
    "at_gun_target": "duel_tank",  # the tank it duels (additive fixture)
    "tank": "tank",  # the tank rifles cannot hurt (scenario 6)
    "garrison_rifles": "garrison_rifles",  # rifles that shoot men, not houses
    "house": "house",  # an enterable neutral building
}

SEEDS = range(30)
WIN_RATE = 0.70

EAST, WEST, NORTH = 0.0, math.pi, -math.pi / 2

# The arena: long enough that both HQs (and the builder squads that spawn
# under them) sit well outside every weapon's reach, so the only fight is the
# one each scenario sets up.
WIDTH, HEIGHT = 70, 24
LINE = 12  # the row every engagement is fought along
STARTS = [
    {"slot": 0, "team": 0, "hq_cell": [1, 1], "sector": "a"},
    {"slot": 1, "team": 1, "hq_cell": [WIDTH - 6, HEIGHT - 6], "sector": "a"},
]
POINTS = [{"id": "mid", "name": "Middle", "type": "victory", "cell": [WIDTH // 2, 2]}]


@pytest.fixture(scope="module")
def data():
    """The stat tables these scenarios are fought on."""
    return fixture_data()


# ---------------------------------------------------------------------------
# Arena helpers
# ---------------------------------------------------------------------------


def _terrain(heavy_cover: tuple[tuple[int, int], ...] = ()) -> list[str]:
    """All-open rows, with a patch of `c` (heavy area cover) at each cell given.

    A patch rather than a single cell: a squad's models stand on their
    formation offsets, and `combat` reads each model's *own* cell's cover, so
    a one-cell crater would leave most of the squad in the open.
    """
    grid = [["."] * WIDTH for _ in range(HEIGHT)]
    for cx, cy in heavy_cover:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if 0 <= cx + dx < WIDTH and 0 <= cy + dy < HEIGHT:
                    grid[cy + dy][cx + dx] = "c"
    return ["".join(row) for row in grid]


def arena(
    seed: int,
    data,
    *,
    heavy_cover: tuple[tuple[int, int], ...] = (),
    houses: tuple[tuple[int, int], ...] = (),
):
    """A two-player sim on an empty field, with optional cover and houses."""
    return make_sim(
        _terrain(heavy_cover),
        seed=seed,
        players=[PlayerSetup("us", 0, 0), PlayerSetup("wehr", 1, 1)],
        data=data,
        config=SimConfig(time_limit_s=10_000.0),
        starts=STARTS,
        points=POINTS,
        neutral_buildings=[{"def": SCENARIO_UNITS["house"], "cell": list(cell)} for cell in houses],
    )


def place(sim, owner: int, unit: str, cell: tuple[int, int], *, heading: float = EAST):
    """Spawn a squad at `cell`, already facing `heading`.

    A team weapon is put straight into `SET_UP`: these scenarios are about
    what a *deployed* gun does, not about how long it takes to unpack.
    """
    squad = sim.spawn_squad(owner, SCENARIO_UNITS[unit], center_of(cell, CELL_M))
    squad.heading = heading
    if squad.state in (SquadState.SETTING_UP, SquadState.SET_UP):
        squad.state = SquadState.SET_UP
        squad.facing = heading
    return squad


def garrison(sim, squad, cell: tuple[int, int]):
    """Put `squad` inside the neutral building whose top-left cell is `cell`."""
    building = next(b for b in sim.state.buildings.values() if b.neutral and b.cell == cell)
    garrison_sys.enter(sim, squad, building)
    return building


def play(sim, seconds: float) -> None:
    for _ in range(int(round(seconds * TICKS_PER_SECOND))):
        sim.tick()


def alive(sim, squad) -> bool:
    """Is this squad still a fighting unit (not wiped, not an abandoned gun)?"""
    live = sim.state.squads.get(squad.id)
    return live is not None and live.has_alive_members and not live.abandoned


def beaten(sim, squad) -> bool:
    """Wiped out, or pinned down and no longer able to shoot back."""
    live = sim.state.squads.get(squad.id)
    return live is None or not live.has_alive_members or live.abandoned or live.pinned


def hp_frac(sim, squad, data) -> float:
    live = sim.state.squads.get(squad.id)
    if live is None:
        return 0.0
    sdef = data.squads[live.def_id]
    return sum(m.hp for m in live.members if m.hp > 0) / (sdef.members * sdef.member_hp)


def win_rate(outcomes: list[bool]) -> float:
    return sum(outcomes) / len(outcomes)


def assert_wins(outcomes: list[bool], who: str) -> None:
    rate = win_rate(outcomes)
    assert rate >= WIN_RATE, f"{who} won {rate:.0%} of {len(outcomes)} seeds (need {WIN_RATE:.0%})"


# ---------------------------------------------------------------------------
# 1. An HMG in heavy cover stops a frontal assault across open ground
# ---------------------------------------------------------------------------

HMG_CELL = (30, LINE)
FRONTAL_CELL = (48, LINE)  # 36 m east: outside the HMG's reach until it closes


def _hmg_frontal(seed: int, data) -> bool:
    sim = arena(seed, data, heavy_cover=(HMG_CELL,))
    hmg = place(sim, 0, "hmg", HMG_CELL, heading=EAST)
    rifles = place(sim, 1, "rifles", FRONTAL_CELL)
    sim.issue(1, [AttackMove(rifles.id, (HMG_CELL[0] + 1, LINE))])
    play(sim, 90.0)
    return beaten(sim, rifles) and alive(sim, hmg)


@pytest.mark.slow
def test_hmg_in_cover_beats_a_frontal_rifle_assault(data):
    assert_wins([_hmg_frontal(seed, data) for seed in SEEDS], "the HMG")


# ---------------------------------------------------------------------------
# 2. ... and loses to the same assault once someone comes round its arc
# ---------------------------------------------------------------------------

# The frontal squad starts inside the gun's *sight* (25 m) as well as its
# arc, and in heavy cover, so the gun has a target it is committed to and
# cannot chew through in seconds. Both matter: from beyond sight range the
# gun would see nothing in its arc and legitimately swing onto the flanker
# instead, and from open ground the frontal squad dies fast enough that the
# gun is free to turn anyway. Neither is the mechanic under test, which is
# that a gun already engaged to its front cannot answer its flank.
PINNED_DOWN_CELL = (41, LINE)  # 22 m east: in sight, in the arc, in heavy cover
FLANK_CELL = (HMG_CELL[0], LINE - 4)  # due north: 90 degrees off an east-facing arc


def _hmg_flanked(seed: int, data) -> bool:
    sim = arena(seed, data, heavy_cover=(HMG_CELL, PINNED_DOWN_CELL))
    hmg = place(sim, 0, "hmg", HMG_CELL, heading=EAST)
    frontal = place(sim, 1, "rifles", PINNED_DOWN_CELL)
    flanker = place(sim, 1, "rifles", FLANK_CELL, heading=math.pi / 2)
    sim.issue(1, [AttackMove(frontal.id, (HMG_CELL[0] + 1, LINE))])
    play(sim, 90.0)
    return not alive(sim, hmg) and (alive(sim, frontal) or alive(sim, flanker))


@pytest.mark.slow
def test_two_rifle_squads_beat_the_hmg_by_flanking_its_arc(data):
    assert_wins([_hmg_flanked(seed, data) for seed in SEEDS], "the riflemen")


# ---------------------------------------------------------------------------
# 3. An AT gun owns the ground it is pointing at, and nothing else
# ---------------------------------------------------------------------------

GUN_CELL = (15, LINE)
TANK_FRONT_CELL = (37, LINE)  # 44 m east of the gun: long range for both


def _at_duel(seed: int, data, gun_facing: float) -> str:
    sim = arena(seed, data)
    gun = place(sim, 0, "at_gun", GUN_CELL, heading=gun_facing)
    tank = place(sim, 1, "at_gun_target", TANK_FRONT_CELL, heading=WEST)
    for _ in range(int(100 * TICKS_PER_SECOND)):
        sim.tick()
        if not alive(sim, tank):
            return "gun"
        if not alive(sim, gun):
            return "tank"
    return "neither"


@pytest.mark.slow
def test_an_at_gun_kills_a_tank_that_drives_into_its_arc(data):
    outcomes = [_at_duel(seed, data, gun_facing=EAST) == "gun" for seed in SEEDS]
    assert_wins(outcomes, "the AT gun")


@pytest.mark.slow
def test_a_tank_that_comes_from_behind_kills_the_at_gun(data):
    outcomes = [_at_duel(seed, data, gun_facing=WEST) == "tank" for seed in SEEDS]
    assert_wins(outcomes, "the tank")


# ---------------------------------------------------------------------------
# 4. A garrison beats identical rifles caught in the open
# ---------------------------------------------------------------------------

HOUSE_CELL = (34, LINE)
OPEN_ATTACKER_CELL = (44, LINE)  # ~20 m from the house: medium range


def _garrison_duel(seed: int, data) -> bool:
    sim = arena(seed, data, houses=(HOUSE_CELL,))
    defenders = place(sim, 0, "garrison_rifles", HOUSE_CELL)
    garrison(sim, defenders, HOUSE_CELL)
    attackers = place(sim, 1, "garrison_rifles", OPEN_ATTACKER_CELL, heading=WEST)
    play(sim, 90.0)
    return alive(sim, defenders) and not alive(sim, attackers)


@pytest.mark.slow
def test_a_garrisoned_squad_beats_identical_rifles_in_the_open(data):
    assert_wins([_garrison_duel(seed, data) for seed in SEEDS], "the garrison")


# ---------------------------------------------------------------------------
# 5. Heavy cover beats no cover at long range
# ---------------------------------------------------------------------------

COVER_CELL = (34, LINE)
# 22 m: the rifle's far band (> 20 m) for both sides, and still inside the
# rifle squad's 25 m sight -- at 28 m neither squad can see the other at all
# and the "duel" is two squads standing in a field.
LONG_RANGE_CELL = (45, LINE)


def _cover_duel(seed: int, data) -> bool:
    sim = arena(seed, data, heavy_cover=(COVER_CELL,))
    defenders = place(sim, 0, "rifles", COVER_CELL)
    attackers = place(sim, 1, "rifles", LONG_RANGE_CELL, heading=WEST)
    play(sim, 120.0)
    return alive(sim, defenders) and not alive(sim, attackers)


@pytest.mark.slow
def test_heavy_cover_beats_open_ground_at_long_range(data):
    assert_wins([_cover_duel(seed, data) for seed in SEEDS], "the squad in cover")


# ---------------------------------------------------------------------------
# 6. Rifles cannot hurt a tank
# ---------------------------------------------------------------------------

TANK_CELL = (34, LINE)
RIFLES_VS_TANK_CELL = (38, LINE)  # 8 m: point blank, the best rifles can do
MAX_TANK_HP_LOSS = 0.05


def _tank_hp_left(seed: int, data) -> float:
    sim = arena(seed, data)
    tank = place(sim, 0, "tank", TANK_CELL, heading=EAST)
    place(sim, 1, "rifles", RIFLES_VS_TANK_CELL, heading=WEST)
    play(sim, 60.0)
    return hp_frac(sim, tank, data)


@pytest.mark.slow
def test_rifles_barely_scratch_a_tank_in_a_minute(data):
    remaining = [_tank_hp_left(seed, data) for seed in SEEDS]
    worst = min(remaining)
    assert worst >= 1.0 - MAX_TANK_HP_LOSS, f"the tank lost {1 - worst:.1%} of its HP in 60 s"
