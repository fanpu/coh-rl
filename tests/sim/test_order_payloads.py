"""`Sim.issue` never raises, whatever an agent puts in an order payload.

`coh/sim/orders.py` promises (global constraint) that an invalid order comes
back as `OrderResult(ok=False, ...)` and is counted — it never propagates an
exception into the env loop. Before the payload-shape gate, an agent that
emitted `Move(sq, (1, 2, 3))` or `Capture(sq, [])` crashed the match, and
`SetFacing(sq, nan)` was *accepted* and permanently broke `state_hash()`.

This module is the fuzz net over that promise: a matrix of malformed payloads
for every order type, issued directly and round-tripped through
`order_from_dict`.
"""

from __future__ import annotations

import math

import pytest

from coh.sim import orders as orders_mod
from coh.sim.orders import (
    ORDER_TYPES,
    Attack,
    AttackMove,
    Build,
    BuyUpgrade,
    Capture,
    Garrison,
    Move,
    Reinforce,
    Research,
    Retreat,
    SetFacing,
    Stop,
    Train,
    Ungarrison,
    order_from_dict,
    order_to_dict,
)
from tests.helpers import make_sim, spawn

# Values that are wrong for *every* field type, so the same list can be swept
# across every field of every order.
JUNK = (
    None,
    True,
    False,
    3.5,
    "nonsense",
    (1, 2, 3),
    (1.5, 2.5),
    ("a", "b"),
    [],
    [1],
    [1, 2, 3],
    {},
    {"cell": 1},
    set(),
    float("nan"),
    float("inf"),
    float("-inf"),
    object(),
    b"bytes",
    -(2**70),
)


def _sim_with_entities():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    weapon = spawn(sim, 0, "hmg_team", (11, 10))
    hq = sim.state.buildings[sim.state.players[0].hq_id]
    return sim, squad, weapon, hq


def _well_formed(sim, squad, weapon, hq):
    """One valid-shaped instance of every order type (not all are *legal*)."""
    return [
        Move(squad.id, (12, 12)),
        AttackMove(squad.id, (12, 12)),
        Attack(squad.id, 999),
        Capture(squad.id, "mid"),
        Garrison(squad.id, hq.id),
        Ungarrison(squad.id),
        Retreat(squad.id),
        Reinforce(squad.id),
        SetFacing(weapon.id, 90.0),
        Build(squad.id, "barracks", (12, 12)),
        Train(hq.id, "rifles"),
        Research(hq.id, "nonexistent_upgrade"),
        BuyUpgrade(squad.id, "nonexistent_upgrade"),
        Stop(squad.id),
    ]


def test_the_matrix_covers_every_order_type():
    """Guard against this file quietly testing a subset of the orders."""
    sim, squad, weapon, hq = _sim_with_entities()
    covered = {type(order) for order in _well_formed(sim, squad, weapon, hq)}
    assert covered == set(ORDER_TYPES.values())


# ---------------------------------------------------------------------------
# The fuzz net
# ---------------------------------------------------------------------------


def test_issue_never_raises_on_malformed_order_payloads():
    sim, squad, weapon, hq = _sim_with_entities()
    templates = _well_formed(sim, squad, weapon, hq)

    issued = 0
    for template in templates:
        for field_name in type(template).__dataclass_fields__:
            for junk in JUNK:
                from dataclasses import replace

                try:
                    order = replace(template, **{field_name: junk})
                except Exception as exc:  # pragma: no cover - dataclass is permissive
                    pytest.fail(f"constructing {template!r} with {field_name}={junk!r} raised {exc!r}")
                (result,) = sim.issue(0, [order])
                assert isinstance(result, orders_mod.OrderResult)
                issued += 1
    assert issued >= 500  # the net is actually wide


def test_the_state_hash_survives_every_malformed_payload():
    """Nothing an agent sends may leave a value `state_hash()` cannot encode."""
    sim, squad, weapon, hq = _sim_with_entities()

    from dataclasses import replace

    for template in _well_formed(sim, squad, weapon, hq):
        for field_name in type(template).__dataclass_fields__:
            for junk in JUNK:
                sim.issue(0, [replace(template, **{field_name: junk})])
                sim.state_hash()  # must not raise, whatever was stored
                sim.tick()


def test_issue_never_raises_on_a_non_order_object():
    sim, squad, weapon, hq = _sim_with_entities()
    for junk in (None, 5, "Move", {"type": "Move", "squad": 1, "cell": [1, 1]}, object()):
        (result,) = sim.issue(0, [junk])
        assert not result.ok


def test_issue_never_raises_on_orders_rebuilt_from_hostile_dicts():
    """`order_from_dict` may reject a dict, but what it *accepts* must be safe."""
    sim, squad, weapon, hq = _sim_with_entities()
    templates = _well_formed(sim, squad, weapon, hq)

    hostile: list[dict] = [
        {},
        {"type": "Move"},
        {"type": "NotAnOrder", "squad": 1},
        {"type": "Move", "squad": 1},
        {"type": "Move", "squad": 1, "cell": [1, 2], "extra": 3},
    ]
    for template in templates:
        base = order_to_dict(template)
        hostile.append(base)
        for key in base:
            if key == "type":
                continue
            for junk in (None, True, "x", 3.5, [], [1, 2, 3], {}):
                hostile.append({**base, key: junk})

    for payload in hostile:
        try:
            order = order_from_dict(payload)
        except ValueError:
            continue  # documented: a malformed *dict* is a ValueError
        (result,) = sim.issue(0, [order])
        assert isinstance(result, orders_mod.OrderResult)


# ---------------------------------------------------------------------------
# The specific payloads the reviewer found
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_order",
    [
        lambda sq, hq: Move(sq, (1, 2, 3)),
        lambda sq, hq: Move(sq, ("a", "b")),
        lambda sq, hq: Move(sq, 5),
        lambda sq, hq: Move(sq, (1.5, 2.5)),
        lambda sq, hq: AttackMove(sq, (1.5, 2.5)),
        lambda sq, hq: Capture(sq, []),
        lambda sq, hq: Build(sq, "barracks", (1.5, 2.5)),
        lambda sq, hq: Build(sq, 7, (1, 2)),
        lambda sq, hq: Attack(sq, "target"),
        lambda sq, hq: Train(hq, 5),
        lambda sq, hq: Research(hq, None),
        lambda sq, hq: BuyUpgrade(sq, 3),
        lambda sq, hq: Garrison(sq, "house"),
    ],
)
def test_reviewer_payloads_are_rejected_not_raised(make_order):
    sim, squad, weapon, hq = _sim_with_entities()
    order = make_order(squad.id, hq.id)
    (result,) = sim.issue(0, [order])
    assert not result.ok
    assert sim.state.players[0].invalid_orders == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_set_facing_rejects_non_finite_angles_and_leaves_the_hash_usable(bad):
    sim, squad, weapon, hq = _sim_with_entities()

    (result,) = sim.issue(0, [SetFacing(weapon.id, bad)])

    assert not result.ok
    assert "finite" in result.reason
    assert weapon.facing is not None and math.isfinite(weapon.facing)
    sim.state_hash()  # the nan never reached the state: still encodable
    sim.run(5)
    sim.state_hash()


def test_set_facing_still_accepts_a_plain_bearing():
    sim, squad, weapon, hq = _sim_with_entities()
    (result,) = sim.issue(0, [SetFacing(weapon.id, 90)])  # int is a real number
    assert result.ok


# ---------------------------------------------------------------------------
# JSON gives lists where orders declare tuples
# ---------------------------------------------------------------------------


def test_a_list_cell_is_accepted_and_normalized_to_a_tuple():
    sim, squad, weapon, hq = _sim_with_entities()
    (result,) = sim.issue(0, [Move(squad.id, [12, 12])])
    assert result.ok
    assert sim.state.squads[squad.id].order == Move(squad.id, (12, 12))
    assert isinstance(sim.state.squads[squad.id].order.cell, tuple)


def test_an_out_of_bounds_list_cell_is_still_out_of_bounds():
    sim, squad, weapon, hq = _sim_with_entities()
    (result,) = sim.issue(0, [Move(squad.id, [-1, 0])])
    assert not result.ok
    assert "out of bounds" in result.reason


# ---------------------------------------------------------------------------
# numpy-typed orders from an RL policy
# ---------------------------------------------------------------------------


def test_numpy_typed_valid_orders_hash_the_same_as_plain_int_orders():
    """`numbers.Integral` / `numbers.Real` let numpy scalars past the payload
    gate; whatever the sim then stores must hash exactly like the plain-int
    equivalent, or two runs of the same match diverge on dtype alone."""
    np = pytest.importorskip("numpy")

    numpy_sim, numpy_squad, numpy_weapon, numpy_hq = _sim_with_entities()
    numpy_sim.issue(
        0,
        [
            Move(np.int64(numpy_squad.id), (np.int64(12), np.int64(12))),
            SetFacing(np.int64(numpy_weapon.id), np.float32(90.0)),
            Train(np.int64(numpy_hq.id), "rifles"),
        ],
    )
    numpy_sim.state_hash()  # must not raise
    numpy_sim.run(5)
    numpy_hash = numpy_sim.state_hash()

    plain_sim, plain_squad, plain_weapon, plain_hq = _sim_with_entities()
    plain_sim.issue(
        0,
        [
            Move(plain_squad.id, (12, 12)),
            SetFacing(plain_weapon.id, 90.0),
            Train(plain_hq.id, "rifles"),
        ],
    )
    plain_sim.run(5)
    plain_hash = plain_sim.state_hash()

    assert numpy_hash == plain_hash
    assert isinstance(numpy_sim.state.squads[numpy_squad.id].order.cell[0], int)


def test_a_1d_numpy_array_cell_is_accepted_and_normalized_to_a_tuple():
    np = pytest.importorskip("numpy")

    sim, squad, weapon, hq = _sim_with_entities()
    (result,) = sim.issue(0, [Move(squad.id, np.array([12, 12]))])
    assert result.ok
    assert sim.state.squads[squad.id].order == Move(squad.id, (12, 12))
    assert isinstance(sim.state.squads[squad.id].order.cell, tuple)
