"""The agent-facing environment layer.

`coh.env` is the *only* thing agents are allowed to import (plus the public
`GameMap` / `GameData` types handed to `Agent.reset`): it owns the
fog-filtered `Observation`, the legal-order set, and the PettingZoo-style
`CohEnv` that drives a `Sim` at a fixed decision interval.

The order schema is re-exported here too, so an agent never has to reach into
`coh.sim`: orders are part of the env's public API.
"""

from coh.env.env import CohEnv
from coh.env.observation import (
    BuildingView,
    GhostView,
    Observation,
    PointView,
    SquadView,
    build_observation,
)
from coh.sim.orders import (
    Attack,
    AttackMove,
    Build,
    BuyUpgrade,
    Capture,
    Garrison,
    Move,
    Order,
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

__all__ = [
    "Attack",
    "AttackMove",
    "Build",
    "BuildingView",
    "BuyUpgrade",
    "Capture",
    "CohEnv",
    "Garrison",
    "GhostView",
    "Move",
    "Observation",
    "Order",
    "PointView",
    "Reinforce",
    "Research",
    "Retreat",
    "SetFacing",
    "SquadView",
    "Stop",
    "Train",
    "Ungarrison",
    "build_observation",
    "order_from_dict",
    "order_to_dict",
]
