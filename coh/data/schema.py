"""Frozen dataclasses describing every game-data table.

These are pure data containers: no gameplay logic beyond small lookup
helpers (`WeaponDef.vs` / `WeaponDef.cover`) that apply the documented
"missing key -> identity mods" fallback. Values are loaded from YAML by
`coh.data.loader`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

COVER_TYPES = ("open", "light", "heavy", "negative", "garrison")

# The factions and squad kinds the sim knows how to play.
FACTIONS = ("us", "wehr")
SQUAD_KINDS = ("infantry", "team_weapon", "vehicle")

# Territory-point types. Maps tag their points with these (`coh.maps.format`
# imports this tuple, the same way it imports `COVER_TYPES`), and
# `economy.point_income` must price every one of them.
POINT_TYPES = (
    "strategic",
    "munitions_low",
    "munitions_med",
    "munitions_high",
    "fuel_low",
    "fuel_med",
    "fuel_high",
    "victory",
)


@dataclass(frozen=True)
class Cost:
    manpower: float = 0
    munitions: float = 0
    fuel: float = 0


@dataclass(frozen=True)
class TargetMods:
    accuracy: float = 1.0
    moving: float = 1.0
    damage: float = 1.0
    penetration: float = 1.0
    rear_penetration: float = 1.0
    suppression: float = 1.0
    priority: float = 0


@dataclass(frozen=True)
class CoverMods:
    accuracy: float = 1.0
    damage: float = 1.0
    suppression: float = 1.0


@dataclass(frozen=True)
class WeaponDef:
    id: str
    damage: float
    ranges: tuple[float, float, float]  # short, medium, long(=max)
    min_range: float
    accuracy: tuple[float, float, float]  # indexed by range band
    penetration: tuple[float, float, float]
    suppression: tuple[float, float, float]  # added to target per bullet
    cooldown: tuple[float, float]  # seconds min,max between shots/bursts
    cooldown_range_mult: tuple[float, float, float]
    burst: tuple[float, float] | None  # burst duration min,max seconds; None = single shot
    rate_of_fire: float  # bullets/sec during a burst
    reload: tuple[float, float]
    reload_every: int  # shots or bursts between reloads
    setup_time: float = 0.0  # 0 for non-team weapons
    arc_deg: float = 360.0  # 360 unless team weapon / hull gun
    moving_accuracy: float = 0.0  # 0 => cannot fire while moving
    nearby_suppression_mult: float = 1.0
    nearby_suppression_radius: float = 0.0
    aoe_radius: float = 0.0
    deflection_damage_mult: float = 1.0
    indirect: bool = False
    scatter_m: float = 0.0  # indirect only: stddev of impact offset at max range
    cover_table: dict[str, CoverMods] = field(default_factory=dict)  # keys subset of COVER_TYPES
    target_table: dict[str, TargetMods] = field(default_factory=dict)  # missing target type -> TargetMods()

    def vs(self, target_type: str) -> TargetMods:
        return self.target_table.get(target_type, TargetMods())

    def cover(self, cover_type: str) -> CoverMods:
        return self.cover_table.get(cover_type, CoverMods())


@dataclass(frozen=True)
class SuppressionDef:
    suppress_at: float
    suppress_recover: float
    pin_at: float
    pin_recover: float
    recovery_per_s: float


@dataclass(frozen=True)
class SquadDef:
    id: str
    faction: str  # "us" | "wehr"
    kind: str  # "infantry" | "team_weapon" | "vehicle"
    members: int
    member_hp: float  # vehicles: members=1, member_hp = vehicle HP
    loadout: tuple[str, ...]  # weapon id per member ("" = unarmed); vehicles: all weapons on the one member
    target_type: str
    cost: Cost
    population: int
    build_time: float
    upkeep_per_min: float
    speed: float
    rotation_deg_s: float  # rotation only meaningful for vehicles/team weapons
    sight: float
    capture_rate: float  # 0 => cannot capture
    builds: tuple[str, ...]  # building ids this squad can construct
    suppression: SuppressionDef | None  # None => immune (vehicles)
    reinforce_cost_mult: float
    reinforce_time_mult: float
    retreat_received_accuracy: float
    upgrade_slots: int
    crushes_light_cover: bool = False
    can_garrison: bool = False


@dataclass(frozen=True)
class BuildingDef:
    id: str
    faction: str
    cost: Cost
    build_time: float
    hp: float
    target_type: str
    footprint: tuple[int, int]  # cells w,h
    produces: tuple[str, ...]
    researches: tuple[str, ...]
    requires: tuple[str, ...]  # building or upgrade ids, all required
    is_hq: bool
    reinforce_radius: float
    sight: float


@dataclass(frozen=True)
class UpgradeDef:  # global research
    id: str
    faction: str
    cost: Cost
    time: float
    requires: tuple[str, ...]
    upkeep_mult: float = 1.0  # Supply Yard levels


@dataclass(frozen=True)
class SquadUpgradeDef:  # per-squad purchase, e.g. BAR
    id: str
    applies_to: tuple[str, ...]
    cost: Cost
    time: float
    requires: tuple[str, ...]
    weapon: str
    count: int  # replaces `count` member weapon slots


@dataclass(frozen=True)
class NeutralBuildingDef:
    id: str
    hp: float
    target_type: str
    capacity: int
    footprint: tuple[int, int]


@dataclass(frozen=True)
class PointIncome:
    manpower: float = 0
    munitions: float = 0
    fuel: float = 0
    population: int = 0


@dataclass(frozen=True)
class EconomyDef:
    start_resources: Cost
    base_income_per_min: Cost
    base_population: int
    max_population: int
    point_income: dict[str, PointIncome]  # keys: strategic, munitions_low/med/high, fuel_low/med/high, victory
    op_income_mult: float
    op_building: dict[str, str]  # faction -> building id
    capture_time_s: float
    neutralize_time_s: float
    capture_radius: float
    tickets: int
    ticket_interval_s: float
    tickets_per_vp_lead: float
    retreat_speed_mult: float
    suppressed_speed_mult: float
    suppressed_accuracy_mult: float
    suppressed_cooldown_mult: float
    noncombat_recovery_delay_s: float
    noncombat_recovery_mult: float
    cover_recovery_mult: dict[str, float]
    reinforce_base_cost_frac: float  # per-model cost = squad cost / members * frac * squad.reinforce_cost_mult
    garrison_collapse_damage_frac: float
    min_manpower_income_frac: float


@dataclass(frozen=True)
class GameData:
    weapons: dict[str, WeaponDef]
    squads: dict[str, SquadDef]
    buildings: dict[str, BuildingDef]
    upgrades: dict[str, UpgradeDef]
    squad_upgrades: dict[str, SquadUpgradeDef]
    neutral: dict[str, NeutralBuildingDef]
    economy: EconomyDef
