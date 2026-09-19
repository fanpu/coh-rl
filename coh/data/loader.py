"""Loads and validates `coh/data/schema.py` tables from a directory of YAML files.

Each table file (`weapons.yaml`, `squads.yaml`, `buildings.yaml`,
`upgrades.yaml`, `economy.yaml`, `neutral.yaml`) is a YAML mapping with a
required top-level `source:` string, an optional `estimated:` mapping of
`id/field -> reason`, and the table body itself (see the per-file loader
functions below for the exact body key(s)).

Validation fails fast with `DataError`, whose message always names the
offending file and key so the source of a bad table is obvious.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import (
    COVER_TYPES,
    FACTIONS,
    POINT_TYPES,
    SQUAD_KINDS,
    BuildingDef,
    Cost,
    CoverMods,
    EconomyDef,
    GameData,
    NeutralBuildingDef,
    PointIncome,
    SquadDef,
    SquadUpgradeDef,
    SuppressionDef,
    TargetMods,
    UpgradeDef,
    WeaponDef,
)

DEFAULT_TABLES_DIR = Path(__file__).parent / "tables"

TABLE_FILE_NAMES = (
    "weapons.yaml",
    "squads.yaml",
    "buildings.yaml",
    "upgrades.yaml",
    "economy.yaml",
    "neutral.yaml",
)

_POINT_INCOME_KEYS = set(POINT_TYPES)
_COST_KEYS = {"manpower", "munitions", "fuel"}
_TARGET_MOD_KEYS = {"accuracy", "moving", "damage", "penetration", "rear_penetration", "suppression", "priority"}
_COVER_MOD_KEYS = {"accuracy", "damage", "suppression"}
_SUPPRESSION_KEYS = {"suppress_at", "suppress_recover", "pin_at", "pin_recover", "recovery_per_s"}
_POINT_INCOME_MOD_KEYS = {"manpower", "munitions", "fuel", "population"}


class DataError(ValueError):
    """Raised for any malformed or inconsistent game-data table.

    The message always includes the offending file name and the entry key.
    """


def load_game_data(tables_dir: Path | None = None) -> GameData:
    """Load and validate all game-data tables.

    With no argument, loads the tables packaged at `coh/data/tables/`
    (populated by a later data-scraping task). Pass `tables_dir` to load a
    different directory, e.g. the hand-written fixtures under
    `tests/data/fixtures/`.
    """
    directory = Path(tables_dir) if tables_dir is not None else DEFAULT_TABLES_DIR
    if not directory.is_dir() or not any((directory / name).exists() for name in TABLE_FILE_NAMES):
        raise DataError(
            f"{directory}: no data table files found (expected {', '.join(TABLE_FILE_NAMES)}); "
            "the packaged coh/data/tables/ set is populated by a later task — pass tables_dir= "
            "to load a fixture directory instead"
        )

    weapons, generic_target_types = _load_weapons(directory / "weapons.yaml")
    squads = _load_squads(directory / "squads.yaml")
    buildings = _load_buildings(directory / "buildings.yaml")
    upgrades, squad_upgrades = _load_upgrades(directory / "upgrades.yaml")
    neutral = _load_neutral(directory / "neutral.yaml")
    economy = _load_economy(directory / "economy.yaml")

    _cross_validate(
        weapons, squads, buildings, upgrades, squad_upgrades, neutral, economy, generic_target_types
    )

    return GameData(
        weapons=weapons,
        squads=squads,
        buildings=buildings,
        upgrades=upgrades,
        squad_upgrades=squad_upgrades,
        neutral=neutral,
        economy=economy,
    )


# --- generic YAML / validation helpers -------------------------------------------------


def _read_doc(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DataError(f"{path.name}: file not found")
    with path.open() as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise DataError(f"{path.name}: expected a YAML mapping at the top level")
    if not isinstance(doc.get("source"), str) or not doc.get("source"):
        raise DataError(f"{path.name}: missing required string 'source' key")
    return doc


def _table(doc: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    body = doc.get(key)
    if body is None:
        raise DataError(f"{path.name}: missing '{key}' table")
    if not isinstance(body, dict):
        raise DataError(f"{path.name}: '{key}' table must be a mapping of id -> fields")
    return body


def _check_keys(d: dict[str, Any], allowed: set[str], required: set[str], path: Path, entry_id: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise DataError(f"{path.name}:{entry_id}: unknown key(s) {sorted(unknown)}")
    missing = required - set(d)
    if missing:
        raise DataError(f"{path.name}:{entry_id}: missing required key(s) {sorted(missing)}")


def _tuple(value: Any, n: int, path: Path, entry_id: str, field_name: str, cast=float) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise DataError(f"{path.name}:{entry_id}: '{field_name}' must be a list of length {n}")
    return tuple(cast(x) for x in value)


def _cost(d: dict[str, Any] | None, path: Path, entry_id: str) -> Cost:
    d = d or {}
    unknown = set(d) - _COST_KEYS
    if unknown:
        raise DataError(f"{path.name}:{entry_id}: unknown cost key(s) {sorted(unknown)}")
    for key, value in d.items():
        if float(value) < 0:
            raise DataError(f"{path.name}:{entry_id}: cost '{key}' is {value}, must not be negative")
    return Cost(manpower=float(d.get("manpower", 0)), munitions=float(d.get("munitions", 0)), fuel=float(d.get("fuel", 0)))


def _non_negative(value: Any, path: Path, entry_id: str, field_name: str) -> float:
    """A float that a negative value would make meaningless (hp, times, rates)."""
    number = float(value)
    if number < 0:
        raise DataError(f"{path.name}:{entry_id}: '{field_name}' is {number}, must not be negative")
    return number


def _non_negative_tuple(values: tuple[float, ...], path: Path, entry_id: str, field_name: str) -> tuple[float, ...]:
    for value in values:
        if value < 0:
            raise DataError(f"{path.name}:{entry_id}: '{field_name}' contains {value}, must not be negative")
    return values


def _enum(value: Any, allowed: tuple[str, ...], path: Path, entry_id: str, field_name: str) -> str:
    text = str(value)
    if text not in allowed:
        raise DataError(f"{path.name}:{entry_id}: '{field_name}' is {text!r}, expected one of {list(allowed)}")
    return text


def _footprint(value: Any, path: Path, entry_id: str) -> tuple[int, int]:
    footprint = _tuple(value, 2, path, entry_id, "footprint", cast=int)
    if footprint[0] < 1 or footprint[1] < 1:
        raise DataError(f"{path.name}:{entry_id}: 'footprint' is {list(footprint)}, must be at least 1x1")
    return footprint


# --- weapons -----------------------------------------------------------------------

_WEAPON_REQUIRED = {
    "damage",
    "ranges",
    "min_range",
    "accuracy",
    "penetration",
    "suppression",
    "cooldown",
    "cooldown_range_mult",
    "burst",
    "rate_of_fire",
    "reload",
    "reload_every",
}
_WEAPON_OPTIONAL = {
    "setup_time",
    "arc_deg",
    "moving_accuracy",
    "nearby_suppression_mult",
    "nearby_suppression_radius",
    "aoe_radius",
    "deflection_damage_mult",
    "indirect",
    "scatter_m",
    "cover_table",
    "target_table",
}


def _cover_table(d: dict[str, Any] | None, path: Path, entry_id: str) -> dict[str, CoverMods]:
    d = d or {}
    result: dict[str, CoverMods] = {}
    for cover_type, mods in d.items():
        if cover_type not in COVER_TYPES:
            raise DataError(f"{path.name}:{entry_id}: cover_table key '{cover_type}' not in {COVER_TYPES}")
        mods = mods or {}
        unknown = set(mods) - _COVER_MOD_KEYS
        if unknown:
            raise DataError(f"{path.name}:{entry_id}: unknown cover mod key(s) {sorted(unknown)} for '{cover_type}'")
        for key, value in mods.items():
            _non_negative(value, path, entry_id, f"cover_table.{cover_type}.{key}")
        result[cover_type] = CoverMods(
            accuracy=float(mods.get("accuracy", 1.0)),
            damage=float(mods.get("damage", 1.0)),
            suppression=float(mods.get("suppression", 1.0)),
        )
    return result


def _target_table(d: dict[str, Any] | None, path: Path, entry_id: str) -> dict[str, TargetMods]:
    d = d or {}
    result: dict[str, TargetMods] = {}
    for target_type, mods in d.items():
        mods = mods or {}
        unknown = set(mods) - _TARGET_MOD_KEYS
        if unknown:
            raise DataError(f"{path.name}:{entry_id}: unknown target mod key(s) {sorted(unknown)} for '{target_type}'")
        for key, value in mods.items():
            if key != "priority":  # priority is a sort key and may be negative
                _non_negative(value, path, entry_id, f"target_table.{target_type}.{key}")
        result[target_type] = TargetMods(
            accuracy=float(mods.get("accuracy", 1.0)),
            moving=float(mods.get("moving", 1.0)),
            damage=float(mods.get("damage", 1.0)),
            penetration=float(mods.get("penetration", 1.0)),
            rear_penetration=float(mods.get("rear_penetration", 1.0)),
            suppression=float(mods.get("suppression", 1.0)),
            priority=float(mods.get("priority", 0)),
        )
    return result


def _ranges(value: Any, path: Path, entry_id: str) -> tuple[float, ...]:
    """The three range bands, which must be strictly ascending and positive."""
    ranges = _tuple(value, 3, path, entry_id, "ranges")
    if not (0 < ranges[0] < ranges[1] < ranges[2]):
        raise DataError(
            f"{path.name}:{entry_id}: 'ranges' {list(ranges)} must be positive and strictly ascending"
        )
    return ranges


def _load_weapons(path: Path) -> tuple[dict[str, WeaponDef], frozenset[str]]:
    """The weapon table, plus the optional `generic_target_types` declaration.

    A target type no weapon has a `target_table` row for silently falls back
    to identity modifiers, which is fine but must be *deliberate*: listing it
    under the top-level `generic_target_types:` key says so, and
    `_cross_validate` rejects any other unmatched target type.
    """
    doc = _read_doc(path)
    body = _table(doc, "weapons", path)
    generic_raw = doc.get("generic_target_types") or []
    if not isinstance(generic_raw, list):
        raise DataError(f"{path.name}: 'generic_target_types' must be a list of target-type names")
    generic_target_types = frozenset(str(t) for t in generic_raw)

    weapons: dict[str, WeaponDef] = {}
    for wid, raw in body.items():
        raw = raw or {}
        _check_keys(raw, _WEAPON_REQUIRED | _WEAPON_OPTIONAL, _WEAPON_REQUIRED, path, wid)
        burst_raw = raw["burst"]
        burst = None if burst_raw is None else _tuple(burst_raw, 2, path, wid, "burst")
        ranges = _ranges(raw["ranges"], path, wid)
        min_range = _non_negative(raw["min_range"], path, wid, "min_range")
        if min_range >= ranges[2]:
            raise DataError(
                f"{path.name}:{wid}: 'min_range' {min_range} is not below the weapon's "
                f"maximum range {ranges[2]} -- the weapon could never fire"
            )
        weapons[wid] = WeaponDef(
            id=wid,
            damage=_non_negative(raw["damage"], path, wid, "damage"),
            ranges=ranges,
            min_range=min_range,
            accuracy=_non_negative_tuple(_tuple(raw["accuracy"], 3, path, wid, "accuracy"), path, wid, "accuracy"),
            penetration=_non_negative_tuple(
                _tuple(raw["penetration"], 3, path, wid, "penetration"), path, wid, "penetration"
            ),
            suppression=_non_negative_tuple(
                _tuple(raw["suppression"], 3, path, wid, "suppression"), path, wid, "suppression"
            ),
            cooldown=_tuple(raw["cooldown"], 2, path, wid, "cooldown"),
            cooldown_range_mult=_tuple(raw["cooldown_range_mult"], 3, path, wid, "cooldown_range_mult"),
            burst=burst,
            rate_of_fire=float(raw["rate_of_fire"]),
            reload=_tuple(raw["reload"], 2, path, wid, "reload"),
            reload_every=int(raw["reload_every"]),
            setup_time=_non_negative(raw.get("setup_time", 0.0), path, wid, "setup_time"),
            arc_deg=float(raw.get("arc_deg", 360.0)),
            moving_accuracy=float(raw.get("moving_accuracy", 0.0)),
            nearby_suppression_mult=float(raw.get("nearby_suppression_mult", 1.0)),
            nearby_suppression_radius=float(raw.get("nearby_suppression_radius", 0.0)),
            aoe_radius=float(raw.get("aoe_radius", 0.0)),
            deflection_damage_mult=float(raw.get("deflection_damage_mult", 1.0)),
            indirect=bool(raw.get("indirect", False)),
            scatter_m=float(raw.get("scatter_m", 0.0)),
            cover_table=_cover_table(raw.get("cover_table"), path, wid),
            target_table=_target_table(raw.get("target_table"), path, wid),
        )
    return weapons, generic_target_types


# --- squads --------------------------------------------------------------------------

_SQUAD_REQUIRED = {
    "faction",
    "kind",
    "members",
    "member_hp",
    "loadout",
    "target_type",
    "cost",
    "population",
    "build_time",
    "upkeep_per_min",
    "speed",
    "rotation_deg_s",
    "sight",
    "capture_rate",
    "builds",
    "suppression",
    "reinforce_cost_mult",
    "reinforce_time_mult",
    "retreat_received_accuracy",
    "upgrade_slots",
}
_SQUAD_OPTIONAL = {"crushes_light_cover", "can_garrison"}


def _load_squads(path: Path) -> dict[str, SquadDef]:
    doc = _read_doc(path)
    body = _table(doc, "squads", path)
    squads: dict[str, SquadDef] = {}
    for sid, raw in body.items():
        raw = raw or {}
        _check_keys(raw, _SQUAD_REQUIRED | _SQUAD_OPTIONAL, _SQUAD_REQUIRED, path, sid)

        kind = _enum(raw["kind"], SQUAD_KINDS, path, sid, "kind")
        members = int(raw["members"])
        if members < 1:
            raise DataError(f"{path.name}:{sid}: 'members' is {members}, must be at least 1")
        if kind == "vehicle" and members != 1:
            raise DataError(f"{path.name}:{sid}: a vehicle must have exactly 1 'members' entry, got {members}")
        loadout_raw = raw["loadout"]
        if not isinstance(loadout_raw, (list, tuple)) or len(loadout_raw) != members:
            raise DataError(f"{path.name}:{sid}: 'loadout' must be a list of length members ({members})")
        loadout = tuple(str(w) for w in loadout_raw)
        if kind == "team_weapon" and not loadout[0]:
            raise DataError(
                f"{path.name}:{sid}: a team weapon's 'loadout' slot 0 must carry the crewed weapon, got an empty slot"
            )

        suppression_raw = raw["suppression"]
        suppression = None
        if suppression_raw is not None:
            _check_keys(suppression_raw, _SUPPRESSION_KEYS, _SUPPRESSION_KEYS, path, sid)
            suppression = SuppressionDef(**{k: float(v) for k, v in suppression_raw.items()})

        squads[sid] = SquadDef(
            id=sid,
            faction=_enum(raw["faction"], FACTIONS, path, sid, "faction"),
            kind=kind,
            members=members,
            member_hp=_non_negative(raw["member_hp"], path, sid, "member_hp"),
            loadout=loadout,
            target_type=str(raw["target_type"]),
            cost=_cost(raw["cost"], path, sid),
            population=int(raw["population"]),
            build_time=_non_negative(raw["build_time"], path, sid, "build_time"),
            upkeep_per_min=_non_negative(raw["upkeep_per_min"], path, sid, "upkeep_per_min"),
            speed=_non_negative(raw["speed"], path, sid, "speed"),
            rotation_deg_s=_non_negative(raw["rotation_deg_s"], path, sid, "rotation_deg_s"),
            sight=_non_negative(raw["sight"], path, sid, "sight"),
            capture_rate=_non_negative(raw["capture_rate"], path, sid, "capture_rate"),
            builds=tuple(str(b) for b in raw["builds"]),
            suppression=suppression,
            reinforce_cost_mult=float(raw["reinforce_cost_mult"]),
            reinforce_time_mult=float(raw["reinforce_time_mult"]),
            retreat_received_accuracy=float(raw["retreat_received_accuracy"]),
            upgrade_slots=int(raw["upgrade_slots"]),
            crushes_light_cover=bool(raw.get("crushes_light_cover", False)),
            can_garrison=bool(raw.get("can_garrison", False)),
        )
    return squads


# --- buildings -----------------------------------------------------------------------

_BUILDING_REQUIRED = {
    "faction",
    "cost",
    "build_time",
    "hp",
    "target_type",
    "footprint",
    "produces",
    "researches",
    "requires",
    "is_hq",
    "reinforce_radius",
    "sight",
}


def _load_buildings(path: Path) -> dict[str, BuildingDef]:
    doc = _read_doc(path)
    body = _table(doc, "buildings", path)
    buildings: dict[str, BuildingDef] = {}
    for bid, raw in body.items():
        raw = raw or {}
        _check_keys(raw, _BUILDING_REQUIRED, _BUILDING_REQUIRED, path, bid)
        buildings[bid] = BuildingDef(
            id=bid,
            faction=_enum(raw["faction"], FACTIONS, path, bid, "faction"),
            cost=_cost(raw["cost"], path, bid),
            build_time=_non_negative(raw["build_time"], path, bid, "build_time"),
            hp=_non_negative(raw["hp"], path, bid, "hp"),
            target_type=str(raw["target_type"]),
            footprint=_footprint(raw["footprint"], path, bid),
            produces=tuple(str(x) for x in raw["produces"]),
            researches=tuple(str(x) for x in raw["researches"]),
            requires=tuple(str(x) for x in raw["requires"]),
            is_hq=bool(raw["is_hq"]),
            reinforce_radius=_non_negative(raw["reinforce_radius"], path, bid, "reinforce_radius"),
            sight=_non_negative(raw["sight"], path, bid, "sight"),
        )
    return buildings


# --- upgrades / squad_upgrades --------------------------------------------------------

_UPGRADE_REQUIRED = {"faction", "cost", "time", "requires"}
_UPGRADE_OPTIONAL = {"upkeep_mult"}
_SQUAD_UPGRADE_REQUIRED = {"applies_to", "cost", "time", "requires", "weapon", "count"}


def _load_upgrades(path: Path) -> tuple[dict[str, UpgradeDef], dict[str, SquadUpgradeDef]]:
    doc = _read_doc(path)
    upgrades_body = _table(doc, "upgrades", path)
    squad_upgrades_body = _table(doc, "squad_upgrades", path)

    upgrades: dict[str, UpgradeDef] = {}
    for uid, raw in upgrades_body.items():
        raw = raw or {}
        _check_keys(raw, _UPGRADE_REQUIRED | _UPGRADE_OPTIONAL, _UPGRADE_REQUIRED, path, uid)
        upgrades[uid] = UpgradeDef(
            id=uid,
            faction=_enum(raw["faction"], FACTIONS, path, uid, "faction"),
            cost=_cost(raw["cost"], path, uid),
            time=_non_negative(raw["time"], path, uid, "time"),
            requires=tuple(str(x) for x in raw["requires"]),
            upkeep_mult=float(raw.get("upkeep_mult", 1.0)),
        )

    squad_upgrades: dict[str, SquadUpgradeDef] = {}
    for uid, raw in squad_upgrades_body.items():
        raw = raw or {}
        _check_keys(raw, _SQUAD_UPGRADE_REQUIRED, _SQUAD_UPGRADE_REQUIRED, path, uid)
        squad_upgrades[uid] = SquadUpgradeDef(
            id=uid,
            applies_to=tuple(str(x) for x in raw["applies_to"]),
            cost=_cost(raw["cost"], path, uid),
            time=_non_negative(raw["time"], path, uid, "time"),
            requires=tuple(str(x) for x in raw["requires"]),
            weapon=str(raw["weapon"]),
            count=int(raw["count"]),
        )

    return upgrades, squad_upgrades


# --- neutral -------------------------------------------------------------------------

_NEUTRAL_REQUIRED = {"hp", "target_type", "capacity", "footprint"}


def _load_neutral(path: Path) -> dict[str, NeutralBuildingDef]:
    doc = _read_doc(path)
    body = _table(doc, "neutral", path)
    neutral: dict[str, NeutralBuildingDef] = {}
    for nid, raw in body.items():
        raw = raw or {}
        _check_keys(raw, _NEUTRAL_REQUIRED, _NEUTRAL_REQUIRED, path, nid)
        neutral[nid] = NeutralBuildingDef(
            id=nid,
            hp=_non_negative(raw["hp"], path, nid, "hp"),
            target_type=str(raw["target_type"]),
            capacity=int(raw["capacity"]),
            footprint=_footprint(raw["footprint"], path, nid),
        )
    return neutral


# --- economy -------------------------------------------------------------------------

_ECONOMY_REQUIRED = {
    "start_resources",
    "base_income_per_min",
    "base_population",
    "max_population",
    "point_income",
    "op_income_mult",
    "op_building",
    "capture_time_s",
    "neutralize_time_s",
    "capture_radius",
    "tickets",
    "ticket_interval_s",
    "tickets_per_vp_lead",
    "retreat_speed_mult",
    "suppressed_speed_mult",
    "suppressed_accuracy_mult",
    "suppressed_cooldown_mult",
    "noncombat_recovery_delay_s",
    "noncombat_recovery_mult",
    "cover_recovery_mult",
    "reinforce_base_cost_frac",
    "garrison_collapse_damage_frac",
    "min_manpower_income_frac",
}


def _load_economy(path: Path) -> EconomyDef:
    doc = _read_doc(path)
    raw = doc.get("economy")
    if raw is None:
        raise DataError(f"{path.name}: missing 'economy' table")
    if not isinstance(raw, dict):
        raise DataError(f"{path.name}: 'economy' table must be a mapping")
    _check_keys(raw, _ECONOMY_REQUIRED, _ECONOMY_REQUIRED, path, "economy")

    point_income_raw = raw["point_income"] or {}
    unknown_points = set(point_income_raw) - _POINT_INCOME_KEYS
    if unknown_points:
        raise DataError(f"{path.name}:economy: unknown point_income key(s) {sorted(unknown_points)}")
    point_income: dict[str, PointIncome] = {}
    for point_type, mods in point_income_raw.items():
        mods = mods or {}
        unknown = set(mods) - _POINT_INCOME_MOD_KEYS
        if unknown:
            raise DataError(f"{path.name}:economy: unknown point_income mod key(s) {sorted(unknown)} for '{point_type}'")
        point_income[point_type] = PointIncome(
            manpower=float(mods.get("manpower", 0)),
            munitions=float(mods.get("munitions", 0)),
            fuel=float(mods.get("fuel", 0)),
            population=int(mods.get("population", 0)),
        )

    cover_recovery_raw = raw["cover_recovery_mult"] or {}
    unknown_cover = set(cover_recovery_raw) - set(COVER_TYPES)
    if unknown_cover:
        raise DataError(f"{path.name}:economy: unknown cover_recovery_mult key(s) {sorted(unknown_cover)}")
    cover_recovery_mult = {str(k): float(v) for k, v in cover_recovery_raw.items()}

    op_building = {str(k): str(v) for k, v in (raw["op_building"] or {}).items()}

    return EconomyDef(
        start_resources=_cost(raw["start_resources"], path, "economy.start_resources"),
        base_income_per_min=_cost(raw["base_income_per_min"], path, "economy.base_income_per_min"),
        base_population=int(raw["base_population"]),
        max_population=int(raw["max_population"]),
        point_income=point_income,
        op_income_mult=float(raw["op_income_mult"]),
        op_building=op_building,
        capture_time_s=float(raw["capture_time_s"]),
        neutralize_time_s=float(raw["neutralize_time_s"]),
        capture_radius=float(raw["capture_radius"]),
        tickets=int(raw["tickets"]),
        ticket_interval_s=float(raw["ticket_interval_s"]),
        tickets_per_vp_lead=float(raw["tickets_per_vp_lead"]),
        retreat_speed_mult=float(raw["retreat_speed_mult"]),
        suppressed_speed_mult=float(raw["suppressed_speed_mult"]),
        suppressed_accuracy_mult=float(raw["suppressed_accuracy_mult"]),
        suppressed_cooldown_mult=float(raw["suppressed_cooldown_mult"]),
        noncombat_recovery_delay_s=float(raw["noncombat_recovery_delay_s"]),
        noncombat_recovery_mult=float(raw["noncombat_recovery_mult"]),
        cover_recovery_mult=cover_recovery_mult,
        reinforce_base_cost_frac=float(raw["reinforce_base_cost_frac"]),
        garrison_collapse_damage_frac=float(raw["garrison_collapse_damage_frac"]),
        min_manpower_income_frac=float(raw["min_manpower_income_frac"]),
    )


# --- cross-table validation ------------------------------------------------------------


def _check_target_types_are_shootable(
    weapons: dict[str, WeaponDef],
    squads: dict[str, SquadDef],
    buildings: dict[str, BuildingDef],
    neutral: dict[str, NeutralBuildingDef],
    generic_target_types: frozenset[str],
) -> None:
    """Every target type must be priced by some weapon, or declared generic.

    `WeaponDef.vs` falls back to identity modifiers for an unknown target
    type, so a typo in one `target_type` would otherwise make a whole unit
    class quietly immune to nothing in particular.
    """
    priced = {target_type for weapon in weapons.values() for target_type in weapon.target_table}
    priced |= generic_target_types
    for file_name, table in (
        ("squads.yaml", squads),
        ("buildings.yaml", buildings),
        ("neutral.yaml", neutral),
    ):
        for entry_id, entry in table.items():
            if entry.target_type not in priced:
                raise DataError(
                    f"{file_name}:{entry_id}: target_type '{entry.target_type}' has no row in any weapon's "
                    "target_table; add one, or list it under weapons.yaml's 'generic_target_types'"
                )


def _check_requires_has_no_cycles(
    buildings: dict[str, BuildingDef], upgrades: dict[str, UpgradeDef]
) -> None:
    """`requires` spans both tables, so the cycle check has to as well."""
    edges: dict[str, tuple[str, ...]] = {}
    origin: dict[str, str] = {}
    for bid, b in buildings.items():
        edges[bid] = b.requires
        origin[bid] = "buildings.yaml"
    for uid, u in upgrades.items():
        edges.setdefault(uid, u.requires)
        origin.setdefault(uid, "upgrades.yaml")

    visiting: list[str] = []
    done: set[str] = set()

    def visit(node: str) -> None:
        if node in done:
            return
        if node in visiting:
            loop = visiting[visiting.index(node):] + [node]
            raise DataError(f"{origin[node]}:{node}: 'requires' cycle {' -> '.join(loop)}")
        visiting.append(node)
        for parent in edges.get(node, ()):
            if parent in edges:
                visit(parent)
        visiting.pop()
        done.add(node)

    for node in edges:
        visit(node)


def _check_every_faction_can_play(
    squads: dict[str, SquadDef],
    buildings: dict[str, BuildingDef],
    upgrades: dict[str, UpgradeDef],
) -> None:
    """A faction with an HQ must be able to build and to take territory.

    Walks the tech tree out from each faction's HQ -- what the HQ produces,
    what those squads build, what those buildings produce and research -- and
    insists the closure contains a builder and a capture-capable squad.
    """
    for faction in sorted({b.faction for b in buildings.values() if b.is_hq}):
        hq = next(b for b in buildings.values() if b.is_hq and b.faction == faction)
        reachable_buildings = {hq.id}
        reachable_upgrades: set[str] = set()
        reachable_squads: set[str] = set()
        changed = True
        while changed:
            changed = False
            for bid in sorted(reachable_buildings):
                bdef = buildings[bid]
                for unit in bdef.produces:
                    if unit in squads and unit not in reachable_squads:
                        reachable_squads.add(unit)
                        changed = True
                for upgrade_id in bdef.researches:
                    if upgrade_id in upgrades and upgrade_id not in reachable_upgrades:
                        reachable_upgrades.add(upgrade_id)
                        changed = True
            unlocked = reachable_buildings | reachable_upgrades
            for sid in sorted(reachable_squads):
                for structure in squads[sid].builds:
                    bdef = buildings.get(structure)
                    if bdef is None or bdef.faction != faction or structure in reachable_buildings:
                        continue
                    if all(req in unlocked for req in bdef.requires):
                        reachable_buildings.add(structure)
                        changed = True

        if not any(squads[sid].builds for sid in reachable_squads):
            raise DataError(
                f"buildings.yaml:{hq.id}: faction '{faction}' can never build anything -- no squad reachable "
                "from its HQ has a non-empty 'builds'"
            )
        if not any(squads[sid].capture_rate > 0 for sid in reachable_squads):
            raise DataError(
                f"squads.yaml: faction '{faction}' can never capture a point -- no squad reachable from its "
                "HQ has capture_rate > 0"
            )


def _cross_validate(
    weapons: dict[str, WeaponDef],
    squads: dict[str, SquadDef],
    buildings: dict[str, BuildingDef],
    upgrades: dict[str, UpgradeDef],
    squad_upgrades: dict[str, SquadUpgradeDef],
    neutral: dict[str, NeutralBuildingDef],
    economy: EconomyDef,
    generic_target_types: frozenset[str],
) -> None:
    for sid, sq in squads.items():
        for w in sq.loadout:
            if w and w not in weapons:
                raise DataError(f"squads.yaml:{sid}: loadout weapon '{w}' not found in weapons")
        for b in sq.builds:
            if b not in buildings:
                raise DataError(f"squads.yaml:{sid}: builds '{b}' not found in buildings")
        if sq.kind == "team_weapon":
            crewed = weapons.get(sq.loadout[0])
            if crewed is not None and crewed.setup_time <= 0:
                raise DataError(
                    f"squads.yaml:{sid}: the crewed weapon '{sq.loadout[0]}' has setup_time "
                    f"{crewed.setup_time}; a team weapon must take time to deploy"
                )

    for uid, su in squad_upgrades.items():
        if su.weapon not in weapons:
            raise DataError(f"upgrades.yaml:{uid}: squad_upgrades weapon '{su.weapon}' not found in weapons")
        for a in su.applies_to:
            if a not in squads:
                raise DataError(f"upgrades.yaml:{uid}: applies_to '{a}' not found in squads")
        for r in su.requires:
            if r not in upgrades and r not in buildings:
                raise DataError(f"upgrades.yaml:{uid}: requires '{r}' not found in upgrades or buildings")

    for uid, u in upgrades.items():
        for r in u.requires:
            if r not in upgrades and r not in buildings:
                raise DataError(f"upgrades.yaml:{uid}: requires '{r}' not found in upgrades or buildings")

    for bid, b in buildings.items():
        for p in b.produces:
            if p not in squads:
                raise DataError(f"buildings.yaml:{bid}: produces '{p}' not found in squads")
        for r in b.researches:
            if r not in upgrades:
                raise DataError(f"buildings.yaml:{bid}: researches '{r}' not found in upgrades")
        for req in b.requires:
            if req not in buildings and req not in upgrades:
                raise DataError(f"buildings.yaml:{bid}: requires '{req}' not found in buildings or upgrades")

    hq_counts: dict[str, int] = {}
    for b in buildings.values():
        hq_counts.setdefault(b.faction, 0)
        if b.is_hq:
            hq_counts[b.faction] += 1
    for faction, count in hq_counts.items():
        if count != 1:
            raise DataError(f"buildings.yaml: faction '{faction}' has {count} is_hq building(s), expected exactly 1")

    for faction in sorted({b.faction for b in buildings.values()}):
        op_id = economy.op_building.get(faction)
        if op_id is None:
            raise DataError(f"economy.yaml:economy: op_building has no entry for faction '{faction}'")
        if op_id not in buildings:
            raise DataError(f"economy.yaml:economy: op_building['{faction}'] = '{op_id}' is not a building")

    missing_income = [point_type for point_type in POINT_TYPES if point_type not in economy.point_income]
    if missing_income:
        raise DataError(
            f"economy.yaml:economy: point_income is missing point type(s) {missing_income}; "
            "every type a map may use must be priced"
        )

    _check_target_types_are_shootable(weapons, squads, buildings, neutral, generic_target_types)
    _check_requires_has_no_cycles(buildings, upgrades)
    _check_every_faction_can_play(squads, buildings, upgrades)
