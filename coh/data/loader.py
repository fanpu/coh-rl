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

_POINT_INCOME_KEYS = {
    "strategic",
    "munitions_low",
    "munitions_med",
    "munitions_high",
    "fuel_low",
    "fuel_med",
    "fuel_high",
    "victory",
}
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

    weapons = _load_weapons(directory / "weapons.yaml")
    squads = _load_squads(directory / "squads.yaml")
    buildings = _load_buildings(directory / "buildings.yaml")
    upgrades, squad_upgrades = _load_upgrades(directory / "upgrades.yaml")
    neutral = _load_neutral(directory / "neutral.yaml")
    economy = _load_economy(directory / "economy.yaml")

    _cross_validate(weapons, squads, buildings, upgrades, squad_upgrades)

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
    return Cost(manpower=float(d.get("manpower", 0)), munitions=float(d.get("munitions", 0)), fuel=float(d.get("fuel", 0)))


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


def _load_weapons(path: Path) -> dict[str, WeaponDef]:
    doc = _read_doc(path)
    body = _table(doc, "weapons", path)
    weapons: dict[str, WeaponDef] = {}
    for wid, raw in body.items():
        raw = raw or {}
        _check_keys(raw, _WEAPON_REQUIRED | _WEAPON_OPTIONAL, _WEAPON_REQUIRED, path, wid)
        burst_raw = raw["burst"]
        burst = None if burst_raw is None else _tuple(burst_raw, 2, path, wid, "burst")
        weapons[wid] = WeaponDef(
            id=wid,
            damage=float(raw["damage"]),
            ranges=_tuple(raw["ranges"], 3, path, wid, "ranges"),
            min_range=float(raw["min_range"]),
            accuracy=_tuple(raw["accuracy"], 3, path, wid, "accuracy"),
            penetration=_tuple(raw["penetration"], 3, path, wid, "penetration"),
            suppression=_tuple(raw["suppression"], 3, path, wid, "suppression"),
            cooldown=_tuple(raw["cooldown"], 2, path, wid, "cooldown"),
            cooldown_range_mult=_tuple(raw["cooldown_range_mult"], 3, path, wid, "cooldown_range_mult"),
            burst=burst,
            rate_of_fire=float(raw["rate_of_fire"]),
            reload=_tuple(raw["reload"], 2, path, wid, "reload"),
            reload_every=int(raw["reload_every"]),
            setup_time=float(raw.get("setup_time", 0.0)),
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
    return weapons


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

        members = int(raw["members"])
        kind = str(raw["kind"])
        loadout_raw = raw["loadout"]
        if not isinstance(loadout_raw, (list, tuple)):
            raise DataError(f"{path.name}:{sid}: 'loadout' must be a list")
        # Infantry and team weapons carry one weapon slot per model. A vehicle is a
        # single model mounting all of its weapons, so its loadout is any non-empty
        # list (main gun first) rather than one entry per member.
        if kind == "vehicle":
            if not loadout_raw:
                raise DataError(f"{path.name}:{sid}: vehicle 'loadout' must list at least one weapon")
        elif len(loadout_raw) != members:
            raise DataError(f"{path.name}:{sid}: 'loadout' must be a list of length members ({members})")
        loadout = tuple(str(w) for w in loadout_raw)

        suppression_raw = raw["suppression"]
        suppression = None
        if suppression_raw is not None:
            _check_keys(suppression_raw, _SUPPRESSION_KEYS, _SUPPRESSION_KEYS, path, sid)
            suppression = SuppressionDef(**{k: float(v) for k, v in suppression_raw.items()})

        squads[sid] = SquadDef(
            id=sid,
            faction=str(raw["faction"]),
            kind=kind,
            members=members,
            member_hp=float(raw["member_hp"]),
            loadout=loadout,
            target_type=str(raw["target_type"]),
            cost=_cost(raw["cost"], path, sid),
            population=int(raw["population"]),
            build_time=float(raw["build_time"]),
            upkeep_per_min=float(raw["upkeep_per_min"]),
            speed=float(raw["speed"]),
            rotation_deg_s=float(raw["rotation_deg_s"]),
            sight=float(raw["sight"]),
            capture_rate=float(raw["capture_rate"]),
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
            faction=str(raw["faction"]),
            cost=_cost(raw["cost"], path, bid),
            build_time=float(raw["build_time"]),
            hp=float(raw["hp"]),
            target_type=str(raw["target_type"]),
            footprint=_tuple(raw["footprint"], 2, path, bid, "footprint", cast=int),
            produces=tuple(str(x) for x in raw["produces"]),
            researches=tuple(str(x) for x in raw["researches"]),
            requires=tuple(str(x) for x in raw["requires"]),
            is_hq=bool(raw["is_hq"]),
            reinforce_radius=float(raw["reinforce_radius"]),
            sight=float(raw["sight"]),
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
            faction=str(raw["faction"]),
            cost=_cost(raw["cost"], path, uid),
            time=float(raw["time"]),
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
            time=float(raw["time"]),
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
            hp=float(raw["hp"]),
            target_type=str(raw["target_type"]),
            capacity=int(raw["capacity"]),
            footprint=_tuple(raw["footprint"], 2, path, nid, "footprint", cast=int),
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


def _cross_validate(
    weapons: dict[str, WeaponDef],
    squads: dict[str, SquadDef],
    buildings: dict[str, BuildingDef],
    upgrades: dict[str, UpgradeDef],
    squad_upgrades: dict[str, SquadUpgradeDef],
) -> None:
    for sid, sq in squads.items():
        for w in sq.loadout:
            if w and w not in weapons:
                raise DataError(f"squads.yaml:{sid}: loadout weapon '{w}' not found in weapons")
        for b in sq.builds:
            if b not in buildings:
                raise DataError(f"squads.yaml:{sid}: builds '{b}' not found in buildings")

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
