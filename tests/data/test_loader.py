import shutil
from pathlib import Path

import pytest
import yaml

from coh.data.loader import DataError, load_game_data
from coh.data.schema import TargetMods

FIXTURES = Path(__file__).parent / "fixtures"


def _copy_fixtures(tmp_path: Path) -> Path:
    dest = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, dest)
    return dest


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, doc: dict) -> None:
    with path.open("w") as f:
        yaml.safe_dump(doc, f)


def test_loads_fixtures():
    data = load_game_data(FIXTURES)
    assert set(data.weapons) == {
        "rifle",
        "bar",
        "hmg",
        "tank_gun",
        "at_gun",
        "mortar",
        "hull_gun",
        "at_gun_duel",
        "duel_tank_gun",
        "garrison_rifle",
    }
    assert set(data.squads) == {
        "rifles",
        "engineers",
        "hmg_team",
        "at_team",
        "mortar_team",
        "tank",
        "pioneers",
        "assault_gun",
        "at_team_duel",
        "duel_tank",
        "garrison_rifles",
    }
    assert set(data.buildings) == {
        "hq_us",
        "hq_wehr",
        "barracks",
        "op_us",
        "op_wehr",
        "tank_depot",
        "panzer_command",
    }
    assert set(data.upgrades) == {"research_1", "phase_2", "supply_yard"}
    assert set(data.squad_upgrades) == {"bar_upgrade", "bar"}
    assert set(data.neutral) == {"house", "house_small", "house_large", "barn"}


def test_rifle_accuracy():
    data = load_game_data(FIXTURES)
    assert data.weapons["rifle"].accuracy == (1.0, 0.5, 0.25)


def test_missing_target_type_falls_back_to_identity_mods():
    data = load_game_data(FIXTURES)
    rifle = data.weapons["rifle"]
    assert rifle.vs("armour_tank") == TargetMods(damage=0.01)
    assert rifle.vs("some_unknown_type") == TargetMods()


def test_tank_gun_target_table_penetration():
    data = load_game_data(FIXTURES)
    mods = data.weapons["tank_gun"].vs("armour_tank")
    assert mods.penetration == 0.5
    assert mods.rear_penetration == 2.0
    mods = data.weapons["at_gun"].vs("armour_tank")
    assert mods.penetration == 0.5
    assert mods.rear_penetration == 2.0


def test_missing_cover_falls_back_to_identity_mods():
    data = load_game_data(FIXTURES)
    from coh.data.schema import CoverMods

    assert data.weapons["rifle"].cover("open") == CoverMods()


def test_no_arg_load_raises_data_error_when_packaged_tables_missing():
    # coh/data/tables/ only has a .gitkeep until a later task populates it.
    with pytest.raises(DataError, match=r"coh[/\\]data[/\\]tables"):
        load_game_data()


# --- validation rule tests --------------------------------------------------------


def test_unknown_key_raises(tmp_path):
    fixtures = _copy_fixtures(tmp_path)
    path = fixtures / "weapons.yaml"
    doc = _load_yaml(path)
    doc["weapons"]["rifle"]["not_a_real_field"] = 1
    _dump_yaml(path, doc)

    with pytest.raises(DataError) as exc:
        load_game_data(fixtures)
    assert "weapons.yaml" in str(exc.value)
    assert "rifle" in str(exc.value)


def test_bad_tuple_length_raises(tmp_path):
    fixtures = _copy_fixtures(tmp_path)
    path = fixtures / "weapons.yaml"
    doc = _load_yaml(path)
    doc["weapons"]["rifle"]["accuracy"] = [1.0, 0.5]
    _dump_yaml(path, doc)

    with pytest.raises(DataError) as exc:
        load_game_data(fixtures)
    assert "weapons.yaml" in str(exc.value)
    assert "rifle" in str(exc.value)
    assert "accuracy" in str(exc.value)


def test_missing_loadout_weapon_raises(tmp_path):
    fixtures = _copy_fixtures(tmp_path)
    path = fixtures / "squads.yaml"
    doc = _load_yaml(path)
    doc["squads"]["rifles"]["loadout"] = ["not_a_weapon", "rifle", "rifle", "rifle"]
    _dump_yaml(path, doc)

    with pytest.raises(DataError) as exc:
        load_game_data(fixtures)
    assert "squads.yaml" in str(exc.value)
    assert "rifles" in str(exc.value)
    assert "not_a_weapon" in str(exc.value)


def test_missing_squad_upgrade_weapon_raises(tmp_path):
    fixtures = _copy_fixtures(tmp_path)
    path = fixtures / "upgrades.yaml"
    doc = _load_yaml(path)
    doc["squad_upgrades"]["bar_upgrade"]["weapon"] = "not_a_weapon"
    _dump_yaml(path, doc)

    with pytest.raises(DataError) as exc:
        load_game_data(fixtures)
    assert "upgrades.yaml" in str(exc.value)
    assert "bar_upgrade" in str(exc.value)
    assert "not_a_weapon" in str(exc.value)


def test_missing_produces_id_raises(tmp_path):
    fixtures = _copy_fixtures(tmp_path)
    path = fixtures / "buildings.yaml"
    doc = _load_yaml(path)
    doc["buildings"]["barracks"]["produces"] = ["not_a_squad"]
    _dump_yaml(path, doc)

    with pytest.raises(DataError) as exc:
        load_game_data(fixtures)
    assert "buildings.yaml" in str(exc.value)
    assert "barracks" in str(exc.value)
    assert "not_a_squad" in str(exc.value)


def test_missing_researches_id_raises(tmp_path):
    fixtures = _copy_fixtures(tmp_path)
    path = fixtures / "buildings.yaml"
    doc = _load_yaml(path)
    doc["buildings"]["hq_us"]["researches"] = ["not_an_upgrade"]
    _dump_yaml(path, doc)

    with pytest.raises(DataError) as exc:
        load_game_data(fixtures)
    assert "buildings.yaml" in str(exc.value)
    assert "hq_us" in str(exc.value)
    assert "not_an_upgrade" in str(exc.value)


def test_missing_requires_id_raises(tmp_path):
    fixtures = _copy_fixtures(tmp_path)
    path = fixtures / "buildings.yaml"
    doc = _load_yaml(path)
    doc["buildings"]["barracks"]["requires"] = ["not_a_building"]
    _dump_yaml(path, doc)

    with pytest.raises(DataError) as exc:
        load_game_data(fixtures)
    assert "buildings.yaml" in str(exc.value)
    assert "barracks" in str(exc.value)
    assert "not_a_building" in str(exc.value)


def test_missing_builds_id_raises(tmp_path):
    fixtures = _copy_fixtures(tmp_path)
    path = fixtures / "squads.yaml"
    doc = _load_yaml(path)
    doc["squads"]["engineers"]["builds"] = ["not_a_building"]
    _dump_yaml(path, doc)

    with pytest.raises(DataError) as exc:
        load_game_data(fixtures)
    assert "squads.yaml" in str(exc.value)
    assert "engineers" in str(exc.value)
    assert "not_a_building" in str(exc.value)


def test_cover_table_key_outside_cover_types_raises(tmp_path):
    fixtures = _copy_fixtures(tmp_path)
    path = fixtures / "weapons.yaml"
    doc = _load_yaml(path)
    doc["weapons"]["rifle"]["cover_table"]["not_a_cover_type"] = {"accuracy": 1.0}
    _dump_yaml(path, doc)

    with pytest.raises(DataError) as exc:
        load_game_data(fixtures)
    assert "weapons.yaml" in str(exc.value)
    assert "rifle" in str(exc.value)
    assert "not_a_cover_type" in str(exc.value)


def test_zero_is_hq_buildings_for_a_faction_raises(tmp_path):
    fixtures = _copy_fixtures(tmp_path)
    path = fixtures / "buildings.yaml"
    doc = _load_yaml(path)
    doc["buildings"]["hq_us"]["is_hq"] = False
    _dump_yaml(path, doc)

    with pytest.raises(DataError) as exc:
        load_game_data(fixtures)
    assert "buildings.yaml" in str(exc.value)
    assert "us" in str(exc.value)


def test_two_is_hq_buildings_for_a_faction_raises(tmp_path):
    fixtures = _copy_fixtures(tmp_path)
    path = fixtures / "buildings.yaml"
    doc = _load_yaml(path)
    doc["buildings"]["barracks"]["is_hq"] = True
    _dump_yaml(path, doc)

    with pytest.raises(DataError) as exc:
        load_game_data(fixtures)
    assert "buildings.yaml" in str(exc.value)
    assert "us" in str(exc.value)


# --- sanity rules the sim relies on -------------------------------------------------
#
# One case per rule. Each mutates a loadable fixture set into exactly one
# malformed shape and asserts the error names the file and the offending key,
# so a scraped table set fails at load rather than halfway through a match.


def _set(doc, path_keys, value):
    node = doc
    for key in path_keys[:-1]:
        node = node[key]
    node[path_keys[-1]] = value


def _squad(sid, field_name, value):
    return "squads.yaml", lambda doc: _set(doc, ["squads", sid, field_name], value)


def _weapon(wid, field_name, value):
    return "weapons.yaml", lambda doc: _set(doc, ["weapons", wid, field_name], value)


def _building(bid, field_name, value):
    return "buildings.yaml", lambda doc: _set(doc, ["buildings", bid, field_name], value)


def _vehicle_with_three_members(doc):
    doc["squads"]["tank"]["members"] = 3
    doc["squads"]["tank"]["loadout"] = ["tank_gun", "", ""]


def _squad_with_no_members(doc):
    doc["squads"]["rifles"]["members"] = 0
    doc["squads"]["rifles"]["loadout"] = []


def _no_capture_capable_unit(doc):
    for sid in ("rifles", "engineers"):
        doc["squads"][sid]["capture_rate"] = 0


BAD_TABLES = [
    # (case id, file, mutation, substrings the message must contain)
    ("squad_kind_enum", *_squad("rifles", "kind", "cavalry"), ["squads.yaml", "rifles", "kind"]),
    ("squad_faction_enum", *_squad("rifles", "faction", "soviet"), ["squads.yaml", "rifles", "faction"]),
    ("building_faction_enum", *_building("barracks", "faction", "soviet"), ["buildings.yaml", "barracks", "faction"]),
    ("vehicle_members", "squads.yaml", _vehicle_with_three_members, ["squads.yaml", "tank", "members"]),
    ("members_at_least_one", "squads.yaml", _squad_with_no_members, ["squads.yaml", "rifles", "members"]),
    (
        "team_weapon_slot0_empty",
        *_squad("hmg_team", "loadout", ["", "", ""]),
        ["squads.yaml", "hmg_team", "loadout"],
    ),
    (
        "team_weapon_slot0_has_no_setup_time",
        *_squad("hmg_team", "loadout", ["rifle", "", ""]),
        ["squads.yaml", "hmg_team", "setup_time"],
    ),
    ("negative_cost", *_squad("rifles", "cost", {"manpower": -240}), ["squads.yaml", "rifles", "cost"]),
    ("negative_build_time", *_squad("rifles", "build_time", -1), ["squads.yaml", "rifles", "build_time"]),
    ("negative_hp", *_building("barracks", "hp", -600), ["buildings.yaml", "barracks", "hp"]),
    ("footprint_below_1x1", *_building("barracks", "footprint", [0, 3]), ["buildings.yaml", "barracks", "footprint"]),
    (
        "neutral_footprint_below_1x1",
        "neutral.yaml",
        lambda doc: _set(doc, ["neutral", "house", "footprint"], [2, 0]),
        ["neutral.yaml", "house", "footprint"],
    ),
    ("ranges_not_ascending", *_weapon("rifle", "ranges", [10, 10, 30]), ["weapons.yaml", "rifle", "ranges"]),
    ("min_range_beyond_max_range", *_weapon("rifle", "min_range", 30), ["weapons.yaml", "rifle", "min_range"]),
    ("negative_accuracy", *_weapon("rifle", "accuracy", [1.0, -0.5, 0.25]), ["weapons.yaml", "rifle", "accuracy"]),
    (
        "negative_penetration",
        *_weapon("rifle", "penetration", [1.0, 1.0, -1.0]),
        ["weapons.yaml", "rifle", "penetration"],
    ),
    (
        "negative_cover_mod",
        "weapons.yaml",
        lambda doc: _set(doc, ["weapons", "rifle", "cover_table", "heavy", "accuracy"], -0.5),
        ["weapons.yaml", "rifle", "heavy"],
    ),
    (
        "negative_target_mod",
        "weapons.yaml",
        lambda doc: _set(doc, ["weapons", "rifle", "target_table", "armour_tank", "damage"], -1.0),
        ["weapons.yaml", "rifle", "armour_tank"],
    ),
    (
        "unshootable_squad_target_type",
        *_squad("rifles", "target_type", "spaceship"),
        ["squads.yaml", "rifles", "spaceship"],
    ),
    (
        "unshootable_neutral_target_type",
        "neutral.yaml",
        lambda doc: _set(doc, ["neutral", "house", "target_type"], "spaceship"),
        ["neutral.yaml", "house", "spaceship"],
    ),
    (
        "op_building_not_a_building",
        "economy.yaml",
        lambda doc: _set(doc, ["economy", "op_building", "us"], "not_a_building"),
        ["economy.yaml", "op_building", "not_a_building"],
    ),
    (
        "point_income_missing_a_point_type",
        "economy.yaml",
        lambda doc: doc["economy"]["point_income"].pop("fuel_med"),
        ["economy.yaml", "point_income", "fuel_med"],
    ),
    (
        "faction_has_no_builder_squad",
        *_building("hq_us", "produces", []),
        ["buildings.yaml", "us", "build"],
    ),
    (
        "faction_cannot_field_a_capture_capable_squad",
        "squads.yaml",
        _no_capture_capable_unit,
        ["us", "capture"],
    ),
    (
        "requires_cycle",
        *_building("barracks", "requires", ["tank_depot"]),
        ["requires", "cycle"],
    ),
]


@pytest.mark.parametrize(
    ("filename", "mutate", "expected"),
    [pytest.param(f, m, e, id=case_id) for case_id, f, m, e in BAD_TABLES],
)
def test_malformed_table_is_rejected(tmp_path, filename, mutate, expected):
    fixtures = _copy_fixtures(tmp_path)
    path = fixtures / filename
    doc = _load_yaml(path)
    mutate(doc)
    _dump_yaml(path, doc)

    with pytest.raises(DataError) as exc:
        load_game_data(fixtures)
    message = str(exc.value)
    for fragment in expected:
        assert fragment in message, f"{fragment!r} missing from {message!r}"


def test_generic_target_types_makes_the_identity_fallback_deliberate(tmp_path):
    """A target type no weapon has a row for is an error unless declared."""
    fixtures = _copy_fixtures(tmp_path)
    squads_path = fixtures / "squads.yaml"
    doc = _load_yaml(squads_path)
    doc["squads"]["rifles"]["target_type"] = "spaceship"
    _dump_yaml(squads_path, doc)
    with pytest.raises(DataError):
        load_game_data(fixtures)

    weapons_path = fixtures / "weapons.yaml"
    doc = _load_yaml(weapons_path)
    doc["generic_target_types"] = ["spaceship"]
    _dump_yaml(weapons_path, doc)
    data = load_game_data(fixtures)
    assert data.squads["rifles"].target_type == "spaceship"
