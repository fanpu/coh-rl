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
    assert set(data.weapons) == {"rifle", "bar", "hmg", "tank_gun", "at_gun", "mortar"}
    assert set(data.squads) == {
        "rifles",
        "engineers",
        "hmg_team",
        "at_team",
        "mortar_team",
        "tank",
        "pioneers",
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
