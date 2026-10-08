"""
Tag-system invariants: object identity, mutation isolation, interface contract,
and round-trip annotation preservation.

These exist because value-only assertions let a shared-mutable-default bug
(`tag: TileTag = TileTag()`) go undetected: every default-constructed tile
shared one tag instance. The tests here assert identity and isolation directly,
parametrized over every tagged kind, so the bug class cannot reappear silently.
"""

import pytest
import stim

from qsnow.experiments.experiment import Experiment
from qsnow.experiments.squarepacking.game import SquarePackingExp
from qsnow.helpers.serialize import export_flake, import_flake, set_data_dir
from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.codes.rsc import SCTile
from qsnow.interface.models import Tag, TileSpec
from qsnow.interface.rules import InjectionRule, Ruleset


def _circuit() -> stim.Circuit:
    return stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z", distance=3, rounds=1
    )


def _square_packing() -> SquarePackingExp:
    return SquarePackingExp(chip=Chip(5, 5), tile=SCTile(distance=3))


# factories, so each call constructs a genuinely fresh instance
TAGGED_KINDS = {
    "Chip": lambda: Chip(2, 2),
    "LogicalTile": lambda: LogicalTile(_circuit()),
    "SCTile": lambda: SCTile(distance=3),
    "Experiment": lambda: Experiment(),
    "SquarePackingExp": _square_packing,
}

# kinds that go through export_json/import_json (generic Experiment included)
SERIALIZABLE_KINDS = ["Chip", "LogicalTile", "SCTile", "Experiment", "SquarePackingExp"]


@pytest.fixture(params=TAGGED_KINDS.keys())
def make(request):
    return TAGGED_KINDS[request.param]


@pytest.fixture
def data_dir(tmp_path):
    set_data_dir(tmp_path / "data")
    yield tmp_path / "data"
    set_data_dir()


class TestConstructionIsolation:
    """Two fresh instances must never share annotation state (the original bug)."""

    def test_instances_do_not_share_tag_object(self, make):
        assert make().tag is not make().tag

    def test_mutating_one_tag_does_not_leak(self, make):
        a, b = make(), make()
        a.tag.desc = "only a"
        a.tag.metadata["k"] = "v"
        assert b.tag.desc != "only a"
        assert "k" not in b.tag.metadata

    def test_tiles_do_not_share_spec_object(self):
        a, b = LogicalTile(_circuit()), LogicalTile(_circuit())
        assert a.spec is not b.spec
        a.spec.generator_args["k"] = "v"
        assert "k" not in b.spec.generator_args

    def test_dataclass_defaults_use_factories(self):
        assert Tag().metadata is not Tag().metadata
        assert TileSpec().generator_args is not TileSpec().generator_args


class TestInterfaceContract:
    """Every tagged kind exposes exactly a plain Tag; tiles expose exactly a TileSpec."""

    def test_tag_is_exactly_base_tag(self, make):
        obj = make()
        assert type(obj.tag) is Tag

    def test_tiles_expose_tile_spec(self):
        for factory in (TAGGED_KINDS["LogicalTile"], TAGGED_KINDS["SCTile"]):
            assert type(factory().spec) is TileSpec

    @pytest.mark.parametrize("kind", SERIALIZABLE_KINDS)
    def test_export_json_desc_contract(self, kind, data_dir):
        obj = TAGGED_KINDS[kind]()
        export_flake(obj, desc="set at export")
        assert obj.tag.desc == "set at export"


class TestCopyIsolation:
    def test_copy_has_independent_tag_and_spec(self):
        for factory in (TAGGED_KINDS["LogicalTile"], TAGGED_KINDS["SCTile"]):
            tile = factory()
            dupe = tile.copy()
            assert dupe.tag is not tile.tag
            assert dupe.spec is not tile.spec
            dupe.tag.desc = "copy only"
            dupe.spec.generator_args["k"] = "v"
            assert tile.tag.desc != "copy only"
            assert "k" not in tile.spec.generator_args

    def test_copy_carries_independent_ruleset(self):
        # regression: both copy() overrides rebuilt from the class-default ruleset,
        # so a ruleset set via the setter was silently lost at the next copy
        for factory in (TAGGED_KINDS["LogicalTile"], TAGGED_KINDS["SCTile"]):
            tile = factory()
            custom = Ruleset(
                [InjectionRule("H", "any", before=[], after=[], name="custom")]
            )
            tile.ruleset = custom
            dupe = tile.copy()
            assert [r.name for r in dupe.ruleset.rules] == ["custom"]
            assert dupe.ruleset is not tile.ruleset
            dupe.ruleset.add_rule(InjectionRule("CX", "any", before=[], after=[]))
            assert len(tile.ruleset.rules) == 1


class TestRoundTripPreservation:
    @pytest.mark.parametrize("kind", SERIALIZABLE_KINDS)
    def test_annotations_survive_export_import(self, kind, data_dir):
        obj = TAGGED_KINDS[kind]()
        obj.tag.name = "named"
        obj.tag.desc = "described"
        obj.tag.metadata = {"campaign": 1}

        restored = import_flake(export_flake(obj))

        assert restored.tag.name == "named"
        assert restored.tag.desc == "described"
        assert restored.tag.metadata == {"campaign": 1}

    def test_double_import_yields_independent_tags(self, data_dir):
        path = export_flake(SCTile(distance=3), desc="original")
        a, b = import_flake(path), import_flake(path)

        assert a.tag is not b.tag
        a.tag.desc = "mutated"
        assert b.tag.desc == "original"

    def test_sc_tile_spec_round_trips(self, data_dir):
        tile = SCTile(distance=3, rounds=2, task="memory_x")

        restored = import_flake(export_flake(tile))

        assert restored.spec.distance == 3
        assert restored.spec.rounds == 2
        assert restored.spec.generator_args == tile.spec.generator_args
