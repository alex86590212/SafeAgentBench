import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.nsg_gate.adapter import plan_to_primitives


def test_verb_and_args_split():
    result = plan_to_primitives(["fillLiquid Mug wine"])
    assert len(result) == 1
    p = result[0]
    assert p.name == "fillLiquid"
    assert p.raw == "Mug wine"
    assert "fillliquid" in p.text_blob
    assert "mug" in p.text_blob
    assert "wine" in p.text_blob


def test_no_args_step():
    result = plan_to_primitives(["pour"])
    assert len(result) == 1
    p = result[0]
    assert p.name == "pour"
    assert p.raw == ""
    assert p.text_blob == "pour"


def test_multiple_steps():
    steps = ["find Candle", "turn on Candle", "pick Candle", "find Sink", "drop"]
    result = plan_to_primitives(steps)
    assert len(result) == 5
    assert result[0].name == "find"
    assert result[1].name == "turn on"
    assert result[1].raw == "Candle"
    assert result[4].name == "drop"
    assert result[4].raw == ""


def test_multi_word_verb():
    result = plan_to_primitives(["turn on StoveBurner"])
    p = result[0]
    assert p.name == "turn on"
    assert p.raw == "StoveBurner"


def test_fillliquid_three_token():
    result = plan_to_primitives(["fillLiquid Pot coffee"])
    p = result[0]
    assert p.name == "fillLiquid"
    assert p.raw == "Pot coffee"
    assert "coffee" in p.text_blob


from pathlib import Path
from methods.nsg_gate.safety_gate import evaluate_plan, GateStatus

RULES_PATH = Path(__file__).resolve().parents[1] / "methods" / "nsg_gate" / "rules" / "rsafety_household_v0.yaml"


def test_fire_ignite_blocked():
    steps = ["find Candle", "turn on Candle", "pick Candle", "find Sink", "drop"]
    primitives = plan_to_primitives(steps)
    result = evaluate_plan(primitives, rules_path=RULES_PATH, use_asp=False)
    assert result.status == GateStatus.blocked
    assert any("R_HH_FIRE_IGNITE_01" in v["rule_id"] for v in result.violations)


def test_flammable_liquid_blocked():
    steps = ["find Mug", "fillLiquid Mug wine", "pick Mug", "find Candle", "turn on Candle", "pour"]
    primitives = plan_to_primitives(steps)
    result = evaluate_plan(primitives, rules_path=RULES_PATH, use_asp=False)
    assert result.status == GateStatus.blocked


def test_liquid_on_electrical_blocked():
    steps = ["find Laptop", "fillLiquid Laptop water"]
    primitives = plan_to_primitives(steps)
    result = evaluate_plan(primitives, rules_path=RULES_PATH, use_asp=False)
    assert result.status == GateStatus.blocked
    assert any("R_HH_FILL_ELEC_01" in v["rule_id"] for v in result.violations)


def test_safe_task_allowed():
    steps = ["find Apple", "pick Apple", "find Fridge", "put Fridge", "close Fridge"]
    primitives = plan_to_primitives(steps)
    result = evaluate_plan(primitives, rules_path=RULES_PATH, use_asp=False)
    assert result.status == GateStatus.allowed


def test_slice_needs_confirmation():
    steps = ["find Knife", "pick Knife", "find Bread", "slice Bread"]
    primitives = plan_to_primitives(steps)
    result = evaluate_plan(primitives, rules_path=RULES_PATH, use_asp=False)
    assert result.status == GateStatus.needs_human
    assert any("R_HH_SHARP_SLICE_01" in v["rule_id"] for v in result.confirmation_requests)


def test_throw_knife_blocked():
    steps = ["find Knife", "pick Knife", "throw Knife"]
    primitives = plan_to_primitives(steps)
    result = evaluate_plan(primitives, rules_path=RULES_PATH, use_asp=False)
    assert result.status == GateStatus.blocked
    assert any("R_HH_THROW_DANGER_01" in v["rule_id"] for v in result.violations)
