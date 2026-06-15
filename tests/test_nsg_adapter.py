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
