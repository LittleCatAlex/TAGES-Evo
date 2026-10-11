import pytest
from tages.labels.lithology import classify, classify_l2, L1_NAMES, L2_NAMES
from tages.labels.hydro_codes import hydro_to_l1, hydro_to_l2

L1_CASES = [
    ("回填土", 1),
    ("砂質泥岩", 6),
    ("泥質砂岩", 6),
    ("砂岩", 6),
    ("板岩", 7),
    ("粉質黏土", 2),
    ("粉質砂", 3),
    ("砂質粉土", 2),
    ("礫石層夾砂", 4),
    ("黏土含少量砂", 2),
    ("砂與泥互層", 5),
    ("", 0),
    (None, 0),
]

L2_CASES = [
    ("回填土", 1),
    ("黏土", 2),
    ("粉砂", 3),
    ("細砂", 4),
    ("粗砂", 5),
    ("砂質泥", 6),
    ("砂與泥互層", 6),
    ("礫石層", 7),
    ("泥岩", 8),
    ("砂質泥岩", 8),
    ("砂岩", 9),
    ("泥質砂岩", 9),
    ("板岩", 10),
    ("花崗岩", 10),
]

@pytest.mark.parametrize("desc,expected", L1_CASES)
def test_classify_l1(desc, expected):
    assert classify(desc, level="l1") == expected

@pytest.mark.parametrize("desc,expected", L2_CASES)
def test_classify_l2(desc, expected):
    assert classify_l2(desc) == expected

def test_hydro_codes():
    assert hydro_to_l2("M") == 2         # ClayMud
    assert hydro_to_l2("Z") == 3         # Silt
    assert hydro_to_l2("fS") == 4        # FineSand
    assert hydro_to_l2("mS") == 5        # CoarseSand
    assert hydro_to_l2("MvfS") == 6      # SandMudMixed
    assert hydro_to_l2("cG") == 7        # Gravel
    assert hydro_to_l2("ms") == 8        # Mudstone
    assert hydro_to_l2("ss") == 9        # Sandstone
    assert hydro_to_l2("br") == 10       # Basement
    assert hydro_to_l1("fS") == 3        # L1 Sand
