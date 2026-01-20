import pytest
from analyzer.utils.querying import (
    lookup,
    deepLookup,
    Pattern,
    PatternMode,
    PatternAnd,
    PatternOr,
    PatternNot,
    DeepPattern,
    gatherByCapture,
    NO_MATCH,
)
from analyzer.utils.structure_tools import ItemWithMeta


class MockObj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def testLookup():
    obj = MockObj(a=1, b=2)
    assert lookup(obj, "a") == 1

    assert lookup(obj, "b") == 2

    with pytest.raises(AttributeError):
        lookup(obj, "c")


def testDeepLookup():
    child = MockObj(x=10)
    parent = MockObj(child=child)

    assert deepLookup(parent, ["child", "x"]) == 10


def testPatternModes():
    p_glob = Pattern("test*", mode=PatternMode.GLOB)
    assert p_glob.match("testing")
    assert not p_glob.match("other")

    p_re = Pattern("^test.*", mode=PatternMode.REGEX)
    assert p_re.match("testing")
    assert not p_re.match("other")

    p_lit = Pattern("test", mode=PatternMode.LITERAL)
    assert p_lit.match("test")
    assert not p_lit.match("testing")

    p_any = Pattern("", mode=PatternMode.ANY)
    assert p_any.match("anything")
    assert p_any.match(None)


def testPatternCapture():
    p = Pattern("val", mode=PatternMode.LITERAL)
    assert p.capture("val") == "val"
    assert p.capture("other") is NO_MATCH


def testLogicalPatterns():
    p1 = Pattern("a", mode=PatternMode.GLOB)
    p2 = Pattern("b", mode=PatternMode.GLOB)

    p_and = PatternAnd([p1, p2])
    assert not p_and.match("a")

    p3 = Pattern("*a*", mode=PatternMode.GLOB)
    p4 = Pattern("*b*", mode=PatternMode.GLOB)
    p_and_2 = PatternAnd([p3, p4])
    assert p_and_2.match("ab")
    assert not p_and_2.match("a")

    # OR
    p_or = PatternOr([p1, p2])
    assert p_or.match("a")
    assert p_or.match("b")
    assert not p_or.match("c")

    p_not = PatternNot(p1)
    assert not p_not.match("a")
    assert p_not.match("b")


def testDeepPattern():
    dp = DeepPattern(key=("data",), pattern=Pattern("val", mode=PatternMode.LITERAL))
    obj = MockObj(data="val")
    assert dp.match(obj)

    obj_bad = MockObj(data="other")
    assert not dp.match(obj_bad)


def testGatherByCapture():
    items = [
        ItemWithMeta("i1", {"type": "A", "id": 1}),
        ItemWithMeta("i2", {"type": "A", "id": 2}),
        ItemWithMeta("i3", {"type": "B", "id": 3}),
    ]

    p = DeepPattern(key=("type",), pattern=Pattern("*", mode=PatternMode.GLOB))

    res = gatherByCapture(p, items)
    assert len(res) == 2

    group_A = next(x for x in res if x.capture == {("type",): "A"})
    assert len(group_A.items) == 2

    group_B = next(x for x in res if x.capture == {("type",): "B"})
    assert len(group_B.items) == 1
