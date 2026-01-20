from analyzer.utils.structure_tools import (
    dictToDot,
    flatten,
    freeze,
    mergeUpdate,
    deepMerge,
    getWithMeta,
    globWithMeta,
    SimpleCache,
)


class MockItem:
    def __init__(self, name, metadata=None, children=None):
        self.name = name
        self.metadata = metadata or {}
        self.children = children or {}

    def __getitem__(self, key):
        return self.children[key]

    def __iter__(self):
        return iter(self.children)


def testDictToDot():
    d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    result = dict(dictToDot(d))
    assert result == {"a": 1, "b.c": 2, "b.d.e": 3}


def testFlatten():
    l = [1, [2, [3, 4], 5], 6]
    assert flatten(l) == [1, 2, 3, 4, 5, 6]
    assert flatten([1, "a"]) == [1, "a"]


def testFreeze():
    d = {"a": [1, 2], "b": {"c": 3}}
    frozen = freeze(d)
    assert isinstance(frozen, frozenset)
    assert hash(frozen)

    l = [1, 2]
    assert freeze(l) == frozenset({1, 2})
    assert freeze([1, 2]) == frozenset({1, 2})


def testMergeUpdate():
    a = {"a": 1, "b": {"x": 1}}
    b = {"b": {"y": 2}, "c": 3}

    mergeUpdate(a, b)
    assert a["a"] == 1
    assert a["b"]["x"] == 1
    assert a["b"]["y"] == 2
    assert a["c"] == 3


def testDeepMerge():
    a = {"a": 1, "b": {"x": 1}}
    b = {"b": {"y": 2}, "c": 3}

    res = deepMerge(a, b)
    assert res["b"]["y"] == 2
    assert "y" not in a["b"]


def testGetWithMeta():
    child = MockItem("child", {"level": 2})
    parent = MockItem("parent", {"level": 1}, {"child": child})

    res = getWithMeta(parent, "child")
    assert res.item is child
    assert res.metadata["level"] == 2
    assert res.metadata["name"] == "child"
    assert res.metadata.parents.parents["level"] == 1


def testGlobWithMeta():
    c1 = MockItem("c1", {"id": 1})
    c2 = MockItem("c2", {"id": 2})
    p = MockItem("root", {}, {"c1": c1, "c2": c2})

    res = globWithMeta(p, ["c*"])
    assert len(res) == 2
    names = sorted([x.item.name for x in res])
    assert names == ["c1", "c2"]

    res_c1 = next(x for x in res if x.item.name == "c1")
    assert res_c1.metadata["id"] == 1


def testSimpleCache():
    cache = SimpleCache(max_size=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3

    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache

    _ = cache["b"]
    cache["d"] = 4
    assert "c" not in cache
    assert "b" in cache
