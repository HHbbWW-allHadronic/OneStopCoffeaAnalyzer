import pytest
import awkward as ak
from analyzer.core.columns import Column, ColumnCollection, coerceFields


class TestColumn:
    def test_initialization(self):
        c1 = Column("a.b")
        assert c1.fields == ("a", "b")
        c2 = Column(("a", "b"))
        assert c2.fields == ("a", "b")
        c3 = Column(c1)
        assert c3.fields == ("a", "b")

    def test_contains(self):
        c1 = Column("a.b.c")
        c2 = Column("a.b")
        c3 = Column("a.x")

        assert c2.contains(c1)
        assert not c1.contains(c2)
        assert not c1.contains(c3)

    def test_extract(self):
        events = ak.Array([{"a": {"b": 1}}, {"a": {"b": 2}}])
        col = Column("a.b")
        result = col.extract(events)
        assert ak.all(result == ak.Array([1, 2]))

    def test_magic_methods(self):
        c1 = Column("a")
        c2 = Column("b")

        c3 = c1 + c2
        assert c3.fields == ("a", "b")

        c4 = "root" + c1
        assert c4.fields == ("root", "a")

        assert c1 == Column("a")
        assert c1 != c2

        assert len(c3) == 2

        assert c3[0] == Column("a")
        assert c3[1:] == Column("b")

        assert str(c1) == "a"
        assert repr(c1) == "a"


class TestColumnCollection:
    def test_initialization(self):
        cols = ColumnCollection(["a.b", "c"])
        assert len(cols.columns) == 2
        assert Column("a.b") in cols.columns
        assert Column("c") in cols.columns

    def test_contains(self):
        cols = ColumnCollection(["a.b", "c"])
        assert cols.contains(Column("a.b.c"))

        assert not cols.contains(Column("a"))

        assert cols.contains(Column("a.b"))

    def test_intersect(self):
        c1 = ColumnCollection(["a", "b.c"])
        c2 = ColumnCollection(["a.x", "b"])

        c1 = ColumnCollection(["a", "b.c"])
        c2 = ColumnCollection(["a.x", "b"])

        res = c1.intersect(c2)
        assert Column("a") in res
        assert Column("b.c") in res
        assert len(res) == 2


def test_coerce_fields():
    assert coerceFields("a.b") == ("a", "b")
    c = Column("a")
    assert coerceFields(c) == ("a",)
    assert coerceFields(("a", "b")) == ("a", "b")


class TestTrackedColumns:
    @pytest.fixture
    def sample_events(self):
        return ak.Array(
            {
                "pt": [1, 2, 3],
                "eta": [0.1, 0.2, 0.3],
                "jets": {"pt": [[10], [20], [30]], "mass": [[1], [2], [3]]},
            }
        )

    def test_initialization_from_events(self, sample_events):
        from analyzer.core.columns import TrackedColumns, EventBackend

        tc = TrackedColumns.fromEvents(
            sample_events, metadata={}, backend=EventBackend.coffea_imm, provenance=0
        )
        assert tc._current_provenance == 0
        assert Column("pt") in tc._column_provenance
        assert Column("jets") in tc._column_provenance

    def test_getitem_setitem(self, sample_events):
        from analyzer.core.columns import TrackedColumns, EventBackend

        tc = TrackedColumns.fromEvents(
            sample_events, metadata={}, backend=EventBackend.coffea_imm, provenance=0
        )

        assert ak.all(tc["pt"] == sample_events.pt)
        assert ak.all(tc["jets.pt"] == sample_events.jets.pt)

        new_col = ak.Array([4, 5, 6])
        tc["new_col"] = new_col
        assert ak.all(tc["new_col"] == new_col)
        assert tc._column_provenance[Column("new_col")] == 0

        tc._current_provenance = 1
        tc["pt"] = new_col
        assert tc._column_provenance[Column("pt")] == 1
        assert ak.all(tc["pt"] == new_col)

    def test_updated_columns(self, sample_events):
        from analyzer.core.columns import TrackedColumns, EventBackend

        tc1 = TrackedColumns.fromEvents(
            sample_events, metadata={}, backend=EventBackend.coffea_imm, provenance=0
        )
        tc2 = tc1.copy()

        tc2 = tc1.copy()

        assert len(tc2.updatedColumns(tc1)) == 0

        tc2._current_provenance = 1
        tc2["pt"] = ak.Array([10, 20, 30])

        updates = tc2.updatedColumns(tc1)
        assert Column("pt") in updates
        assert len(updates) == 1

    def test_copy_integrity(self, sample_events):
        from analyzer.core.columns import TrackedColumns, EventBackend

        metadata = {"meta_key": "meta_val"}
        tc_orig = TrackedColumns.fromEvents(
            sample_events,
            metadata=metadata,
            backend=EventBackend.coffea_imm,
            provenance=0,
        )
        tc_orig.pipeline_data["test_list"] = [1, 2, 3]
        tc_orig.pipeline_data["test_dict"] = {"a": 1}

        tc_copy = tc_orig.copy()

        assert tc_copy._current_provenance == tc_orig._current_provenance
        assert tc_copy.metadata == tc_orig.metadata
        assert tc_copy.pipeline_data == tc_orig.pipeline_data

        tc_copy.pipeline_data["test_list"].append(4)
        tc_copy.pipeline_data["test_dict"]["b"] = 2

        assert tc_orig.pipeline_data["test_list"] == [1, 2, 3]
        assert "b" not in tc_orig.pipeline_data["test_dict"]
        assert tc_copy.pipeline_data["test_list"] == [1, 2, 3, 4]
        assert tc_copy.pipeline_data["test_dict"]["b"] == 2

        tc_copy.metadata["new_key"] = "new_val"
        assert "new_key" not in tc_orig.metadata

        tc_copy._current_provenance = 1
        new_pt = ak.Array([9, 9, 9])
        tc_copy["pt"] = new_pt

        assert ak.all(tc_copy["pt"] == new_pt)
        assert ak.all(tc_orig["pt"] == sample_events.pt)

        assert tc_copy._column_provenance[Column("pt")] == 1
        assert tc_orig._column_provenance[Column("pt")] == 0

    def test_allowed_inputs_outputs(self, sample_events):
        from analyzer.core.columns import TrackedColumns, EventBackend

        tc = TrackedColumns.fromEvents(
            sample_events,
            metadata={},
            backend=EventBackend.coffea_imm,
            provenance=0,
        )

        with tc.allowedInputs(["pt"]):
            _ = tc["pt"]
            with pytest.raises(RuntimeError):
                _ = tc["eta"]

        with tc.allowedOutputs(["new_col"]):
            tc["new_col"] = ak.Array([1, 2, 3])
            with pytest.raises(RuntimeError):
                tc["forbidden_col"] = ak.Array([1, 2, 3])


class TestUtils:
    def test_set_column(self):
        from analyzer.core.columns import setColumn

        events = ak.Array([{"a": {"b": 1}}, {"a": {"b": 2}}])

        events = setColumn(events, "a.b", ak.Array([3, 4]))
        assert ak.all(events.a.b == ak.Array([3, 4]))

        events = setColumn(events, "x.y", ak.Array([5, 6]))
        assert ak.all(events.x.y == ak.Array([5, 6]))

    def test_get_all_columns(self):
        from analyzer.core.columns import getAllColumns

        events = ak.Array([{"a": {"b": 1, "c": 2}, "d": 3}])
        cols = getAllColumns(events.layout)

        assert Column("a.b") in cols
        assert Column("a.c") in cols
        assert Column("d") in cols
        assert Column("a") in cols

    def test_merge_columns(self):
        from analyzer.core.columns import mergeColumns, TrackedColumns, EventBackend

        events = ak.Array({"a": [1], "b": [2]})
        tc1 = TrackedColumns.fromEvents(events, {}, EventBackend.coffea_imm, 0)
        tc2 = TrackedColumns.fromEvents(events, {}, EventBackend.coffea_imm, 0)

        tc2._current_provenance = 1
        tc2["b"] = ak.Array([3])

        merged = mergeColumns([tc1, tc2])
        assert merged._column_provenance[Column("b")] == 1
        assert ak.all(merged["b"] == ak.Array([3]))

    def test_add_selection(self):
        from analyzer.core.columns import addSelection, TrackedColumns, EventBackend

        events = ak.Array({"a": [1]})
        tc = TrackedColumns.fromEvents(events, {}, EventBackend.coffea_imm, 0)

        addSelection(tc, "test_cut", ak.Array([True]))

        assert "Selections" in tc.pipeline_data
        assert "test_cut" in tc.pipeline_data["Selections"]
        assert Column("Selection.test_cut") in tc._column_provenance
