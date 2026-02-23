import pytest
import awkward as ak
from analyzer.core.columns import Column, ColumnCollection, coerceFields


class TestColumn:
    def testInitialization(self):
        c1 = Column("a.b")
        assert c1.fields == ("a", "b")
        c2 = Column(("a", "b"))
        assert c2.fields == ("a", "b")
        c3 = Column(c1)
        assert c3.fields == ("a", "b")

    def testContains(self):
        c1 = Column("a.b.c")
        c2 = Column("a.b")
        c3 = Column("a.x")

        assert c2.contains(c1)
        assert not c1.contains(c2)
        assert not c1.contains(c3)

    def testExtract(self):
        events = ak.Array([{"a": {"b": 1}}, {"a": {"b": 2}}])
        col = Column("a.b")
        result = col.extract(events)
        assert ak.all(result == ak.Array([1, 2]))

    def testMagicMethods(self):
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
    def testInitialization(self):
        cols = ColumnCollection(["a.b", "c"])
        assert len(cols.columns) == 2
        assert Column("a.b") in cols.columns
        assert Column("c") in cols.columns

    def testContains(self):
        cols = ColumnCollection(["a.b", "c"])
        assert cols.contains(Column("a.b.c"))

        assert not cols.contains(Column("a"))

        assert cols.contains(Column("a.b"))

    def testIntersect(self):
        c1 = ColumnCollection(["a", "b.c"])
        c2 = ColumnCollection(["a.x", "b"])

        c1 = ColumnCollection(["a", "b.c"])
        c2 = ColumnCollection(["a.x", "b"])

        res = c1.intersect(c2)
        assert Column("a") in res
        assert Column("b.c") in res
        assert len(res) == 2


def testCoerceFields():
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

    def testInitializationFromEvents(self, sample_events):
        from analyzer.core.columns import TrackedColumns, EventBackend

        tc = TrackedColumns.fromEvents(
            sample_events, metadata={}, backend=EventBackend.coffea_imm, provenance=0
        )
        assert tc._current_provenance == 0
        assert Column("pt") in tc._column_provenance
        assert Column("jets") in tc._column_provenance

    def testGetItemSetitem(self, sample_events):
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

    def testUpdatedColumns(self, sample_events):
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

    def testCopyIntegrity(self, sample_events):
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

    def testAllowedInputsOutputs(self, sample_events):
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

    def testParentColumnProvenanceHashes(self, sample_events):
        from analyzer.core.columns import TrackedColumns, EventBackend, Column

        tc = TrackedColumns.fromEvents(
            sample_events, metadata={}, backend=EventBackend.coffea_imm, provenance=0
        )

        # 1) Simple update on an existing nested column
        tc._current_provenance = 1
        tc["jets.pt"] = ak.Array([[100], [200], [300]])

        assert tc._column_provenance[Column("jets.pt")] == 1
        jets_hash1 = tc._column_provenance[Column("jets")]
        assert jets_hash1 == hash((0, 1))

        # 2) Second update on a sibling child of the same parent
        tc._current_provenance = 2
        tc["jets.mass"] = ak.Array([[10], [20], [30]])

        assert tc._column_provenance[Column("jets.mass")] == 2
        jets_hash2 = tc._column_provenance[Column("jets")]
        # Important: the hash incorporates the previous state of the parent
        # (which encapsulates all previous child updates), NOT just the original 0
        assert jets_hash2 == hash((jets_hash1, 2))
        assert jets_hash2 != hash(
            (0, 2)
        )  # Failing condition exactly as user experienced it

        # 3) Updating a non-existent deeply nested column branches
        tc._current_provenance = 3
        tc["a.b.c"] = ak.Array([1, 2, 3])

        assert tc._column_provenance[Column("a.b.c")] == 3
        b_hash1 = tc._column_provenance[Column("a.b")]
        a_hash1 = tc._column_provenance[Column("a")]

        assert b_hash1 == hash((None, 3))
        assert a_hash1 == hash((None, 3))

        # 4) Updating another branch on the newly created tree
        tc._current_provenance = 4
        tc["a.b.d"] = ak.Array([4, 5, 6])

        b_hash2 = tc._column_provenance[Column("a.b")]
        a_hash2 = tc._column_provenance[Column("a")]

        assert b_hash2 == hash((b_hash1, 4))
        assert a_hash2 == hash((a_hash1, 4))

        # 5) Checking parents up the chain updating separately
        tc._current_provenance = 5
        tc["a.x"] = ak.Array([7, 8, 9])

        a_hash3 = tc._column_provenance[Column("a")]
        assert a_hash3 == hash((a_hash2, 5))

        # 6) Reassigning the parent entirely overrides its hash and its children's hashes with the active provenance
        tc._current_provenance = 6
        tc["jets"] = ak.Array(
            {"pt": [[1], [2], [3]], "mass": [[2], [3], [4]], "energy": [[3], [4], [5]]}
        )
        assert tc._column_provenance[Column("jets")] == 6
        assert tc._column_provenance[Column("jets.pt")] == 6
        assert tc._column_provenance[Column("jets.mass")] == 6
        assert tc._column_provenance[Column("jets.energy")] == 6

        # 7) Updating child again after parent reassignment correctly uses the reassigned provenance as base
        tc._current_provenance = 7
        tc["jets.pt"] = ak.Array([[100], [200], [300]])
        assert tc._column_provenance[Column("jets.pt")] == 7
        assert tc._column_provenance[Column("jets")] == hash((6, 7))

        # 8) Adding a new child to an existing structure
        tc._current_provenance = 8
        tc["jets.new_child"] = ak.Array([[5], [6], [7]])
        assert tc._column_provenance[Column("jets.new_child")] == 8
        assert tc._column_provenance[Column("jets")] == hash((hash((6, 7)), 8))

    def testGetKeyForColumns(self, sample_events):
        from analyzer.core.columns import TrackedColumns, EventBackend, Column
        from analyzer.utils.structure_tools import freeze

        tc = TrackedColumns.fromEvents(
            sample_events, metadata={}, backend=EventBackend.coffea_imm, provenance=0
        )

        key_jets_orig = tc.getKeyForColumns([Column("jets")])
        key_jets_pt_orig = tc.getKeyForColumns([Column("jets.pt")])

        # 1) Updating a child changes the key for the parent AND the child,
        # but NOT for a sibling child.
        tc._current_provenance = 1
        tc["jets.pt"] = ak.Array([[100], [200], [300]])

        key_jets_upd1 = tc.getKeyForColumns([Column("jets")])
        key_jets_pt_upd1 = tc.getKeyForColumns([Column("jets.pt")])
        key_jets_mass_upd1 = tc.getKeyForColumns([Column("jets.mass")])

        assert key_jets_upd1 != key_jets_orig
        assert key_jets_pt_upd1 != key_jets_pt_orig

        # 2) Updating another sibling changes the parent again, but not the previous child
        tc._current_provenance = 2
        tc["jets.mass"] = ak.Array([[10], [20], [30]])

        key_jets_upd2 = tc.getKeyForColumns([Column("jets")])
        key_jets_mass_upd2 = tc.getKeyForColumns([Column("jets.mass")])
        key_jets_pt_upd2 = tc.getKeyForColumns([Column("jets.pt")])

        assert key_jets_upd2 != key_jets_upd1
        assert key_jets_mass_upd2 != key_jets_mass_upd1
        assert key_jets_pt_upd2 == key_jets_pt_upd1  # Unchanged!

        # 3) Updating parent entirely overrides key for parent and all children
        tc._current_provenance = 3
        tc["jets"] = ak.Array(
            {"pt": [[1], [2], [3]], "mass": [[2], [3], [4]], "energy": [[3], [4], [5]]}
        )

        key_jets_upd3 = tc.getKeyForColumns([Column("jets")])
        key_jets_pt_upd3 = tc.getKeyForColumns([Column("jets.pt")])
        key_jets_energy_upd3 = tc.getKeyForColumns([Column("jets.energy")])

        assert key_jets_upd3 != key_jets_upd2
        assert key_jets_pt_upd3 != key_jets_pt_upd2
        assert key_jets_energy_upd3 is not None

        # 4) Deeply nested fields testing (adding a new nested branch)
        tc._current_provenance = 4
        tc["a.b.c"] = ak.Array([1, 2, 3])

        key_a_orig = tc.getKeyForColumns([Column("a")])
        key_ab_orig = tc.getKeyForColumns([Column("a.b")])
        key_abc_orig = tc.getKeyForColumns([Column("a.b.c")])

        tc._current_provenance = 5
        tc["a.b.c"] = ak.Array([4, 5, 6])

        key_a_upd = tc.getKeyForColumns([Column("a")])
        key_ab_upd = tc.getKeyForColumns([Column("a.b")])
        key_abc_upd = tc.getKeyForColumns([Column("a.b.c")])

        assert key_a_upd != key_a_orig
        assert key_ab_upd != key_ab_orig
        assert key_abc_upd != key_abc_orig

        # And ensuring multiple columns queried together reflect composite correct keys
        assert tc.getKeyForColumns(
            [Column("a"), Column("jets")]
        ) != tc.getKeyForColumns([Column("a")])
        assert tc.getKeyForColumns(
            [Column("a"), Column("jets")]
        ) != tc.getKeyForColumns([Column("jets")])


class TestUtils:
    def testSetColumn(self):
        from analyzer.core.columns import setColumn

        events = ak.Array([{"a": {"b": 1}}, {"a": {"b": 2}}])

        events = setColumn(events, "a.b", ak.Array([3, 4]))
        assert ak.all(events.a.b == ak.Array([3, 4]))

        events = setColumn(events, "x.y", ak.Array([5, 6]))
        assert ak.all(events.x.y == ak.Array([5, 6]))

    def testGetAllColumns(self):
        from analyzer.core.columns import getAllColumns

        events = ak.Array([{"a": {"b": 1, "c": 2}, "d": 3}])
        cols = getAllColumns(events.layout)

        assert Column("a.b") in cols
        assert Column("a.c") in cols
        assert Column("d") in cols
        assert Column("a") in cols

    def testMergeColumns(self):
        from analyzer.core.columns import mergeColumns, TrackedColumns, EventBackend

        events = ak.Array({"a": [1], "b": [2]})
        tc1 = TrackedColumns.fromEvents(events, {}, EventBackend.coffea_imm, 0)
        tc2 = TrackedColumns.fromEvents(events, {}, EventBackend.coffea_imm, 0)

        tc2._current_provenance = 1
        tc2["b"] = ak.Array([3])

        merged = mergeColumns([tc1, tc2])
        assert merged._column_provenance[Column("b")] == 1
        assert ak.all(merged["b"] == ak.Array([3]))

    def testAddSelection(self):
        from analyzer.core.columns import addSelection, TrackedColumns, EventBackend

        events = ak.Array({"a": [1]})
        tc = TrackedColumns.fromEvents(events, {}, EventBackend.coffea_imm, 0)

        addSelection(tc, "test_cut", ak.Array([True]))

        assert "Selections" in tc.pipeline_data
        assert "test_cut" in tc.pipeline_data["Selections"]
        assert Column("Selection.test_cut") in tc._column_provenance
