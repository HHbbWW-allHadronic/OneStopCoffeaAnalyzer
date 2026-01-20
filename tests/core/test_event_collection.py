from analyzer.core.event_collection import (
    chunkN,
    FileInfo,
    FileSet,
    FileChunk,
)
import pytest


def testChunkN():
    chunks = chunkN(100, 10)
    assert len(chunks) == 10
    assert chunks == {(i * 10, (i + 1) * 10) for i in range(10)}

    c2 = chunkN(105, 10)
    assert len(c2) == 10

    total = sum(y - x for x, y in c2)
    assert total == 105


def testFileInfoChunking():
    fi = FileInfo("f1", nevents=100)
    assert not fi.is_chunked

    fi.iChunk(20)
    assert fi.is_chunked
    assert len(fi.chunks) == 5
    assert fi.target_chunk_size == 20

    fi2 = FileInfo("f2", nevents=50)
    fi3 = fi2.chunked(10)
    assert not fi2.is_chunked
    assert fi3.is_chunked
    assert len(fi3.chunks) == 5


def testFileInfoOps():
    f1 = FileInfo("f1", nevents=100, chunks={(0, 10)})
    f2 = FileInfo("f1", nevents=100, chunks={(10, 20)})

    f3 = f1 + f2
    assert f3.chunks == {(0, 10), (10, 20)}

    f4 = f3 - f1
    assert f4.chunks == {(10, 20)}

    f1 += f2
    assert f1.chunks == {(0, 10), (10, 20)}
    f1 -= f2
    assert f1.chunks == {(0, 10)}


def testFileInfoIntersection():
    f1 = FileInfo("f1", nevents=100, chunks={(0, 10), (10, 20)})
    f2 = FileInfo("f1", nevents=100, chunks={(10, 20), (20, 30)})

    assert f1.intersects(f2)
    f_int = f1.intersection(f2)
    assert f_int.chunks == {(10, 20)}

    f3 = FileInfo("f1", nevents=100, chunks={(50, 60)})
    assert not f1.intersects(f3)


def testFileSetOps():
    f1 = FileInfo("f1", nevents=100, chunks={(0, 10)})
    f2 = FileInfo("f2", nevents=50, chunks={(0, 50)})

    fs1 = FileSet(files={"f1": f1})
    fs2 = FileSet(files={"f2": f2})

    fs3 = fs1 + fs2
    assert len(fs3.files) == 2
    assert "f1" in fs3.files
    assert "f2" in fs3.files

    f1_other = FileInfo("f1", nevents=100, chunks={(10, 20)})
    fs4 = FileSet(files={"f1": f1_other})

    fs5 = fs1 + fs4
    assert len(fs5.files) == 1
    assert fs5.files["f1"].chunks == {(0, 10), (10, 20)}

    fs6 = fs5 - fs1
    assert fs6.files["f1"].chunks == {(10, 20)}


def testFileSetSplit():
    files = {f"f{i}": FileInfo(f"f{i}", nevents=10) for i in range(5)}
    fs = FileSet(files=files)

    splits = fs.splitFiles(2)
    assert len(splits) == 3

    all_files = set()
    for sub_fs in splits.values():
        all_files.update(sub_fs.files.keys())
    assert len(all_files) == 5
    assert all_files == set(files.keys())


def testFileChunkOverlaps():
    fc1 = FileChunk("f1", 0, 10)
    fc2 = FileChunk("f1", 5, 15)
    assert fc1.overlaps(fc2)
    assert fc2.overlaps(fc1)

    fc3 = FileChunk("f1", 10, 20)
    assert fc1.overlaps(fc3)

    fc4 = FileChunk("f1", 20, 30)
    assert not fc1.overlaps(fc4)

    fc5 = FileChunk("f2", 0, 10)
    assert not fc1.overlaps(fc5)
