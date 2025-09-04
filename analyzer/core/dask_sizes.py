from analyzer.core.results import BaseResult, SubSectorResult, MultiSectorResult
from analyzer.core.histograms import HistogramCollection
import hist
from dask.sizeof import sizeof
from analyzer.datasets.fileset import FileSet
from analyzer.datasets.files import SampleFile


@sizeof.register(hist.hist.Hist)
def approxSizeofHist(obj):
    return sizeof(obj.view(flow=True))


@sizeof.register(HistogramCollection)
def approxSizeofHistogramCollection(obj):
    return sizeof(obj.histogram)


@sizeof.register(BaseResult)
def approxSizeofBaseResult(obj):
    return sum(sizeof(x) for x in obj.histograms.values())


@sizeof.register(SubSectorResult)
def approxSizeofSubsectorResult(obj):
    return 1000 + sizeof(obj.base_result)


@sizeof.register(MultiSectorResult)
def approxSizeofMultiSectorResult(obj):
    return sum(sizeof(x) for x in obj.root.values())


@sizeof.register(FileSet)
def approxSizeofFileset(obj):
    return sum(
        (sizeof(k) + sizeof(x) + sizeof(y) for k, (x, y) in obj.files.items())
    ) + sizeof(obj.form)


@sizeof.register(SampleFile)
def approxSizeofSampleFile(obj):
    return sizeof(obj.paths)
