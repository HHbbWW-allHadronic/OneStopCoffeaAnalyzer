from analyzer.datasets import FileSet, DatasetRepo

from rich import print
from analyzer.core.results import SampleResult
from analyzer.core.executor import AnalysisTask
from analyzer.core.analyzer import Analyzer


try:
    pass

    LPCQUEUE_AVAILABLE = True
except ImportError as e:
    LPCQUEUE_AVAILABLE = False


def getMissingFileset(target: FileSet, prepped: FileSet, processed: FileSet):
    failed_to_process = prepped - processed
    failed_to_prep = target.withoutFiles(prepped.justChunked())
    return (failed_to_process + failed_to_prep)

def getSamplePatch(sample_result: SampleResult, dataset_repo):

    sample = dataset_repo[sample_result.sample_id]
    ran = sample_result.file_set_ran
    proc = sample_result.file_set_processed
    target = sample.getFileSet(proc.file_retrieval_kwargs)
    patch_set = getMissingFileset(target, ran, proc)

    ret = AnalysisTask(
        sample_id=sample_result.sample_id,
        sample_params=sample_result.params,
        file_set=patch_set,
        analyzer=Analyzer(
            region_analyzers=[x.region for x in sample_result.results.values()]
        ),
    )
    return ret
