import uuid
import hashlib
from pathlib import Path

import awkward as ak
from attrs import define, field

from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from analyzer.utils.file_tools import copyFile
from analyzer.utils.structure_tools import dictToDot, dotFormat


@define
class SaveEventsH5(AnalyzerModule):
    """
    Analyzer module that serializes event-level data to HDF5 files for NN training.

    - Jet-level variables are written as (n_events, n_jets) datasets named
      "{jet_col}_{var}" (with '.' replaced by '_').
    - Event-level variables are written as (n_events,) datasets named after the
      Column (with '.' replaced by '_').

    Assumes it is run in a region where every event has at least `n_jets` jets in
    `jet_col`. Raises otherwise.

    Parameters
    ----------
    prefix : str
        Destination directory prefix where the output HDF5 files will be copied.
        (You said this will be the region name/path.)
    jet_col : Column
        Jet collection column (e.g. Column("goodJet")).
    jet_vars : list[str]
        Fields on the jet collection (e.g. ["pt","eta","phi","mass"]).
    n_jets : int
        Number of leading jets to write per event.
    event_vars : list[Column], optional
        Event-level columns to write (e.g. [Column("HT"), Column("b_dijet_12.mass")]).
    output_format : str, optional
        Filename template expanded with metadata fields plus file_id and uuid.
        Note: uuid is only used for the *local* temp filename unless you include
        "{uuid}" in this format.
    """

    prefix: str
    jet_col: Column
    jet_vars: list[str]
    n_jets: int
    event_vars: list[Column] = field(factory=list)
    output_format: str = (
        "{dataset_name}__{sample_name}__{file_id}"
        "__{chunk.event_start}_{chunk.event_stop}.h5"
    )

    def run(self, columns, params):
        file_id = (
            hashlib.md5((columns.metadata["chunk"]["file_path"]).encode())
            .hexdigest()
            .upper()
        )
        uid = str(uuid.uuid4())

        target_name = dotFormat(
            self.output_format,
            **dict(dictToDot(columns.metadata)),
            file_id=file_id,
            uuid=uid,
        )
        target = f"{self.prefix}/{target_name}"

        base = Path("localsaved")
        base.mkdir(exist_ok=True, parents=True)
        local_filename = base / f"{uid}.h5"

        try:
            import h5py

            jets = columns[self.jet_col]

            # Validate jet multiplicity before writing anything
            min_jets = ak.min(ak.num(jets))
            if min_jets is None:
                return columns, []

            min_jets = int(min_jets)
            if min_jets < self.n_jets:
                raise ValueError(
                    f"SaveH5: collection '{self.jet_col}' requires at least {self.n_jets} jets/event, "
                    f"but found an event with {min_jets} in chunk "
                    f"{columns.metadata['chunk']['file_path']} "
                    f"[{columns.metadata['chunk']['event_start']}, "
                    f"{columns.metadata['chunk']['event_stop']}]"
                )

            jet_prefix = str(self.jet_col).replace(".", "_")

            with h5py.File(local_filename, "w") as f:
                # Per-jet datasets: (n_events, n_jets)
                for v in self.jet_vars:
                    field_name = v if (v!="btag") else columns.metadata["era"]["btag_scale_factors"]["tagger"]
                    arr = jets[field_name][:, : self.n_jets]
                    data = ak.to_numpy(arr)
                    f.create_dataset(
                        f"{jet_prefix}_{v}",
                        data=data,
                        compression="gzip",
                    )

                # Event-level datasets: (n_events,)
                for c in self.event_vars:
                    arr = columns[c]
                    data = ak.to_numpy(arr)
                    name = str(c).replace(".", "_")
                    f.create_dataset(
                        name,
                        data=data,
                        compression="gzip",
                    )

            copyFile(local_filename, target)
        finally:
            local_filename.unlink(missing_ok=True)

        return columns, []

    def inputs(self, metadata):
        return [self.jet_col] + list(self.event_vars)

    def outputs(self, metadata):
        return []
