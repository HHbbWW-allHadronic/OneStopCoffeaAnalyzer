import pytest
import hist
import numpy as np
from analyzer.postprocessing.transforms.hist_transforms import (
    NormalizeSystematicByProjection,
)
from analyzer.core.results import Histogram
from analyzer.utils.structure_tools import ItemWithMeta


def test_normalize_systematics_by_projection():
    h = hist.Hist(
        hist.axis.Regular(5, 0, 10, name="A"),
        hist.axis.Regular(4, 0, 10, name="B"),
        hist.axis.StrCategory(["up", "down", "nom"], name="variation"),
        storage=hist.storage.Weight(),  # use Weight storage so we have variances
    )

    rng = np.random.default_rng(42)

    h.fill(
        A=rng.uniform(0, 10, 1000),
        B=rng.uniform(0, 10, 1000),
        variation="nom",
        weight=rng.uniform(0.5, 1.5, 1000),
    )

    h.fill(
        A=rng.uniform(0, 10, 500),
        B=rng.uniform(0, 10, 500),
        variation="up",
        weight=rng.uniform(0.5, 1.5, 500),
    )

    # We create the item to process
    class MockPh:
        def __init__(self, name, histogram, axes):
            self.name = name
            self.histogram = histogram
            self.axes = axes

    item = MockPh("test_hist", h, h.axes)

    # Run the transform
    transform = NormalizeSystematicByProjection(
        normalize_within=["A"], pre_sf_name="nom", variation_axis="variation"
    )

    out_items = transform([(item, {})])
    hout = out_items[0][0].histogram
    for idx_A in range(-1, len(h.axes["A"]) + 1):
        if idx_A == -1:
            loc_A = hist.underflow
        elif idx_A == len(h.axes["A"]):
            loc_A = hist.overflow
        else:
            loc_A = idx_A

        sum_nom = hout[{"A": loc_A, "variation": "nom"}].sum().value
        sum_up = hout[{"A": loc_A, "variation": "up"}].sum().value
        orig_sum_nom = h[{"A": loc_A, "variation": "nom"}].sum().value
        orig_sum_up = h[{"A": loc_A, "variation": "up"}].sum().value

        if orig_sum_up > 0 and orig_sum_nom > 0:
            np.testing.assert_allclose(sum_nom, sum_up, rtol=1e-5)
        elif orig_sum_nom == 0 or orig_sum_up == 0:
            if orig_sum_up == 0:
                assert sum_up == 0

    # Also verify that down and nom are unchanged
    np.testing.assert_array_equal(
        h[{"variation": "nom"}].view(flow=True),
        hout[{"variation": "nom"}].view(flow=True),
    )
    np.testing.assert_array_equal(
        h[{"variation": "down"}].view(flow=True),
        hout[{"variation": "down"}].view(flow=True),
    )
