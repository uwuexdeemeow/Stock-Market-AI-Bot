"""Legacy label test split out of tests/test_corrected_audit.py (labels.py is in legacy/)."""

import numpy as np
import pandas as pd
import pytest

from labels import make_spy_forward_return, make_direction_target


def test_missing_benchmark_rejected_but_raw_labels_work():
    frame = pd.DataFrame({"Close": np.arange(100., 110.)})
    with pytest.raises(ValueError, match="benchmark"):
        make_spy_forward_return(frame)
    with pytest.raises(ValueError, match="Benchmark"):
        make_direction_target(frame.Close, prediction_target="excess_return")
    assert len(make_direction_target(frame, prediction_target="raw_return")) == len(frame)
