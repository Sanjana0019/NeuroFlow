import pytest

from evaluation.calibrate import compute_pearson_correlation, run_calibration


def test_pearson_correlation_math():
    """Verifies Pearson correlation r calculation correctness."""
    # Perfect positive correlation -> 1.0
    assert compute_pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0
    # Perfect negative correlation -> -1.0
    assert compute_pearson_correlation([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == -1.0
    # Zero variation / invalid -> 0.0
    assert compute_pearson_correlation([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) == 0.0


@pytest.mark.asyncio
async def test_calibration_benchmark_threshold():
    """Runs 30-example calibration benchmark and confirms Pearson correlation > 0.85."""
    results = await run_calibration("evaluation/calibration/annotated_set.json")
    assert results["sample_size"] == 30
    assert results["threshold_met"] is True
    assert results["measured_pearson_correlation"] > 0.85
