import os
import pandas as pd
import numpy as np
from ecological import compute_ecocoherence, validate_ecocoherence


def test_compute_ecocoherence_basic():
    """Test basic eco-coherence computation with a simple cycle."""
    # Create test data
    test_data = pd.DataFrame({
        "source": ["A", "B", "C", "D"],
        "target": ["B", "C", "D", "A"],
        "weight": [1.0, 1.0, 1.0, 1.0]
    })
    test_data.to_csv("data/test_edges.csv", index=False)

    result = compute_ecocoherence("data/test_edges.csv", weight_col="weight")
    os.remove("data/test_edges.csv")

    assert isinstance(result, dict), "Result should be a dictionary"
    assert "C_eco" in result, "Missing C_eco key"
    assert "N" in result, "Missing N key"
    assert "fertility_proxy" in result, "Missing fertility_proxy key"
    assert 0.0 <= result["C_eco"] <= 1.0, "C_eco out of expected range for uniform cycle"
    assert result["N"] == 4, "Incorrect number of nodes"
    assert result["fertility_proxy"] >= 0, "Fertility proxy should be non-negative"


def test_compute_ecocoherence_missing_weights():
    """Test handling of missing weight column."""
    test_data = pd.DataFrame({
        "source": ["A", "B"],
        "target": ["B", "A"]
    })
    test_data.to_csv("data/test_edges_no_weight.csv", index=False)

    result = compute_ecocoherence("data/test_edges_no_weight.csv")
    os.remove("data/test_edges_no_weight.csv")

    assert result["C_eco"] > 0, "C_eco should be computable with default weights"
    assert result["N"] == 2, "Incorrect number of nodes"


def test_compute_ecocoherence_disconnected():
    """Test warning for disconnected graph."""
    test_data = pd.DataFrame({
        "source": ["A", "C"],
        "target": ["B", "D"],
        "weight": [1.0, 1.0]
    })
    test_data.to_csv("data/test_edges_disconnected.csv", index=False)

    with np.testing.assert_logs(level="WARNING") as cm:
        result = compute_ecocoherence("data/test_edges_disconnected.csv")
    os.remove("data/test_edges_disconnected.csv")

    assert "Warning: Graph is not weakly connected" in cm.output[0]


def test_compute_ecocoherence_invalid_input():
    """Test error handling for invalid input."""
    with np.testing.assert_raises(ValueError):
        compute_ecocoherence("data/nonexistent.csv")
    with np.testing.assert_raises(ValueError):
        compute_ecocoherence("data/invalid_edges.csv")  # Missing source/target


def test_validate_ecocoherence():
    """Test validation function against SFH expected range."""
    valid_result = {"C_eco": 1.4, "N": 10, "fertility_proxy": 1.0}
    invalid_result = {"C_eco": 3.0, "N": 10, "fertility_proxy": 1.0}

    assert validate_ecocoherence(valid_result) is True, "Valid C_eco should pass"
    assert validate_ecocoherence(invalid_result) is False, "Invalid C_eco should fail"
    with np.testing.assert_logs(level="WARNING") as cm:
        validate_ecocoherence(invalid_result)
    assert "Warning: C_eco (3.0) outside expected SFH range (0.5, 2.0)" in cm.output[0]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
