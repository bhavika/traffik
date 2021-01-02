import traffik.dataset as gd
import pytest
import os
import numpy as np
import numpy.testing
import torch


@pytest.fixture
def mock_env_variables(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/tmp")


def test_setup_paths(mock_env_variables):
    actual = gd.setup("berlin")

    assert actual == {
        "output_path": "/tmp/processed",
        "node": "/tmp/intermediate/berlin_nodes_5.npy",
        "edge": "/tmp/intermediate/berlin_edges_5.npy",
    }
    assert os.path.exists(actual["output_path"]) is True


def test_combine_grids():
    train_grid = np.diag(np.diag(np.arange(9).reshape((3, 3))))
    validation_grid = np.diag(np.diag(np.arange(start=10, stop=19).reshape((3, 3))))
    test_grid = np.zeros((3, 3))

    grid = gd.combine_grids(
        "unity-city",
        train_grid,
        validation_grid,
        test_grid,
        data_type="max_vol",
        save=False,
    )
    numpy.testing.assert_equal(grid, [[10, 0, 0], [0, 14, 0], [0, 0, 18]])


def test_combine_grids_different_shapes():
    train_grid = np.array([1, 0, 0])
    validation_grid = np.array([0])
    test_grid = np.array([1])

    grid = gd.combine_grids(
        "unity-city",
        train_grid,
        validation_grid,
        test_grid,
        data_type="max_vol",
        save=False,
    )
    numpy.testing.assert_equal(grid, [1, 1, 1])
