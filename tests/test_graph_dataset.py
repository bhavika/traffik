import traffik.build_graph_dataset as gd
import pytest
import os


@pytest.fixture
def mock_env_variables(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/tmp")


def test_setup_paths(mock_env_variables):
    actual = gd.setup("berlin")

    assert actual == {
        "output_path": "/tmp/processed",
        "node": "/tmp/processed/berlin/berlin_nodes_5.npy",
        "edge": "/tmp/processed/berlin/berlin_edges_5.npy",
    }
    assert os.path.exists(actual["output_path"]) is True
