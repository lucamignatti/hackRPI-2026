import importlib.util

import pytest

HAS_GENESIS = importlib.util.find_spec("genesis") is not None


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="genesis-world not installed (the torch backend still runs)")
    for item in items:
        if "genesis" in item.keywords and not HAS_GENESIS:
            item.add_marker(skip)
