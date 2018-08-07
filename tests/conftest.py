import pytest

from lucid.modelzoo.vision_models import InceptionV1


@pytest.fixture
def inceptionv1():
    model = InceptionV1()
    model.load_graphdef()
    return model


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    else:
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
