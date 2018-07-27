import pytest

from lucid.modelzoo.vision_models import InceptionV1


@pytest.fixture
def inceptionv1():
    model = InceptionV1()
    model.load_graphdef()
    return model
