import pytest

from lucid.modelzoo.aligned_activations import NUMBER_OF_AVAILABLE_SAMPLES
from lucid.modelzoo.vision_models import AlexNet, InceptionV1
from lucid.recipes.activation_atlas import activation_atlas, aligned_activation_atlas
from lucid.misc.io import save

# Run test with just 1/10th of available samples
subset = NUMBER_OF_AVAILABLE_SAMPLES // 10


@pytest.mark.skip(reason="takes too long to complete on CI")
def test_activation_atlas():
    model = AlexNet()
    layer = model.layers[1]
    atlas = activation_atlas(model, layer, number_activations=subset)
    save(atlas, "tests/recipes/results/activation_atlas/atlas.jpg")


@pytest.mark.skip(reason="takes too long to complete on CI")
def test_aligned_activation_atlas():
    model1 = AlexNet()
    layer1 = model1.layers[1]

    model2 = InceptionV1()
    layer2 = model2.layers[8]  # mixed4d

    atlasses = aligned_activation_atlas(
        model1, layer1, model2, layer2, number_activations=subset
    )
    path = "tests/recipes/results/activation_atlas/aligned_atlas-{}-of-{}.jpg".format(index, len(atlasses))
    for index, atlas in enumerate(atlasses):
        save(atlas, path)
