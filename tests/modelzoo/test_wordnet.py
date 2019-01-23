import pytest

import nltk

nltk.download("wordnet")
from nltk.corpus import wordnet as wn

from lucid.modelzoo.wordnet import (
    id_from_synset,
    synset_from_id,
    imagenet_synset_ids,
    imagenet_synsets,
    imagenet_synset_from_description,
)


@pytest.fixture()
def synset():
    return wn.synset("great_white_shark.n.01")


@pytest.fixture()
def synset_id():
    return "n01484850"


def test_id_from_synset(synset, synset_id):
    result = id_from_synset(synset)
    assert result == synset_id


def test_synset_from_id(synset_id, synset):
    result = synset_from_id(synset_id)
    assert result == synset


def test_imagenet_synset_ids(synset_id):
    synset_ids = imagenet_synset_ids()
    assert len(synset_ids) == 1000
    assert synset_id in synset_ids


def test_imagenet_synsets(synset):
    synsets = imagenet_synsets()
    assert len(synsets) == 1000
    assert synset in synsets


def test_imagenet_synset_from_description(synset):
    synset_from_description = imagenet_synset_from_description("white shark")
    assert synset == synset_from_description


def test_imagenet_synset_from_description_raises(synset):
    with pytest.raises(ValueError, match=r'.*great_white_shark.*tiger_shark.*'):
        imagenet_synset_from_description("shark")
