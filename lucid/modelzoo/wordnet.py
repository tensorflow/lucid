# Copyright 2019 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helpers for using WordNet Synsets.

When comparing different models, be aware that they may encode their predictions in
different orders. Do not compare outputs of models without ensuring their outputs are
in the same order! We recommend relying on WordNet's synsets to uniquely identify a
label. Let's clarify these terms:

## Labels ("Labrador Retriever")

Label are totally informal and vary between implementations. We aim to provide a list of
model labels in the `.labels` property. These may include differen labels in different
orders for each model.

For translating between textual labels and synsets, plase use the labels and synsets
collections on models. There's no other foolproof way of goinfg from a descriptive text
label to a precise synset definition.


## Synset IDs ("n02099712")

Synset IDs are identifiers used by the ILSVRC2012 ImageNet classification contest.
We provide `id_from_synset()` to format them correctly.


## Synsets Names ('labrador_retriever.n.01')

Synset names are a wordnet internal concept. When youw ant to create a synset but don't
know its precise name, we offer `imagenet_synset_from_description()` to search for a
synset containing the description in its name that is also one of the synsets used for
the ILSVRC2012.


## Label indexes (logits[i])

When obtaining predictions from a model, they will often be provided in the form of a
BATCH by NUM_CLASSES multidimensional array. In order to map those to human readable
strings, please use a model's `.labels` or `.synsets` or `.synset_ids` property. We aim
to provide these in the same ordering as the model was trained on. Unfortunately these
may be subtly different between models.

"""

from cachetools.func import lru_cache

import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

from lucid.misc.io import load


IMAGENET_SYNSETS_PATH = "gs://modelzoo/labels/ImageNet_standard_synsets.txt"


def id_from_synset(synset):
    return "{}{:08}".format(synset.pos(), synset.offset())


def synset_from_id(id_str):
    assert len(id_str) == 1 + 8
    pos, offset = id_str[0], int(id_str[1:])
    return wn.synset_from_pos_and_offset(pos, offset)


@lru_cache(maxsize=1)
def imagenet_synset_ids():
    return load(IMAGENET_SYNSETS_PATH, split=True)


@lru_cache(maxsize=1)
def imagenet_synsets():
    return [synset_from_id(id) for id in imagenet_synset_ids()]


@lru_cache()
def imagenet_synset_from_description(search_term):
    names_and_synsets = [(synset.name(), synset) for synset in imagenet_synsets()]
    candidates = [
        synset for (name, synset) in names_and_synsets if search_term.lower().replace(' ', '_') in name
    ]
    hits = len(candidates)
    if hits == 1:
        return candidates[0]
    if hits == 0:
        message = "Could not find any imagenet synset with search term {}."
        raise ValueError(message.format(search_term))
    else:
      message = "Found {} imagenet synsets with search term {}: {}."
      names = [synset.name() for synset in candidates]
      raise ValueError(message.format(hits, search_term, ", ".join(names)))
