# Copyright 2018 The Lucid Authors. All Rights Reserved.
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

"""Utilities for breaking lists into chunks."""


def batch(list, batch_size=None, num_batches=None):
    assert not (batch_size is not None and num_batches)
    if batch_size is not None:
        return _batches_of_max_size(list, batch_size)
    elif num_batches is not None:
        return _n_batches(list, num_batches)
    else:
        raise RuntimeError("Either batch size or num_batches needs to be specified!")


def _batches_of_max_size(list, size):
    for i in range(0, len(list), size):
        yield list[i : i + size]


def _n_batches(list, size):
    raise NotImplementedError
