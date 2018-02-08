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

"""Helper for using sklearn.decomposition on high-dimensional tensors.

Provides ChannelReducer, a wrapper around sklearn.decomposition to help them
apply to arbitrary rank tensors. It saves lots of annoying reshaping.
"""

import numpy as np
import sklearn.decomposition

class ChannelReducer(object):
  """Helper for dimensionality reduction to the innermost dimension of a tensor.

  This class wraps sklearn.decomposition classes to help them apply to arbitrary
  rank tensors. It saves lots of annoying reshaping.

  See the original sklearn.decomposition documentation:
  http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
  """

  def __init__(self, n_features=3, reduction_alg="NMF", **kwargs):
    """Constructor for ChannelReducer.

    Inputs:
      n_features: Numer of dimensions to reduce inner most dimension to.
      reduction_alg: A string or sklearn.decomposition class. Defaults to
        "NMF" (non-negative matrix facotrization). Other options include:
        "PCA", "FastICA", and "MiniBatchDictionaryLearning". The name of any of
        the sklearn.decomposition classes will work, though.
      kwargs: Additional kwargs to be passed on to the reducer.
    """
    if isinstance(reduction_alg, str):
      reduction_alg = sklearn.decomposition.__getattribute__(reduction_alg)
    self.n_features = n_features
    self._reducer = reduction_alg(n_features, **kwargs)

  @classmethod
  def _apply_flat(cls, f, acts):
    """Utility for applying f to inner dimension of acts.

    Flattens acts into a 2D tensor, applies f, then unflattens so that all
    dimesnions except innermost are unchanged.
    """
    orig_shape = acts.shape
    acts_flat = acts.reshape([-1, acts.shape[-1]])
    new_flat = f(acts_flat)
    if not isinstance(new_flat, np.ndarray):
      return new_flat
    shape = list(orig_shape[:-1]) + [-1]
    return new_flat.reshape(shape)

  def fit(self, acts):
    return ChannelReducer._apply_flat(self._reducer.fit, acts)

  def fit_transform(self, acts):
    return ChannelReducer._apply_flat(self._reducer.fit_transform, acts)

  def transform(self, acts):
    return ChannelReducer._apply_flat(self._reducer.transform, acts)

  #TODO: decide whether to expose _reducer params higher up, figure
  # out any transforms that need to be done.

  #def __getattr__(self, name):
  #  return getattr(self._reducer, name)

  def __dir__(self):
    dynamic_attrs = dir(self._reducer)
    return self.__dict__.keys()# + dynamic_attrs
