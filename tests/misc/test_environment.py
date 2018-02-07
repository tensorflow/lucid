from __future__ import absolute_import, division, print_function

import pytest

from lucid.misc.environment import is_notebook_environment

def test_is_notebook_environment():
  is_notebook = is_notebook_environment()
  assert not is_notebook  # tests aren't usually run in notebook env
