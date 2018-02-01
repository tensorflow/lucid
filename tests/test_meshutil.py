
import pytest

import io
import numpy as np
from lucid.misc import meshutil

test_obj = u'''
# comment
#
v 0 0 0
v 0 1 0
v 1 1 0
v 1 0 0

vt 0 0
vt 0 1
vt 1 1
vt 1 0

vn 1 0 1
vn 0 1 1
vn 1 1 1
vn 1 0 1

f 1 2 3
f 2/2/2 3/3/3 4/4/4
f 1//1 2//2 3//3 4//4
'''


def test_load_obj():
  f = io.StringIO(test_obj)
  mesh = meshutil.load_obj(f)
  vert_n = len(mesh['position'])
  assert vert_n == 10
  assert mesh['position'].shape == (vert_n, 3)
  assert mesh['normal'].shape == (vert_n, 3)
  assert mesh['uv'].shape == (vert_n, 2)
  assert mesh['face'].shape == ((1+1+2)*3,)