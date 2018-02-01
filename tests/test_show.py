
from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
from lucid.misc import show


def test_show_image():
  show.image(np.eye(10))
  prefix = '<img src="data:image/png;base64,'
  golden_output = ('<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoA'
      'AAAKCAAAAACoWZBhAAAAD0lEQVR4nGP4z4AANGcDACRmCfdF8JCiAAAAAElFTkSuQmCC">')
  assert show._last_html_output == golden_output
 
  