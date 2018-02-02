"""show() smoke tests
show relies heavily on a notebook environment, so we can only ahve some smoke
tests in the test suite. There is also a notebook with tests that you can run
to cover common scenarios.
"""
from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
import lucid.util.show as show



def test_show_image():
  show.image(np.eye(5))
  assert show._last_data_output.startswith(b'\x89PNG')


def test_show_images():
  labels = ["one", "two", "three"]
  show.images([np.eye(5)] * 3, labels=labels)
  golden_output = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAAAA"
  "ACoBHk5AAAAEklEQVR4nGP4z8DAwMDAgJ0CAErTBPw/r52mAAAAAElFTkSuQmCC")
  # test image is in there
  assert golden_output in show._last_html_output
  # test image labels are all there
  assert all("{}<br/>".format(i) in show._last_html_output for i in labels)
