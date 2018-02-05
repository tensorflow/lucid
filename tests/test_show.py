"""show() smoke tests
show relies heavily on a notebook environment, so we can only ahve some smoke
tests in the test suite. There is also a notebook with tests that you can run
to cover common scenarios.
"""
from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
import lucid
import lucid.util.show as show
import IPython.display


golden_eye_html = ('<img src="data:image/PNG;base64,iVBORw0KGgoAAAANSUhEUgAAAAU'
  'AAAAFCAAAAACoBHk5AAAAEklEQVR4nGP4z8DAwMDAgJ0CAErTBPw/r52mAAAAAElFTkSuQmCC">')


def test_show_image(mocker):
  mock_display = mocker.patch('IPython.display.display')
  mock_html = mocker.patch('IPython.display.HTML')

  show.image(np.eye(5))

  mock_html.assert_called_once_with(golden_eye_html)
  mock_display.assert_called_once()


def test_show_images(mocker):
  mock_html = mocker.patch('IPython.display.HTML')
  labels = ["one", "two", "three"]

  show.images([np.eye(5)] * 3, labels=labels)

  mock_html.assert_called_once()
  args, _ = mock_html.call_args_list[0]
  html_arg = args[0]
  # check for img tag without closing bracket
  assert golden_eye_html[:-1] in html_arg
  # check that label strings are in output
  assert all(label in html_arg for label in labels)
