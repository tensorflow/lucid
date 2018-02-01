from __future__ import absolute_import, division, print_function

import pytest

import os
import numpy as np

HAVE_COLAB_NVIDIA = (os.path.exists('/usr/lib64-nvidia/') and
                     os.path.exists('/opt/bin/nvidia-smi'))


@pytest.mark.skipif(not HAVE_COLAB_NVIDIA, reason="GPU Colab kernel only")
def test_gl_context():
  from lucid.misc import colab_gl
  import OpenGL.GL as gl

  w, h = 200, 100
  colab_gl.create_opengl_context((w, h))

  # Render triangle
  gl.glClear(gl.GL_COLOR_BUFFER_BIT)
  gl.glBegin(gl.GL_TRIANGLES)
  gl.glColor3f(1, 0, 0)
  gl.glVertex2f(0,  1)
  gl.glColor3f(0, 1, 0)
  gl.glVertex2f(-1, -1)
  gl.glColor3f(0, 0, 1)
  gl.glVertex2f(1, -1)
  gl.glEnd()

  # Read result
  img_buf = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)    
  img = np.frombuffer(img_buf, np.uint8).reshape(h, w, 3)[::-1]

  assert all(img[0, 0] == 0)  # black corner
  assert all(img[0,-1] == 0)  # black corner
  assert img[10,w//2].argmax() == 0  # red corner
  assert img[-1,  10].argmax() == 1  # green corner    
  assert img[-1, -10].argmax() == 2  # blue corner