from __future__ import absolute_import, division, print_function

import pytest

import os
import numpy as np

HAVE_COLAB_NVIDIA = (os.path.exists('/usr/lib64-nvidia/') and
                     os.path.exists('/opt/bin/nvidia-smi'))


WIDTH, HEIGHT = 200, 100

if HAVE_COLAB_NVIDIA:
  from lucid.misc.gl import glcontext  # must be imported before OpenGL.GL
  import OpenGL.GL as gl
  from lucid.misc.gl import glrenderer
  
  glcontext.create_opengl_context((WIDTH, HEIGHT))


@pytest.mark.skipif(not HAVE_COLAB_NVIDIA, reason="GPU Colab kernel only")
def test_gl_context():
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
  img_buf = gl.glReadPixels(0, 0, WIDTH, HEIGHT, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)    
  img = np.frombuffer(img_buf, np.uint8).reshape(HEIGHT, WIDTH, 3)[::-1]

  assert all(img[0, 0] == 0)  # black corner
  assert all(img[0,-1] == 0)  # black corner
  assert img[10, WIDTH//2].argmax() == 0  # red corner
  assert img[-1,  10].argmax() == 1  # green corner    
  assert img[-1, -10].argmax() == 2  # blue corner
  

@pytest.mark.skipif(not HAVE_COLAB_NVIDIA, reason="GPU Colab kernel only")
def test_glrenderer():
  w, h = 400, 200
  renderer = glrenderer.MeshRenderer((w, h))
  renderer.fovy = 90
  position = [[0, 1, -1], [-2, -1,-1], [2, -1, -1]]
  color = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  img = renderer.render_mesh(position, color)
  img, alpha = img[..., :3], img[..., 3]
  
  assert all(img[0, 0] == 0)  # black corner
  assert all(img[0,-1] == 0)  # black corner
  assert img[10, w//2].argmax() == 0  # red corner
  assert img[-1,  10].argmax() == 1  # green corner    
  assert img[-1, -10].argmax() == 2  # blue corner
  assert np.abs(img.sum(-1)-alpha).max() < 1e-5