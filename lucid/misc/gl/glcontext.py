"""Headless GPU-accelerated OpenGL context creation on Google Colaboratory.

Typical usage:

    # Optional PyOpenGL configuratiopn can be done here.
    # import OpenGL
    # OpenGL.ERROR_CHECKING = True

    # 'glcontext' must be imported before any OpenGL.* API.
    from lucid.misc.gl.glcontext import create_opengl_context

    # Now it's safe to import OpenGL and EGL functions
    import OpenGL.GL as gl

    # create_opengl_context() creates a GL context that is attached to an
    # offscreen surface of the specified size. Note that rendering to buffers
    # of other sizes and formats is still possible with OpenGL Framebuffers.
    #
    # Users are expected to directly use the EGL API in case more advanced
    # context management is required.
    width, height = 640, 480
    create_opengl_context((width, height))

    # OpenGL context is available here.

"""

from __future__ import print_function

# pylint: disable=unused-import,g-import-not-at-top,g-statement-before-imports

try:
  import OpenGL
except:
  print('This module depends on PyOpenGL.')
  print('Please run "\033[1m!pip install -q pyopengl\033[0m" '
        'prior importing this module.')
  raise

import ctypes
from ctypes import pointer
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# OpenGL loading workaround.
#
# * PyOpenGL tries to load libGL, but we need libOpenGL, see [1,2].
#   This could have been solved by a symlink libGL->libOpenGL, but:
#
# * Python 2.7 can't find libGL and linEGL due to a bug (see [3])
#   in ctypes.util, that was only wixed in Python 3.6.
#
# So, the only solution I've found is to monkeypatch ctypes.util
# [1] https://devblogs.nvidia.com/egl-eye-opengl-visualization-without-x-server/
# [2] https://devblogs.nvidia.com/linking-opengl-server-side-rendering/
# [3] https://bugs.python.org/issue9998
_find_library_old = ctypes.util.find_library
try:

  def _find_library_new(name):
    return {
        'GL': 'libOpenGL.so',
        'EGL': 'libEGL.so',
    }.get(name, _find_library_old(name))
  ctypes.util.find_library = _find_library_new
  import OpenGL.GL as gl
  import OpenGL.EGL as egl
except:
  print('Unable to load OpenGL libraries. '
        'Make sure you use GPU-enabled backend.')
  print('Press "Runtime->Change runtime type" and set '
        '"Hardware accelerator" to GPU.')
  raise
finally:
  ctypes.util.find_library = _find_library_old


def create_opengl_context(surface_size=(640, 480)):
  """Create offscreen OpenGL context and make it current.

  Users are expected to directly use EGL API in case more advanced
  context management is required.

  Args:
    surface_size: (width, height), size of the offscreen rendering surface.
  """
  egl_display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)

  major, minor = egl.EGLint(), egl.EGLint()
  egl.eglInitialize(egl_display, pointer(major), pointer(minor))

  config_attribs = [
      egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT, egl.EGL_BLUE_SIZE, 8,
      egl.EGL_GREEN_SIZE, 8, egl.EGL_RED_SIZE, 8, egl.EGL_DEPTH_SIZE, 24,
      egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT, egl.EGL_NONE
  ]
  config_attribs = (egl.EGLint * len(config_attribs))(*config_attribs)

  num_configs = egl.EGLint()
  egl_cfg = egl.EGLConfig()
  egl.eglChooseConfig(egl_display, config_attribs, pointer(egl_cfg), 1,
                      pointer(num_configs))

  width, height = surface_size
  pbuffer_attribs = [
      egl.EGL_WIDTH,
      width,
      egl.EGL_HEIGHT,
      height,
      egl.EGL_NONE,
  ]
  pbuffer_attribs = (egl.EGLint * len(pbuffer_attribs))(*pbuffer_attribs)
  egl_surf = egl.eglCreatePbufferSurface(egl_display, egl_cfg, pbuffer_attribs)

  egl.eglBindAPI(egl.EGL_OPENGL_API)

  egl_context = egl.eglCreateContext(egl_display, egl_cfg, egl.EGL_NO_CONTEXT,
                                     None)
  egl.eglMakeCurrent(egl_display, egl_surf, egl_surf, egl_context)