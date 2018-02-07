"""OpenGL Mesh rendering utils."""

from contextlib import contextmanager
import numpy as np

import OpenGL.GL as gl

from .meshutil import perspective


class GLObject(object):
  def __del__(self):
    self.release()
  def __enter__(self):
    bind_func, const = self._bind
    bind_func(const, self)
  def __exit__(self, *args):
    bind_func, const = self._bind
    bind_func(const, 0)
    
class FBO(GLObject):
  _bind = gl.glBindFramebuffer, gl.GL_FRAMEBUFFER
  def __init__(self):
    self._as_parameter_ = gl.glGenFramebuffers(1)
  def release(self):
    gl.glDeleteFramebuffers(1, [self._as_parameter_])
    

class Texture(GLObject):
  _bind = gl.glBindTexture, gl.GL_TEXTURE_2D
  def __init__(self):
    self._as_parameter_ = gl.glGenTextures(1)
  def release(self):
    gl.glDeleteTextures([self._as_parameter_])
    

class Shader(GLObject):

  def __init__(self, vp_code, fp_code):
    # Importing here, when gl context is already present.
    # Otherwise get expection on Python3 because of PyOpenGL bug.
    from OpenGL.GL import shaders
    self._as_parameter_ = self._shader = shaders.compileProgram(
        shaders.compileShader( vp_code, gl.GL_VERTEX_SHADER ),
        shaders.compileShader( fp_code, gl.GL_FRAGMENT_SHADER )
    )
    self._uniforms = {}
    
  def release(self):
    gl.glDeleteProgram(self._as_parameter_)
  
  def __getitem__(self, uniform_name):
    if uniform_name not in self._uniforms:
      self._uniforms[uniform_name] = gl.glGetUniformLocation(self, uniform_name)
    return self._uniforms[uniform_name]

  def __enter__(self):
    return self._shader.__enter__()
  def __exit__(self, *args):
    return self._shader.__exit__(*args)
      
    
class MeshRenderer(object):
  def __init__(self, size):
    self.size = size
    self.fbo = FBO()
    self.color_tex = Texture()
    self.depth_tex = Texture()
    w, h = size
    
    with self.color_tex:
      gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0,
                      gl.GL_RGBA, gl.GL_FLOAT, None)
      
    with self.depth_tex:
      gl.glTexImage2D.wrappedOperation(
          gl.GL_TEXTURE_2D, 0,gl.GL_DEPTH24_STENCIL8, w, h, 0,
          gl.GL_DEPTH_STENCIL, gl.GL_UNSIGNED_INT_24_8, None)
    
    with self.fbo:
      gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                             gl.GL_TEXTURE_2D, self.color_tex, 0)
      gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT,
                             gl.GL_TEXTURE_2D, self.depth_tex, 0)
      gl.glViewport(0, 0, w, h)
      assert gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
        
    self.shader = Shader(vp_code='''
      #version 130
      uniform mat4 MVP;
      in vec4 data;
      out vec4 aData;

      void main() {
        aData = data;
        gl_Position = MVP * gl_Vertex;
      }
    ''', 
    fp_code='''
      #version 130
      in vec4 aData;
      out vec4 fragColor;
      void main() {
        fragColor = aData;
      }
    ''')
    
    self.fovy = 10.0
    self.aspect = 1.0*w/h
    self.znear, self.zfar = 0.01, 100.0
    
  @contextmanager
  def _bind_attrib(self, i, arr):
    if arr is None:
      yield
      return
    arr = np.ascontiguousarray(arr, np.float32)
    coord_n = arr.shape[-1]
    gl.glEnableVertexAttribArray(i)
    gl.glVertexAttribPointer(i, coord_n, gl.GL_FLOAT, gl.GL_FALSE, 0, arr)
    yield
    gl.glDisableVertexAttribArray(i)
    
  def proj_matrix(self):
    return perspective(self.fovy, self.aspect, self.znear, self.zfar)
    
  def render_mesh(self, position, uv, face=None,
                  clear_color=[0, 0, 0, 0],
                  modelview=np.eye(4)):
    MVP = modelview.T.dot(self.proj_matrix())
    MVP = np.ascontiguousarray(MVP, np.float32)
    position = np.ascontiguousarray(position, np.float32)
    with self.fbo:
      gl.glClearColor(*clear_color)
      gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
      
      with self.shader, self._bind_attrib(0, position), self._bind_attrib(1, uv):
        gl.glUniformMatrix4fv(self.shader['MVP'], 1, gl.GL_FALSE, MVP)
        gl.glEnable(gl.GL_DEPTH_TEST)
        if face is not None:
          face = np.ascontiguousarray(face, np.uint32)
          gl.glDrawElements(gl.GL_TRIANGLES, face.size, gl.GL_UNSIGNED_INT, face)
        else:
          vert_n = position.size//position.shape[-1]
          gl.glDrawArrays(gl.GL_TRIANGLES, 0, vert_n)
        gl.glDisable(gl.GL_DEPTH_TEST)
      
      w, h = self.size
      frame = gl.glReadPixels(0, 0, w, h, gl.GL_RGBA, gl.GL_FLOAT)
      frame = frame.reshape(h, w, 4)  # fix PyOpenGL bug
      frame = frame[::-1]  # verical flip to match GL convention
      return frame