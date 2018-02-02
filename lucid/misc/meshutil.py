"""3D mesh manipulation utilities."""

from builtins import str
import numpy as np


def frustum(left, right, bottom, top, znear, zfar):
  """Create view frustum"""
  assert right != left
  assert bottom != top
  assert znear != zfar

  M = np.zeros((4, 4), dtype=np.float32)
  M[0, 0] = +2.0 * znear / (right - left)
  M[2, 0] = (right + left) / (right - left)
  M[1, 1] = +2.0 * znear / (top - bottom)
  M[3, 1] = (top + bottom) / (top - bottom)
  M[2, 2] = -(zfar + znear) / (zfar - znear)
  M[3, 2] = -2.0 * znear * zfar / (zfar - znear)
  M[2, 3] = -1.0
  return M


def perspective(fovy, aspect, znear, zfar):
  """Create perspective projection matrix"""
  assert znear != zfar
  h = np.tan(fovy / 360.0 * np.pi) * znear
  w = h * aspect
  return frustum(-w, w, -h, h, znear, zfar)


def anorm(x, axis=None, keepdims=False):
  """Compute L2 norms alogn specified axes."""
  return np.sqrt((x*x).sum(axis=axis, keepdims=keepdims))


def normalize(v, axis=None, eps=1e-10):
  """L2 Normalize alogn specified axes."""
  return v / max(anorm(v, axis=axis, keepdims=True), eps)


def lookat(eye, target=[0, 0, 0], up=[0, 1, 0]):
  """Generate LookAt modelview matrix."""
  eye = np.float32(eye)
  forward = normalize(target - eye)
  side = normalize(np.cross(forward, up))
  up = np.cross(side, forward)
  M = np.eye(4, dtype=np.float32)
  R = M[:3, :3]
  R[:] = [side, up, -forward]
  M[:3, 3] = -R.dot(eye)
  return M


def homotrans(M, p):
  p = np.asarray(p)
  if p.shape[-1] == M.shape[1]-1:
    p = np.append(p, np.ones_like(p[...,:1]), -1)
  p = np.dot(p, M.T)
  return p[...,:-1] / p[...,-1:]


def _parse_face(line):
  for chunk in line.split():
    vt = [0, 0, 0]
    for i, c in enumerate(chunk.split('/')):
      if c:
        vt[i] = int(c)
    yield tuple(vt)


def load_obj(fn):
  """Load 3d mesh form .obj' file.
  
  Args:
    fn: Input file name or file-like object.
    
  Returns:
    dictionary with following keys:
      position: np.float32, (n, 3) array, vertex positions
      uv: np.float32, (n, 2) array, vertex uv coordinates
      normal: np.float32, (n, 3) array, vertex uv normals
      face: np.int32, (k*3,) traingular face indices
  """
  position = [np.zeros(3, dtype=np.float32)]
  normal = [np.zeros(3, dtype=np.float32)]
  uv = [np.zeros(2, dtype=np.float32)]
  

  tuple2idx = {}
  out_position, out_normal, out_uv, out_face = [], [], [], []
  
  input_file = open(fn) if isinstance(fn, str) else fn
  for line in input_file:
    line = line.strip()
    if not line or line[0] == '#':
      continue
    tag, line = line.split(' ', 1)
    if tag == 'v':
      position.append(np.fromstring(line, sep=' '))
    elif tag == 'vt':
      uv.append(np.fromstring(line, sep=' '))
    elif tag == 'vn':
      normal.append(np.fromstring(line, sep=' '))
    elif tag == 'f':
      face_idx = []
      for vt in _parse_face(line):
        if vt not in tuple2idx:
          # create new output vertex
          pos_idx, uv_idx, normal_idx = vt
          out_position.append(position[pos_idx])
          out_normal.append(normal[normal_idx])
          out_uv.append(uv[uv_idx])
          tuple2idx[vt] = len(out_position)-1
        face_idx.append(tuple2idx[vt])
      # generate face triangles
      for i in range(1, len(face_idx)-1):
        for vi in [0, i, i+1]:
          out_face.append(face_idx[vi])

  return dict(
      position=np.float32(out_position),
      normal=np.float32(out_normal),
      uv=np.float32(out_uv),
      face=np.int32(out_face))