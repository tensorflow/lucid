"""3D mesh manipulation utilities."""

from builtins import str
from collections import OrderedDict
import numpy as np


def frustum(left, right, bottom, top, znear, zfar):
  """Create view frustum matrix."""
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
  """Create perspective projection matrix."""
  assert znear != zfar
  h = np.tan(fovy / 360.0 * np.pi) * znear
  w = h * aspect
  return frustum(-w, w, -h, h, znear, zfar)


def anorm(x, axis=None, keepdims=False):
  """Compute L2 norms alogn specified axes."""
  return np.sqrt((x*x).sum(axis=axis, keepdims=keepdims))


def normalize(v, axis=None, eps=1e-10):
  """L2 Normalize along specified axes."""
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


def _parse_vertex_tuple(s):
  """Parse vertex indices in '/' separated form (like 'i/j/k', 'i//k' ...)."""
  vt = [0, 0, 0]
  for i, c in enumerate(s.split('/')):
    if c:
      vt[i] = int(c)
  return tuple(vt)


def _unify_rows(a):
  """Unify lengths of each row of a."""
  lens = np.fromiter(map(len, a), np.int32)
  if not (lens[0] == lens).all():
    out = np.zeros((len(a), lens.max()), np.float32)
    for i, row in enumerate(a):
      out[i, :lens[i]] = row
  else:
    out = np.float32(a)
  return out


def load_obj(fn):
  """Load 3d mesh form .obj' file.
  
  Args:
    fn: Input file name or file-like object.
    
  Returns:
    dictionary with the following keys (some of which may be missing):
      position: np.float32, (n, 3) array, vertex positions
      uv: np.float32, (n, 2) array, vertex uv coordinates
      normal: np.float32, (n, 3) array, vertex uv normals
      face: np.int32, (k*3,) traingular face indices
  """
  position = [np.zeros(3, dtype=np.float32)]
  normal = [np.zeros(3, dtype=np.float32)]
  uv = [np.zeros(2, dtype=np.float32)]
  
  tuple2idx = OrderedDict()
  trinagle_indices = []
  
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
      output_face_indices = []
      for chunk in line.split():
        # tuple order: pos_idx, uv_idx, normal_idx
        vt = _parse_vertex_tuple(chunk)
        if vt not in tuple2idx:  # create a new output vertex?
          tuple2idx[vt] = len(tuple2idx)
        output_face_indices.append(tuple2idx[vt])
      # generate face triangles
      for i in range(1, len(output_face_indices)-1):
        for vi in [0, i, i+1]:
          trinagle_indices.append(output_face_indices[vi])
  
  outputs = {}
  outputs['face'] = np.int32(trinagle_indices)
  pos_idx, uv_idx, normal_idx = np.int32(list(tuple2idx)).T
  if np.any(pos_idx):
    outputs['position'] = _unify_rows(position)[pos_idx]
  if np.any(uv_idx):
    outputs['uv'] = _unify_rows(uv)[uv_idx]
  if np.any(normal_idx):
    outputs['normal'] = _unify_rows(normal)[normal_idx]
  return outputs
