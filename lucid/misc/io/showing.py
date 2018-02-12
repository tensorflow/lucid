# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Methods for displaying images from Numpy arrays."""

from __future__ import absolute_import, division, print_function

from io import BytesIO
import base64
import logging
import numpy as np
import IPython.display
from string import Template

from lucid.misc.io.serialize_array import serialize_array, array_to_jsbuffer


# create logger with module name, e.g. lucid.misc.io.showing
log = logging.getLogger(__name__)


def _display_html(html_str):
  IPython.display.display(IPython.display.HTML(html_str))


def _image_url(array, fmt='png', mode="data", quality=70, domain=None):
  """Create a data URL representing an image from a PIL.Image.

  Args:
    image: a numpy
    mode: presently only supports "data" for data URL

  Returns:
    URL representing image
  """
  # TODO: think about supporting saving to CNS, potential params: cns, name
  # TODO: think about saving to Cloud Storage
  supported_modes = ("data")
  if mode not in supported_modes:
    message = "Unsupported mode '%s', should be one of '%s'."
    raise ValueError(message, mode, supported_modes)

  image_data = serialize_array(array, fmt=fmt, quality=quality)
  base64_byte_string = base64.b64encode(image_data).decode('ascii')
  return "data:image/" + fmt.upper() + ";base64," + base64_byte_string


# public functions


def image(array, domain=None, w=None, format='png'):
  """Display an image.

  Args:
    array: NumPy array representing the image
    fmt: Image format e.g. png, jpeg
    domain: Domain of pixel values, inferred from min & max values if None
    w: width of output image, scaled using nearest neighbor interpolation.
      size unchanged if None
  """
  data_url = _image_url(array, domain=domain)
  html = '<img src=\"' + data_url + '\">'
  _display_html(html)


def images(arrays, labels=None, domain=None, w=None):
  """Display a list of images with optional labels.

  Args:
    arrays: A list of NumPy arrays representing images
    labels: A list of strings to label each image.
      Defaults to show index if None
    domain: Domain of pixel values, inferred from min & max values if None
    w: width of output image, scaled using nearest neighbor interpolation.
      size unchanged if None
  """

  s = '<div style="display: flex; flex-direction: row;">'
  for i, array in enumerate(arrays):
    url = _image_url(array)
    label = labels[i] if labels is not None else i
    s += """<div style="margin-right:10px;">
              {label}<br/>
              <img src="{url}" style="margin-top:4px;">
            </div>""".format(label=label, url=url)
  s += "</div>"
  _display_html(s)


def show(thing, domain=(0, 1)):
  """Display a nupmy array without having to specify what it represents.

  This module will attempt to infer how to display your tensor based on its
  rank, shape and dtype. rank 4 tensors will be displayed as image grids, rank
  2 and 3 tensors as images.
  """
  if isinstance(thing, np.ndarray):
    rank = len(thing.shape)
    if rank == 4:
      log.debug("Show is assuming rank 4 tensor to be a list of images.")
      images(thing, domain=domain)
    elif rank in (2, 3):
      log.debug("Show is assuming rank 2 or 3 tensor to be an image.")
      image(thing, domain=domain)
    else:
      log.warn("Show only supports numpy arrays of rank 2-4. Using repr().")
      print(repr(thing))
  elif isinstance(thing, (list, tuple)):
    log.debug("Show is assuming list or tuple to be a collection of images.")
    images(thing, domain=domain)
  else:
    log.warn("Show only supports numpy arrays so far. Using repr().")
    print(repr(thing))


def textured_mesh(mesh, texture, background='0xffffff'):
  texture_data_url = _image_url(texture, fmt='jpeg', quality=90)

  code = Template('''
  <script src="https://cdn.rawgit.com/mrdoob/three.js/r89/build/three.min.js"></script>
  <script src="https://cdn.rawgit.com/mrdoob/three.js/r89/examples/js/controls/OrbitControls.js"></script>

  <script>
  "use strict";

  var camera, scene, renderer, controls;

  init();
  animate();

  function init() {
    var width = 800, height = 600;

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.z = 1.5;
    scene.add(camera);

    controls = new THREE.OrbitControls( camera );

    var geometry = new THREE.BufferGeometry();
    geometry.addAttribute( 'position', new THREE.BufferAttribute($verts, 3 ) );
    geometry.addAttribute( 'uv', new THREE.BufferAttribute($uvs, 2) );
    geometry.setIndex(new THREE.BufferAttribute($faces, 1 ));
    geometry.computeVertexNormals();

    var loader = new THREE.TextureLoader();
    var texture = loader.load( '$tex_data_url' );

    var material = new THREE.MeshBasicMaterial({
        'map': texture,
        'color': 0xffffff,
        'side': THREE.DoubleSide,
    });


    var mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    scene.background = new THREE.Color( $background );

    renderer = new THREE.WebGLRenderer();
    renderer.setSize(width, height);

    document.body.appendChild(renderer.domElement);
  }

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  </script>
  ''').substitute(
      verts = array_to_jsbuffer(mesh['position'].ravel()),
      uvs = array_to_jsbuffer(mesh['uv'].ravel()),
      faces = array_to_jsbuffer(np.uint32(mesh['face'].ravel())),
      tex_data_url = texture_data_url,
      background = background,
  )
  _display_html(code)
  
def textured_mesh_adv(mesh, texture, background='0xffffff', unfolding = 0.0, shading = True, animate = False, unfolding_speed = 2.):
  texture_data_url = _image_url(texture, fmt='jpeg', quality=90)

  code = Template('''
  <script src="https://cdn.rawgit.com/mrdoob/three.js/r89/build/three.min.js"></script>
  <script src="https://cdn.rawgit.com/mrdoob/three.js/r89/examples/js/controls/OrbitControls.js"></script>

  <script type="x-shader/x-vertex" id="vertexShader"> 
    uniform float unfolding_perc;
    varying vec2 text_coord;
    varying float shading;
    void main () {
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      vec4 plane_position = vec4((uv.x*2.0-1.0),(uv.y*2.0-1.0),0,1);
      gl_Position = gl_Position * vec4(1.0-unfolding_perc) + plane_position * vec4(unfolding_perc);
      
      //not normalized on purpose to simulate the rotation
      vec3 light_vector = normalize(cameraPosition-position)*vec3(1.0-unfolding_perc) + normal*vec3(unfolding_perc);
      shading = dot(normal,light_vector);
      
      text_coord = uv;
      //shading = 0.5;
    }
  </script>

  <script type="x-shader/x-fragment" id="fragmentShader">  
    uniform float unfolding_perc;
    uniform float texture_perc;
    varying vec2  text_coord;
    varying float shading;
    uniform sampler2D texture1;
    
    void main() {  
      gl_FragColor = texture2D(texture1, text_coord) * texture_perc + vec4(text_coord,0,1) * (1.0-texture_perc);
      if($shading){
        gl_FragColor *= vec4(vec3(shading),1); 
      }
    }
  </script>  

  <script>
  "use strict";

  var camera, scene, renderer, controls, mesh;
  var inc_texture = 0.0015;
  var inc_unfolding = $unfolding_speed / 1000.0;
  var unfolding_perc = -Math.PI/2;
  init();
  animate();

  function init() {
    var width = 800, height = 600;

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.z = 1.5;
    scene.add(camera);

    controls = new THREE.OrbitControls( camera );

    var geometry = new THREE.BufferGeometry();
    geometry.addAttribute( 'position', new THREE.BufferAttribute($verts, 3 ) );
    geometry.addAttribute( 'uv', new THREE.BufferAttribute($uvs, 2) );
    geometry.setIndex(new THREE.BufferAttribute($faces, 1 ));
    geometry.computeVertexNormals();

    var material_unf = new THREE.ShaderMaterial( {
      uniforms: {
      
        unfolding_perc: { value: 0.8 },
        texture_perc:   { value: 1.0 },
        texture1: { type: 't', value: THREE.ImageUtils.loadTexture( '$tex_data_url' ) }
      },
      side: THREE.DoubleSide,
      vertexShader: document.getElementById( 'vertexShader' ).textContent,
      fragmentShader: document.getElementById( 'fragmentShader' ).textContent
    });
    

    mesh = new THREE.Mesh(geometry, material_unf);
    scene.add(mesh);
    scene.background = new THREE.Color( $background );

    mesh.material.uniforms.unfolding_perc.value = $unfolding;
    mesh.material.uniforms.texture_perc.value = 1.0;

    renderer = new THREE.WebGLRenderer();
    renderer.setSize(width, height);

    document.body.appendChild(renderer.domElement);
  }


  function animate() {

    if($animate){
      //mesh.material.uniforms.unfolding_perc.value += inc_unfolding;
      unfolding_perc += inc_unfolding;
      mesh.material.uniforms.unfolding_perc.value = (Math.sin(unfolding_perc)+1)/2;
      if(mesh.material.uniforms.unfolding_perc.value > 1 || mesh.material.uniforms.unfolding_perc.value < 0){
        inc_unfolding = -inc_unfolding;
      }
    }
    
    /*    
    mesh.material.uniforms.texture_perc.value += inc_texture;
    if(mesh.material.uniforms.texture_perc.value > 1 || mesh.material.uniforms.texture_perc.value < 0){
      inc_texture = -inc_texture;
    }
    */   
    
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
    
  }
  </script>
  ''').substitute(
      verts = array_to_jsbuffer(mesh['position'].ravel()),
      uvs = array_to_jsbuffer(mesh['uv'].ravel()),
      faces = array_to_jsbuffer(np.uint32(mesh['face'].ravel())),
      tex_data_url = texture_data_url,
      background = background,
      unfolding = unfolding,
      shading = 'true' if shading else 'false',
      animate = 'true' if animate else 'false',
      unfolding_speed = unfolding_speed
  )
  #print(code)
  _display_html(code)
