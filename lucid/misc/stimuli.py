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

"""Helpers for generating synthetic stimuli for probing network behavior."""

import numpy as np


def sample_binary_image(size, alias_factor=10, color_a=(1,1,1), color_b=(0,0,0),
                        boundary_line=False, boundary_width=1,
                        blur_beyond_radius=None, fade_beyond_radius=None,
                        fade_over_distance=10, fade_color=(.5, .5, .5), **kwds):
  """Highly flexible tool for sampling binary images.

  Many stimuli of interest are "binary" in that they have two regions. For
  example, a curve stimuli has an interio and exterior region. Ideally, such a
  stimulus should be rendered with antialiasing. Additionlly, there are many
  styling options that effect how one might wish to render the image: selecting
  the color for interior and exterior, showing the boundary between the regions
  instead of interior vs exterior, and much more.

  This function provides a flexible rendering tool that supports many options.
  We assume the image is reprented in the "f-rep" or implicit funciton
  convention: the image is represented by a function which maps (x,y) values
  to a sclar, with negative representing the object interior and positive
  representing the exterior.

  The general usage would look smething like:

      @sample_binary_image(size, more_options)
      def img(x,y):
        return (negative if interior, positive if exterior)

  Or alternatively:

      sampler = sample_binary_image(size, more_options)
      def img_f(x,y):
        return (negative if interior, positive if exterior)
      img = sampler(img_f)

  Arguments:
    size: Size of image to be rendered in pixels.
    alias_factor: Number of samples to use in aliasing.
    color_a: Color of exterior. A 3-tuple of floats between 0 and 1. Defaults
      to white (1,1,1).
    color_b: Color of interior or boundary. A 3-tuple of floats between 0 and 1.
      Defaults to black (0,0,0).
    boundary_line: Draw boundary instead of interior vs exterior.
    boundary_width: If drawing boundary, number of pixels wide boundary line
      should be. Defaults to 1 pixel.
    blur_beyond_radius: If not None, blur the image outside a given radius.
      Defaults to None.
    fade_beyond_radius: If not None, fade the image to fade_color outside a
      given radius. Defaults to None.
    fade_over_distance: Controls rate of fading.
    fade_color: Color to fade to, if fade_beyond_radius is set. Defaults to
      (.5, .5, .5).

  Returns:
    A function which takes a function mapping (x,y) -> float and returns a
    numpy array of shape [size, size, 3].
  """


  # Initial setup
  color_a, color_b = np.asarray(color_a).reshape([1,1,3]), np.asarray(color_b).reshape([1,1,3])
  fade_color = np.asarray(fade_color).reshape([1,1,3])
  X = (np.arange(size) - size//2)
  X, Y = X[None, :], X[:, None]
  alias_offsets = [ tuple(np.random.uniform(-.5, .5, size=2)) for n in range(alias_factor) ]
  boundary_offsets = [ (boundary_width*np.cos(2*np.pi*n/16.), boundary_width*np.sin(2*np.pi*n/16.)) for n in range(16) ]

  # Setup for blur / fade stuff
  radius = np.sqrt(X**2+Y**2)
  offset_scale = 1
  fade_coef = 0
  if blur_beyond_radius is not None:
    offset_scale += np.maximum(0, radius-blur_beyond_radius)
  if fade_beyond_radius is not None:
    fade_coef = np.maximum(0, radius-fade_beyond_radius)
    fade_coef /= float(fade_over_distance)
    fade_coef = np.clip(fade_coef, 0, 1)[..., None]

  # The function we'll return.
  # E is an "energy function" mapping (x,y) -> float
  # (and vectorized for numpy support)
  # such that it is negative on interior reigions and positive on exterior ones.
  def sampler(E):

    # Naively smaple an image
    def sample(x_off, y_off):
      # note: offset_scale controls blurring
      vals = E(X + offset_scale * x_off, Y + offset_scale * y_off)
      return np.greater_equal(vals, 0).astype("float32")

    def boundary_sample(x_off, y_off):
      imgs = [sample(x_off + bd_off_x, y_off + bd_off_y)
              for bd_off_x, bd_off_y in boundary_offsets]
      # If we are on the boundary, some smaples will be zero and others one.
      # as a result, the mean will be in the middle.
      vals = np.mean(imgs, axis=0)
      vals = 2*np.abs(vals-0.5)
      return np.greater_equal(vals, 0.99).astype("float32")

    # Sample anti-aliased image
    sampler = boundary_sample if boundary_line else sample
    img = np.mean([sampler(*offset) for offset in alias_offsets], axis=0)
    img = np.clip(img, 0, 1)[..., None]

    # final transformations to colorize and fade
    img = img*color_a + (1-img)*color_b
    img = (1-fade_coef)*img + fade_coef*fade_color
    return img
  return sampler


def rmin(X, Y, r):
  """A  "rounded minimum" function.

  Useful for creating bevelled intersections with implicit geometry image
  representations.

  See Olah (2011), "Manipulation of Implicit Functions (With an Eye on CAD)".
  https://christopherolah.wordpress.com/2011/11/06/manipulation-of-implicit-functions-with-an-eye-on-cad/
  """
  r_ = np.maximum(r, 1e-6)
  v1 = Y + r*np.sin(np.pi/4 + np.arcsin(np.clip((X-Y)/r_/np.sqrt(2), -1, 1))) - r
  v2 = np.minimum(X,Y)
  cond = np.less_equal(np.abs(X-Y), r)
  return np.where(cond, v1, v2)


def rounded_corner(orientation, r, angular_width=90, size=224, **kwds):
  """Generate a "rounded corner" stimulus.

  This function is a flexible generator of "rounded corner" stimuli. It returns
  an image, represented as a numpy array of shape [size, size, 3].

  Arguments:
    orientation: The orientation of the curve, in degrees.
    r: radius of the curve
    angular_width: when r=0 and we have sharp corner, this controls the angle
      of the corner. For other radius values, it controls the sharpness of the
      corner and eventual divergence of the outer lines. Specified in degrees,
      defaults to 90 to give a traditional curve.
    size: Size of image.

    Note: Also inherits many arguments from sample_binary_image().

  Returns:
    An image, represented as a numpy array of shape [size, size, 3].
  """
  orientation *= 2 * np.pi / 360.
  orientation += np.pi/2.
  angular_width *= 2 * np.pi / 360.
  ang1, ang2 = orientation - angular_width / 2., orientation + angular_width / 2. + np.pi
  @sample_binary_image(size, **kwds)
  def img(X, Y):
    X_, Y_ = np.cos(ang1)*X + np.sin(ang1)*Y,  np.cos(ang2)*X + np.sin(ang2)*Y
    X_, Y_ = X_ + r*(1-1/np.sqrt(2.)), Y_ + r*(1-1/np.sqrt(2.))
    E = rmin(X_,Y_,r)
    return 10*E
  return img
