
# optvis

There's a lot of ways one might wish to customize optimization-based feature
visualization. You might wish to change the objective, the way your image is
paramaterized, certain kinds of transformations it is robust to, or more.

As a result, it's hard to create an abstraction that gracefullystands up to all
the directions might want to explore. Typically, we end up just writing new
functions to perform the visualization we want.

This framework tries to provide a very flexible abstraction that allow one to
explore a large space of possibilities. In the case that you want to do
something it doesn't directly support, it should gracefully allow you to
continue using subcomponents.

***Try the [tutorial](https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/tutorial.ipynb) in your browser!***

## Abstraction

* **Objectives** describe what you want to optimize, whether a single neuron,
an interpolation of neurons, deepdream, or style transfer. They also include any
regularizing terms you might wish to add to your objective, such as penalties on
output variation, or terms encouraging diversity.

* **Paramaterizations** describe how you wish to paramaterize an image or  a set
of images. This can have a big impact on the resulting visualization.

* **transformers** describe transformations you wish to be applied to your image
before the network sees it. Often, these are stochastic transformations you wish
for the visualization to be robust to, such as jitter.

* **render_vis()** can finally be used to visualize a combination of these.

## Examples

optvis aims to be very easy to use for simple cases. For example, visualizing a
single neuron takes only one line:

```python
_ = render.render_vis(model, "mixed4a_pre_relu:499")
```

*Note that `render_vis()` creates and then destroys a new session/graph, so that
memory doesn't leak. `model` should be from moralex@'s modelzoo.*

This is equivelant to the slightly longer objective description:

```python
obj = objectives.channel("mixed4a_pre_relu", 2)
render.render_vis(model, obj)
```

Visualizing a pair of neurons:

```python
channel = lambda n: objectives.channel("mixed4a_pre_relu", n)
obj = channel(2) + channel(3)
render.render_vis(model, obj)
```

Using a different paramaterization:

```python
obj = objectives.channel("mixed4a_pre_relu", 2)
param_f = lambda: param.image(128, fft=False, decorrelate=False)
render.render_vis(model, obj, param_f)
```

Using a different optimizer:

```python
obj = objectives.channel("mixed4a_pre_relu", 2)
opt = tf.train.AdamOptimizer()
render.render_vis(model, obj, optimizer=opt)
```

<!--
Rendering 4 different paramaterizations at once:

```python
obj = objectives.channel("mixed4a_pre_relu", 2)
param_f = lambda: tf.concat([
    param.rgb_sigmoid(param.naive([1, 128, 128, 3])),
    param.fancy_colors(param.naive([1, 128, 128, 8])/1.3),
    param.rgb_sigmoid(param.laplacian_pyramid([1, 128, 128, 3])/2.),
    param.fancy_colors(param.laplacian_pyramid([1, 128, 128, 8])/2./1.3),
], 0)
render.render_vis(model, obj, param_f)
```
-->

See the [Lucid tutorial](https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/tutorial.ipynb) for more!


