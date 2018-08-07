<img src="https://storage.googleapis.com/lucid-static/common/stickers/channels-visualizations.jpg" width="782"></img>

# Lucid

<!--*DeepDream, but sane. Home of cats, dreams, and interpretable neural networks.*-->

[![PyPI project status](https://img.shields.io/pypi/status/Lucid.svg)]()
[![Travis build status](https://img.shields.io/travis/tensorflow/lucid.svg)](https://travis-ci.org/tensorflow/lucid)
[![Code coverage](https://img.shields.io/coveralls/github/tensorflow/lucid.svg)](https://coveralls.io/github/tensorflow/lucid)
[![Supported Python version](https://img.shields.io/pypi/pyversions/Lucid.svg)]()
[![PyPI release version](https://img.shields.io/pypi/v/Lucid.svg)](https://pypi.org/project/Lucid/)


Lucid is a collection of infrastructure and tools for research in neural
network interpretability.

In particular, it provides state of the art implementations of [feature
visualization techniques](https://distill.pub/2017/feature-visualization/),
and flexible abstractions that make it very easy to explore new research
directions.


# Special consideration for TensorFlow dependency

Lucid requires `tensorflow`, but does not explicitly depend on it in `setup.py`. Due to the way [tensorflow is packaged](https://github.com/tensorflow/tensorflow/issues/7166) and some deficiencies in how pip handles dependencies, specifying either the GPU or the non-GPU version of tensorflow will conflict with the version of tensorflow your already may have installed.

If you don't want to add your own dependency on tensorflow, you can specify which tensorflow version you want lucid to install by selecting from `extras_require` like so: `lucid[tf]` or `lucid[tf_gpu]`.

**In actual practice, we recommend you use your already installed version of tensorflow.**

# Notebooks

Start visualizing neural networks ***with no setup***. The following notebooks
run right from your browser, thanks to [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb). It's a Jupyter notebook environment that requires no setup to use and runs entirely in the cloud.

You can run the notebooks on your local machine, too. Clone the repository and find them in the `notebooks` subfolder. You will need to run a local instance of the [Jupyter notebook environment](http://jupyter.org/install.html) to execute them.

## Tutorial Notebooks

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/tutorial.ipynb">
<img src="https://storage.googleapis.com/lucid-static/common/stickers/colab-tutorial.png" width="500" alt=""></img>
</a>


## Feature Visualization Notebooks
*Notebooks corresponding to the [Feature Visualization](https://distill.pub/2017/feature-visualization/) article*

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/feature-visualization/negative_neurons.ipynb">
<img src="https://storage.googleapis.com/lucid-static/feature-visualization/stickers/colab-neuron-negative.png" width="500" alt=""></img>
</a>

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/feature-visualization/neuron_diversity.ipynb">
<img src="https://storage.googleapis.com/lucid-static/feature-visualization/stickers/colab-neuron-diversity.png" width="500" alt=""></img>
</a>

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/feature-visualization/neuron_interaction.ipynb">
<img src="https://storage.googleapis.com/lucid-static/feature-visualization/stickers/colab-neuron-interaction.png" width="500" alt=""></img>
</a>

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/feature-visualization/regularization.ipynb">
<img src="https://storage.googleapis.com/lucid-static/feature-visualization/stickers/colab-regularization.png" width="500" alt=""></img>
</a>

## Building Blocks Notebooks
*Notebooks corresponding to the [Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/) article*


<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/SemanticDictionary.ipynb">
<img src="https://storage.googleapis.com/lucid-static/building-blocks/stickers/colab-semantic-dict.png" width="500" alt=""></img>
</a>
<br>
<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/ActivationGrid.ipynb">
<img src="https://storage.googleapis.com/lucid-static/building-blocks/stickers/colab-grid.png" width="500" alt=""></img>
</a>
<br>
<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/AttrSpatial.ipynb">
<img src="https://storage.googleapis.com/lucid-static/building-blocks/stickers/colab-spatial-attr.png" width="500" alt=""></img>
</a>
<br>
<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/AttrChannel.ipynb">
<img src="https://storage.googleapis.com/lucid-static/building-blocks/stickers/colab-channel-attr.png" width="500" alt=""></img>
</a>
<br>
<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/NeuronGroups.ipynb">
<img src="https://storage.googleapis.com/lucid-static/building-blocks/stickers/colab-neuron-groups.png" width="500" alt=""></img>
</a>


## Differentiable Image Parameterizations Notebooks
*Notebooks corresponding to the [Differentiable Image Parameterizations](https://distill.pub/2018/differentiable-parameterizations/) article*

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/aligned_interpolation.ipynb">
<img src="https://storage.googleapis.com/lucid-static/differentiable-parameterizations/stickers/colab-interpolate.png" width="500" alt=""></img>
</a>

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/style_transfer_2d.ipynb">
<img src="https://storage.googleapis.com/lucid-static/differentiable-parameterizations/stickers/colab-style-beyond-vgg.png" width="500" alt=""></img>
</a>

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/xy2rgb.ipynb">
<img src="https://storage.googleapis.com/lucid-static/differentiable-parameterizations/stickers/colab-xy2rgb.png" width="500" alt=""></img>
</a>

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/transparency.ipynb">
<img src="https://storage.googleapis.com/lucid-static/differentiable-parameterizations/stickers/colab-transparent.png" width="500" alt=""></img>
</a>

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/texture_synth_3d.ipynb">
<img src="https://storage.googleapis.com/lucid-static/differentiable-parameterizations/stickers/colab-3d-texture.png" width="500" alt=""></img>
</a>

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/style_transfer_3d.ipynb">
<img src="https://storage.googleapis.com/lucid-static/differentiable-parameterizations/stickers/colab-3d-style.png" width="500" alt=""></img>
</a>


## Miscellaneous Notebooks

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/misc/feature_inversion_caricatures.ipynb">
<img src="https://storage.googleapis.com/lucid-static/misc/stickers/colab-feature-inversion.ipynb.png" width="500" alt=""></img>
</a>
<br>
<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/misc/neuron_interaction_grids.ipynb">
<img src="https://storage.googleapis.com/lucid-static/misc/stickers/colab-interaction-grid.png" width="500" alt=""></img>
</a>

# Recomended Reading

* [Feature Visualization](https://distill.pub/2017/feature-visualization/)
* [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/)
* [Using Artiﬁcial Intelligence to Augment Human Intelligence](https://distill.pub/2017/aia/)
* [Visualizing Representations: Deep Learning and Human Beings](http://colah.github.io/posts/2015-01-Visualizing-Representations/)
* [Differentiable Image Parameterizations](https://distill.pub/2018/differentiable-parameterizations/)

## Related Talks
* [Lessons from a year of Distill ML Research](https://www.youtube.com/watch?v=jlZsgUZaIyY) (Shan Carter, OpenVisConf)
* [Machine Learning for Visualization](https://www.youtube.com/watch?v=6n-kCYn0zxU) (Ian Johnson, OpenVisConf)

<br>
<br>

# Additional Information

## License and Disclaimer

You may use this software under the Apache 2.0 License. See [LICENSE](LICENSE).

This project is research code. It is not an official Google product.


## Development

### Style guide deviations

We use naming conventions to help differentiate tensors, operations, and values:

* Suffix variable names representing **tensors** with `_t`
* Suffix variable names representing **operations** with `_op`
* Don't suffix variable names representing concrete values

Usage example:

```
global_step_t = tf.train.get_or_create_global_step()
global_step_init_op = tf.variables_initializer([global_step_t])
global_step = global_step_t.eval()
```

### Running Tests

Use `tox` to run the test suite in both Python 2 and Python 3 environments.

To also run slower integration tests (marked with `pytest.mark.slow`), specify the `--run-slow` option for pytest, which can be passed through `tox` like so:

```
tox -- --run-slow
```

To run tests only for a specific module, pass a folder to `tox`:
`tox tests/misc/io`

To run tests only in a specific environment, pass the environment's identifier
via the `-e` flag: `tox -e py27`.

After adding dependencies to `setup.py`, run tox with the `--recreate` flag to
update the environments' dependencies.
