# Lucid

<!--*DeepDream, but sane. Home of cats, dreams, and interpretable neural networks.*-->

[![PyPI project status](https://img.shields.io/pypi/status/Lucid.svg)]()
[![Travis build status](https://img.shields.io/travis/tensorflow/lucid.svg)](https://travis-ci.org/tensorflow/lucid)
[![Code coverage](https://img.shields.io/coveralls/github/tensorflow/lucid.svg)](https://coveralls.io/github/tensorflow/lucid)
[![Supported Python version](https://img.shields.io/pypi/pyversions/Lucid.svg)]()
[![PyPI release version](https://img.shields.io/pypi/v/Lucid.svg)](https://pypi.org/project/Lucid/)


Lucid is a collection of infrastructure and tools for research in neural
network interpretability.

**We're not currently supporting tensorflow 2!**

If you'd like to use lucid in colab which defaults to tensorflow 2, add this magic to a cell before you import tensorflow:

```%tensorflow_version 1.x```

**Lucid is research code, not production code. We provide no guarantee it will work for your use case. Lucid is maintained by volunteers who are unable to provide significant technical support.**

* [📓 **Notebooks**](#notebooks) -- Get started without any setup!
* [📚 **Reading**](#recomended-reading) -- Learn more about visualizing neural nets.
* [💬 **Community**](#community) -- Want to get involved? Please reach out!
* [🔧 **Additional Information**](#additional-information) -- Licensing, code style, etc.
* [🔬 **Start Doing Research!**](https://github.com/tensorflow/lucid/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3Aresearch) -- Want to get involved? We're trying to research openly!
* [📦 **Visualize your own model**](https://github.com/tensorflow/lucid/wiki/Importing-Models-into-Lucid) -- How to import your own model for visualization

<br>

# Notebooks

Start visualizing neural networks ***with no setup***. The following notebooks
run right from your browser, thanks to [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb). It's a Jupyter notebook environment that requires no setup to use and runs entirely in the cloud.

You can run the notebooks on your local machine, too. Clone the repository and find them in the `notebooks` subfolder. You will need to run a local instance of the [Jupyter notebook environment](http://jupyter.org/install.html) to execute them.

## Tutorial Notebooks

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/tutorial.ipynb">
<img src="https://storage.googleapis.com/lucid-static/common/stickers/colab-tutorial.png" width="500" alt=""></img>
</a>


<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/modelzoo.ipynb">
<img src="https://storage.googleapis.com/lucid-static/common/stickers/colab-modelzoo.png" width="500" alt=""></img>
</a>

<!--If you want to study techniques for visualizing and understanding neural networks, it's important to be able to try your experiments on multiple models. As of lucid v0.3, we provide a consistent API for interacting with 27 different vision models.-->

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

<br>

## Activation Atlas Notebooks
*Notebooks corresponding to the [Activation Atlas](https://distill.pub/2019/activation-atlas/) article*

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/activation-atlas/activation-atlas-collect.ipynb">
<img src="https://storage.googleapis.com/modelzoo/tmp/activation-atlas/stickers/lucid-notebook-1-collect.png" width="500" alt="Collecting activations"></img>
</a>

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/activation-atlas/activation-atlas-simple.ipynb">
<img src="https://storage.googleapis.com/modelzoo/tmp/activation-atlas/stickers/lucid-notebook-2-atlas.png" width="500" alt="Simple activation atlas"></img>
</a>

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/activation-atlas/class-activation-atlas.ipynb">
<img src="https://storage.googleapis.com/modelzoo/tmp/activation-atlas/stickers/lucid-notebook-3-class-atlas.png" width="500" alt="Class activation atlas"></img>
</a>

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/activation-atlas/activation-atlas-adversarial.ipynb">
<img src="https://storage.googleapis.com/modelzoo/tmp/activation-atlas/stickers/lucid-notebook-4-patches.png" width="500" alt="Activation atlas patches"></img>
</a>

## Miscellaneous Notebooks

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/misc/feature_inversion_caricatures.ipynb">
<img src="https://storage.googleapis.com/lucid-static/misc/stickers/colab-feature-inversion.ipynb.png" width="500" alt=""></img>
</a>
<br>
<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/misc/neuron_interaction_grids.ipynb">
<img src="https://storage.googleapis.com/lucid-static/misc/stickers/colab-interaction-grid.png" width="500" alt=""></img>
</a>

<br> 

# Recomended Reading

* [Feature Visualization](https://distill.pub/2017/feature-visualization/)
* [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/)
* [Using Artiﬁcial Intelligence to Augment Human Intelligence](https://distill.pub/2017/aia/)
* [Visualizing Representations: Deep Learning and Human Beings](http://colah.github.io/posts/2015-01-Visualizing-Representations/)
* [Differentiable Image Parameterizations](https://distill.pub/2018/differentiable-parameterizations/)
* [Activation Atlas](https://distill.pub/2019/activation-atlas/)

## Related Talks
* [Lessons from a year of Distill ML Research](https://www.youtube.com/watch?v=jlZsgUZaIyY) (Shan Carter, OpenVisConf)
* [Machine Learning for Visualization](https://www.youtube.com/watch?v=6n-kCYn0zxU) (Ian Johnson, OpenVisConf)

# Community

We're in `#proj-lucid` on the Distill slack ([join link](http://slack.distill.pub)).

We'd love to see more people doing research in this space!

<br>

# Additional Information

## License and Disclaimer

You may use this software under the Apache 2.0 License. See [LICENSE](LICENSE).

This project is research code. It is not an official Google product.

## Special consideration for TensorFlow dependency

Lucid requires `tensorflow`, but does not explicitly depend on it in `setup.py`. Due to the way [tensorflow is packaged](https://github.com/tensorflow/tensorflow/issues/7166) and some deficiencies in how pip handles dependencies, specifying either the GPU or the non-GPU version of tensorflow will conflict with the version of tensorflow your already may have installed.

If you don't want to add your own dependency on tensorflow, you can specify which tensorflow version you want lucid to install by selecting from `extras_require` like so: `lucid[tf]` or `lucid[tf_gpu]`.

**In actual practice, we recommend you use your already installed version of tensorflow.**
