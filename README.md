

<img src="https://storage.googleapis.com/lucid-link-images/lucid_repo-alpha-warn.png" width="648" alt="Lucid is alpha software. We're still sorting out the kinks in using it outside Google. The API is very much in flux. Unless you want to be an alpha user and help us test Lucid, please come back towards the end of February."></img>


<br>


# Lucid

<!--*DeepDream, but sane. Home of cats, dreams, and interpretable neural networks.*-->

[![PyPI](https://img.shields.io/pypi/status/Lucid.svg)]()
![Build status](https://travis-ci.org/tensorflow/lucid.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/tensorflow/lucid/badge.svg?branch=master)](https://coveralls.io/github/tensorflow/lucid?branch=master)
[![PyPI](https://img.shields.io/pypi/pyversions/Lucid.svg)]()
[![PyPI version](https://badge.fury.io/py/Lucid.svg)](https://badge.fury.io/py/Lucid)

Lucid is a collection of infrastructure and tools for research in neural
network interpretability.

<!--In particular, it provides state of the art implementations of [feature
visualization techniques](https://distill.pub/2017/feature-visualization/),
and flexible abstractions that make it very easy to explore new research
directions.-->





<!--
# Dive In with Colab Notebooks

Start visualizing neural networks ***with no setup***. The following notebooks
run in your browser.
-->


# License and Disclaimer

You may use this software under the Apache 2.0 License. See [LICENSE](LICENSE).

This project is research code. It is not an official Google product.


# Running tests

Use `tox` to run the test suite on all supported environments.

To run tests only for a specific module, pass a folder to `tox`:
`tox tests/misc/io`

To run tests only in a specific environment, pass the environment's identifier
via the `-e` flag: `tox -e py27`.

After adding dependencies to `setup.py`, run tox with the `--recreate` flag to
update the environments' dependencies.

# References

See our article on [feature
visualization techniques](https://distill.pub/2017/feature-visualization/).

<!--
# Notebooks (still under developement)

**Core Notebooks:**

<a href="https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/tutorial.ipynb">
<img src="https://storage.googleapis.com/lucid-static/common/stickers/colab-tutorial.png" width="500" alt=""></img>
</a>
<br>

<br>

**Building Blocks** -- Notebooks corresponding to the [Building Blocks of Interpretability]() article:


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
-->
