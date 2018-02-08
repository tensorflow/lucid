🔶 **Lucid is alpha software.**
We're still sorting out the kinks in using it outside Google.
The API is very much in flux.
Unless you want to be an alpha user and help us test Lucid,
please come back towards the end of February.

<br>


# Lucid

<!--*DeepDream, but sane. Home of cats, dreams, and interpretable neural networks.*-->

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
