# How to Contribute

We'd love to accept your patches and contributions to this project! There are
just a few small guidelines you need to follow.

### Pull Requests

Most submissions, including submissions by project collaborators, require review. Files in `scratch/` and `notebooks/` are excempt. We use GitHub pull requests for this purpose. Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests. Before sending your pull requests, please make sure you've completed this list.

- You've read this document, the [Contributing Guidelines](CONTRIBUTING.md).
- Your changes are consistent with the [Coding Style](https://github.com/tensorflow/lucid/blob/master/CONTRIBUTING.md#coding-style).
- You've run all [Unit Tests](https://github.com/tensorflow/lucid/blob/master/CONTRIBUTING.md#unit-tests) on all supported versions of Python.
- You've added at least integration-level unit tests for your code. A reasonable indicator is that your PR doesn't substantially reduce [test coverage](https://coveralls.io/github/tensorflow/lucid).
- If you've added new files, you've [included a License](https://github.com/tensorflow/lucid/blob/master/CONTRIBUTING.md#unit-tests) at the top of those files.
- You've signed Google's [Contributor License Agreement (CLA)](https://cla.developers.google.com/). No worries about this—you do [not surrender ownership of your contribution, and you do not give up any of your rights to use your contribution elsewhere](https://cla.developers.google.com/about).

### Coding Style

We aim to conform to
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).

Additionally, we use naming conventions to help differentiate tensors, operations, and values:

* Suffix variable names representing **tensors** with `_t`
* Suffix variable names representing **operations** with `_op`
* Don't suffix variable names representing concrete values

Usage example:

```
global_step_t = tf.train.get_or_create_global_step()
global_step_init_op = tf.variables_initializer([global_step_t])
global_step = global_step_t.eval()
```

Other than that we currently have no automated and enforced coding style. We also follow no coding style for non-python code in this repository at the moment.

### Unit Tests

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

If you prefer to run tests directly with pytest, ensure you manually install the test dependencies. Within the lucid repository, run:

```
pip install .[test]
```

#### During Development

If you'd like to develop using [TDD](https://en.wikipedia.org/wiki/Test-driven_development), we recommend  calling the tests you're currently working on [using `pytest` directly](https://docs.pytest.org/en/latest/usage.html), e.g. `python -m pytest tests/path/to/your/test.py`. Please don't forget to run all tests using `tox` before submitting a PR, though!


### License

Include a license at the top of new files.

* [Python license example](https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/nets_factory.py#L1)
* [HTML license example](https://github.com/tensorflow/lucid/blob/master/lucid/scratch/js/src/Sprite.html#L1)
* [JavaScript/TypeScript license example](https://github.com/tensorflow/lucid/blob/master/lucid/scratch/js/src/index.js#L1)

### Contributor License Agreements

Please fill out either the individual or corporate Contributor License Agreement (CLA).

  * If you are an individual writing original source code and you're sure you own the intellectual property, then you'll need to sign an [individual CLA](https://code.google.com/legal/individual-cla-v1.0.html).
  * If you work for a company that wants to allow you to contribute your work, then you'll need to sign a [corporate CLA](https://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and instructions for how to sign and return it. Once we receive it, we'll be able to accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed the CLA can be accepted into the main repository.

# How to become a `collaborator`

We expect most contributors to use [pull requests](https://github.com/tensorflow/lucid/blob/master/CONTRIBUTING.md#pull-requests) to submit their contributions.

If you want to be involved more closely, we do welcome all help and are generally happy to add members of the community as collaborators. Collaborators can review & accept pull requests. We have the following expectations before we add you as a collaborator:

- You have submitted at least one substantial pull request that was merged.
- You have a reasonable expectation of making additional future contributions of similar substance.
- The collaborator status makes it easier for you to contribute. This could be, for example, because you contribute frequent small patches, or because you need to administrate this Github repository.
