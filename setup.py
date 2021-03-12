# Copyright 2017 The TensorFlow Lucid Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script for TensorFlow Lucid."""

import lucid
from setuptools import setup, find_packages

version = lucid.__version__

test_deps = ["future", "twine", "pytest", "pytest-mock", "python-coveralls"]

extras = {
    "test": test_deps,
    "tf": ["tensorflow>=1.6.0"],
    "tf_gpu": ["tensorflow-gpu>=1.6.0"],
}

setup(
    name="lucid",
    packages=find_packages(exclude=[]),
    version=version,
    description=(
        "Collection of infrastructure and tools for research in "
        "neural network interpretability."
    ),
    author="The Lucid Authors",
    author_email="deepviz@google.com",
    url="https://github.com/tensorflow/lucid",
    download_url=(
        "https://github.com/tensorflow/lucid" "/archive/v{}.tar.gz".format(version)
    ),
    license="Apache License 2.0",
    keywords=[
        "tensorflow",
        "tensor",
        "machine learning",
        "neural networks",
        "convolutional neural networks",
        "feature visualization",
        "optimization",
    ],
    install_requires=[
        "numpy<=1.19",
        "scipy",
        "scikit-learn",
        "umap-learn",
        "nltk",
        "ipython",
        "pillow",
        "future",
        "decorator",
        "pyopengl",
        "filelock",
        "cachetools",
        "more-itertools",
    ],
    setup_requires=["pytest-runner"],
    tests_require=test_deps,
    extras_require=extras,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
