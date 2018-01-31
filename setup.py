from setuptools import setup, find_packages

test_deps = [
  'future',
  'twine',
  'pytest',
]

extras = {
  'test': test_deps,
}

setup(
  name = 'lucid',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  description = ('Collection of infrastructure and tools for research in '
    'neural network interpretability.'),
  author = 'The Deepviz Authors',
  author_email = 'deepviz@google.com',
  url = 'https://github.com/tensorflow/lucid',
  download_url = 'https://github.com/tensorflow/lucid/archive/0.0.1.tar.gz',
  license = 'Apache License 2.0',
  keywords = ['tensorflow', 'tensor', 'machine learning', 'neural networks',
    'convolutional neural networks', 'feature visualization', 'optimization'],
  install_requires = [
    'future',
    'decorator',
    'tensorflow',
    'numpy',
  ],
  tests_require = test_deps,
  extras_require=extras,
  classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Education',
    'License :: OSI Approved :: Apache Software License',
    'Natural Language :: English',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    # 'Programming Language :: Python :: 3.4',
    # 'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    # 'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Software Development :: Libraries :: Python Modules',
  ],
)
