#!/bin/sh
# Run this script once to create the workspace for us to export models
# from caffe. This is necessary because caffe-tensorflow needs old tf versions.

# Create and enter virtualenv
virtualenv workspace
cd workspace
. bin/activate

# Install virtualenv dependencies
pip install tensorflow==1.2
git clone https://github.com/linkfluence/caffe-tensorflow
