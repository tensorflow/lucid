#!/bin/sh

# Config
MODEL_NAME="VGG_ILSVRC_16_layers"
PROTO_URL="https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt"
CAFFE_URL="http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel"

# Enter virtualenv (for old TF version, etc)
cd workspace/
. bin/activate

# Fetch caffe model files
echo ""
echo "----------------------------------------"
echo "Fetching models"
echo "----------------------------------------"
curl -sS ${PROTO_URL} --output ${MODEL_NAME}.prototxt
curl -sS ${CAFFE_URL} --output ${MODEL_NAME}.caffemodel
  
# Upgrade prototxt, in case it's for an old unsupported version of caffe.
echo ""
echo "----------------------------------------"
echo "Attempting to upgrade proto file format"
echo "----------------------------------------"
upgrade_net_proto_text ${MODEL_NAME}.prototxt ${MODEL_NAME}_upgraded.prototxt

# caffe-tensorflow will fail if .tmp/ already exists
rm -rf .tmp/

# Run caffe-tensorflow conversion script
echo ""
echo "----------------------------------------"
echo "Attempting to convert to tensorflow"
echo "----------------------------------------"
python caffe-tensorflow/convert.py           \
  --caffemodel ${MODEL_NAME}.caffemodel      \
  --standalone-output-path ${MODEL_NAME}.pb  \
  ${MODEL_NAME}_upgraded.prototxt

echo "Frozen tensflow graph:"
echo "  workspace/${MODEL_NAME}.pb"
