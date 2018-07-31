
# Extract list of urls for checkpoints
cat slim_readme_table.md          \
  | sed "s/[|\\(\\)]/\n/g"        \
  | grep download.tensorflow.org  \
  > url_list.txt

# Create proposed list of model names (will need to be modified)
cat url_list.txt                                      \
  | sed "s/\\//\n/g"                                  \
  | grep "gz$"                                        \
  | sed "s/_201[0-9]_[0-9][0-9]_[0-9][0-9].tar.gz//g" \
  | sed "s/_[0-9][0-9]_[0-9][0-9]_201[0-9].tar.gz//g" \
  > names.txt

# TODO - make sure to sanity check the output of this script before
# running the next one.
