

One way to import caffe models is [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) which now has a `--standalone-output-path` for directly crating frozen models (see ethereon/caffe-tensorflow#76).

Getting this to work reliably takes a number of tricks, including using an old tensorflow version, running a caffe model update script, and manually cleaning up temp files. The included scripts should handle all of this for you.

(1) Install global dependencies:
  ```sh
  apt install -y caffe-cpu
  pip install virtualenv
  ```
(2) Run `make_workspace.sh` (in this directory) to create isolated workspace
(3) Edit the "Config" section in `import_model.sh`
(4) Run `import_model.sh` (in this directory) to import specified model
