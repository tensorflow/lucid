# Importing models from tf-slim

(1) Install dependencies *into this directory*:
  ```sh
  git clone https://github.com/tensorflow/models.git
  wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/python/tools/freeze_graph.py
  ```
(2) Save latest table of slim models and checkpoints as `slim_readme_table.md`
(3) Run `parse_readme_table.sh` (from within this directory)
(4) Sanity check the output (`url_list.txt` and `names.txt`)
(5) Run `import_models.ipy` (from within this directory)
(6) See `frozen/*.pb` and `model_defs.py`
