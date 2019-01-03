import random
import json
import tempfile
import subprocess
import os.path as osp
import uuid

from IPython.core.magic import register_cell_magic

from lucid.misc.io.showing import _display_html
from lucid.misc.io.reading import read

_svelte_temp_dir = tempfile.mkdtemp(prefix="svelte_")

_template = """
  <div id='$id'></div>
  <script>
  $js
  </script>
  <script>
    var app = new $name({
        target: document.querySelector('#$id'),
        data: $data,
      });
  </script>
 """

def js_id(name):
  # name_str will become the name of a javascript variable, and can't contain dashes.
  return name + "_" + str(uuid.uuid4()).replace('-', '_')

def build_svelte(html_fname):
  js_fname = html_fname.replace(".html", ".js")
  cmd = "svelte compile --format iife " + html_fname + " > " + js_fname
  print(cmd)
  try:
    print(subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT))
  except subprocess.CalledProcessError as exception:
    print("Svelte build failed! Output:\n{}".format(exception.output.decode()))
  return js_fname


def SvelteComponent(name, path):
  """Display svelte components in iPython.

  Args:
    name: name of svelte component (must match component filename when built)
    path: path to compile svelte .js file or source svelte .html file.
      (If html file, we try to call svelte and build the file.)

  Returns:
    A function mapping data to a rendered svelte component in ipython.
  """
  if path[-3:] == ".js":
    js_path = path
  elif path[-5:] == ".html":
    print("Trying to build svelte component from html...")
    js_path = build_svelte(path)
  js_content = read(js_path, mode='r')
  def inner(data):
    id_str = js_id(name)
    html = _template \
        .replace("$js", js_content) \
        .replace("$name", name) \
        .replace("$data", json.dumps(data)) \
        .replace("$id", id_str)
    _display_html(html)
  return inner


@register_cell_magic
def html_define_svelte(line, cell):
  base_name = line.split()[0]
  id_str = js_id(base_name)
  html_fname = osp.join(_svelte_temp_dir, id_str + ".html")
  with open(html_fname, "w") as f:
    f.write(cell)
  component = SvelteComponent(id_str, html_fname)
  globals()[base_name] = component
