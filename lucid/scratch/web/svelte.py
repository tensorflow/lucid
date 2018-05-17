import random
import json
import tempfile
import subprocess
import os
import os.path as osp
import shutil
import uuid
import re

from IPython.core.magic import register_cell_magic

from lucid.misc.io.showing import _display_html
from lucid.misc.io.reading import read

_svelte_temp_dir = tempfile.mkdtemp(prefix="svelte_")
_svelte_shipped_dir = osp.join(osp.dirname(__file__), "components/")


_template = """
  <div id='$div_id'></div>
  <script>
  $js
  </script>
  <script>
    var app = new $name({
        target: document.querySelector('#$div_id'),
        data: $data,
      });
  </script>
  """

_default_svelte_script_tag = """
 <script>
   $auto_sub_component_imports
   export default {
     data() {
       return {};
     },
     components: $auto_sub_components
   }
 </script>
 """

def short_rand_str():
  """For this module, we need random names, with the following desirata:
  
  * Valid in js variable names and html ids.
  * Not too ugly when read by humans.
  
  We don't expect that many instances and so aren't very worried about
  collisions.
  """
  return str(uuid.uuid4()).replace('-', '')[:10]


def run_cmd(cmd):
  try:
    subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as exception:
    print(exception.output)
    raise RuntimeError("Failed executing shell command: " + cmd)

def build_svelte(html_fname, full_name):
  js_fname = html_fname.replace(".html", ".js")
  #cmd = "svelte compile --format iife " + html_fname + " > " + js_fname
  cmd = "(cd " + _svelte_temp_dir + "; ./node_modules/rollup/bin/rollup --config rollup.config.js -i " \
      + html_fname + " -n " + full_name + " -o " + js_fname + ")"
  run_cmd(cmd)
  return js_fname


def used_svelte_components(html):
  all_elements = re.findall("<[ ]*([a-zA-Z_]+)", html)
  upper_elements = [e for e in all_elements if e[0] == e[0].upper()]
  upper_elements = list(set(upper_elements))
  used_components = {}
  for e in upper_elements:
    if e in svelte_components:
      used_components[e] = svelte_components[e]
    else:
      print("Warning: Unknown component <" + e + ">.")
  return used_components.values()


def expand_svelte_html(html, sub_components):
  if "<script>" not in html:
    html += _default_svelte_script_tag

  auto_import_str = ""
  for component in sub_components:
    auto_import_str += "import " + component.full_name + " from " \
                    + "\"./" + component.full_name + ".html\";\n"

  auto_sub_components_str = "\n{\n"
  for component in sub_components:
    auto_sub_components_str += "  " + component.base_name + ": " \
                            + component.full_name + ",\n"
  auto_sub_components_str += "}"

  if "$auto_sub_components" in html:
    html = html.replace("$auto_sub_components", auto_sub_components_str)
  elif len(sub_components) > 0:
    print("Warning: despite appearing to use sub-components, $auto_sub_components is missing.")
  if "$auto_sub_component_imports" in html:
    html = html.replace("$auto_sub_component_imports", auto_import_str)
  elif len(sub_components) > 0:
    print("Warning: despite appearing to use sub-components, $auto_sub_component_imports is missing.")

  return html

def component_from_html(base_name, html):
  full_name = base_name + "_" + short_rand_str()
  html_fname = osp.join(_svelte_temp_dir, full_name + ".html")
  sub_components = used_svelte_components(html)

  html = expand_svelte_html(html, sub_components)

  with open(html_fname, "w") as f:
    f.write(html)
  return SvelteComponent(base_name, full_name, html_fname,
                         sub_components=sub_components)


svelte_components = {}


class SvelteComponent:

  def __init__(self, base_name, full_name, html_path, sub_components=()):
    """Display svelte components in iPython.
    Args:
      name: name of svelte component (must match component filename when built)
      html_path: source svelte .html file.
    Returns:
      A function mapping data to a rendered svelte component in ipython.
    """
    assert html_path[-5:] == ".html"
    #print("Trying to build svelte component from html...")
    js_path = build_svelte(html_path, full_name)
    self.base_name = base_name
    self.full_name = full_name
    self.html_content = read(html_path) # For debugging
    self.js_content = read(js_path)
    self.sub_components = sub_components

  def register(self):
    if self.base_name in svelte_components:
      print("Redefining " + self.base_name + " ...")
    svelte_components[self.base_name] = self

  def make_instance_html(self, data):
    full_js_content = ""
    #for component in self.sub_components:
    #  full_js_content += component.js_content + "\n"
    full_js_content += self.js_content
    div_id = self.base_name + "_div_" + short_rand_str()
    html = _template \
        .replace("$js", full_js_content) \
        .replace("$name", self.full_name) \
        .replace("$data", json.dumps(data)) \
        .replace("$div_id", div_id)
    return html

  def __call__(self, data):
    html = self.make_instance_html(data)
    _display_html(html)


@register_cell_magic
def html_define_svelte(line, cell):
  base_name = line.split()[0]
  component_from_html(base_name, cell).register()


@register_cell_magic
def html_svelte(line, cell):
  component_from_html("Temp", cell)({})


# Copy over compoents / libraries
for fname in os.listdir(_svelte_shipped_dir):
  src_path = osp.join(_svelte_shipped_dir, fname)
  dst_path = osp.join(_svelte_temp_dir, fname)
  shutil.copy(src_path, dst_path)    

# npm install to get dependencies
setup_cmd = "(cd " + _svelte_temp_dir + "; npm install)"
run_cmd(setup_cmd)

# Build & register all shipped components
for fname in os.listdir(_svelte_temp_dir):
  dst_path = osp.join(_svelte_temp_dir, fname)
  name, ext = osp.splitext(osp.split(dst_path)[1])
  if ext == ".html":
    SvelteComponent(name, name, dst_path).register()
