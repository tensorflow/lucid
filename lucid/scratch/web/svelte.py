import random
import json

from lucid.misc.io.showing import _display_html
from lucid.misc.io.reading import read


_template = """
        <div id='$id'></div>
        <script>$js</script>
        <script>
          var app = new $name({
              target: document.querySelector('#$id'),
              data: $data,
            });
        </script>
       """

def SvelteComponent(name, js_path):
    js_content = read(js_path)
    def inner(data):
        id_str = name + "_" + hex(random.randint(0, 1e8))[2:]
        html = _template \
            .replace("$js", js_content) \
            .replace("$name", name) \
            .replace("$data", json.dumps(data)) \
            .replace("$id", id_str)
        _display_html(html)
    return inner

