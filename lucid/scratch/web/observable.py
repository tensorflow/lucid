import json
from lucid.misc.io.showing import _display_html

def renderObservable(url, cells=None, data=None):
  """Display observable notebook cells in iPython.

  Args:
    url: url fragment to observable notebook. ex: '@observablehq/downloading-and-embedding-notebooks'
    cells: an array of strings for the names of cells you want to render. ex: ['viewof stage', 'viewof x']
    data: a dictionary of variables that you'd like to overwrite. ex: {'x': 200, 'width': 500}
  """

  head = """
  <div id="output"></div>
  <div>
    <a target="_blank" href='https://observablehq.com/{}'>source</a>
  </div>
  <script type="module">
  """.format(url)
  
  runtimeImport = "import {Runtime} from 'https://unpkg.com/@observablehq/notebook-runtime?module';"
  
  notebookImport = "import notebook from 'https://api.observablehq.com/{0}.js';".format(url)
  
  cellsSerialized = "let cells = {};".format(json.dumps(cells))
  dataSerialized = "let data = {};".format(json.dumps(data))
  
  code = """
  const outputEl = document.getElementById("output");
  
  // Converts data into a map
  let dataMap = new Map();
  if (data) {
    Object.keys(data).forEach(key => {
      dataMap.set(key, data[key]);
    });
  }
  
  // Converts cells into a map
  let cellsMap = new Map();
  if (cells) {
    cells.forEach((key, i) => {
      const element = document.createElement("div");
      outputEl.appendChild(element)
      cellsMap.set(key, element)
    });
  }
  
  function render(_node, value) {
    if (!(value instanceof Element)) {
      const el = document.createElement("span");
      el.innerHTML = value;
      value = el;
    }
    if (_node.firstChild !== value) {
      if (_node.firstChild) {
        while (_node.lastChild !== _node.firstChild) _node.removeChild(_node.lastChild);
        _node.replaceChild(value, _node.firstChild);
      } else {
        _node.appendChild(value);
      }
    }
  }
  
  Runtime.load(notebook, (variable) => {
  
    // Override a variable with a passed value
    if (dataMap.has(variable.name)) {
      variable.value = dataMap.get(variable.name)
    }
    
    // Render the output to the corrent element
    if (cellsMap.has(variable.name)) {
      return { fulfilled: (value) => render(cellsMap.get(variable.name), value) }; 
    } else {
      return true;
    }
    
  });
  """
  
  foot = "</script>"
  
  _display_html(
      head + runtimeImport + notebookImport + cellsSerialized + dataSerialized + code + foot
  )
