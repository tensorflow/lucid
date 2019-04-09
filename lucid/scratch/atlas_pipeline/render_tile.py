"""
Controller that runs the user defined render function on each cell in a tile
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import math

def render_tile(cells, ti, tj, render, params, metadata, layout, summary):
  """
    Render each cell in the tile and stitch it into a single image
  """
  image_size = params["cell_size"] * params["n_tile"]
  tile = Image.new("RGB", (image_size, image_size), (255,255,255))
  keys = cells.keys()
  for i,key in enumerate(keys):
    print("cell", i+1, "/", len(keys), end='\r')
    cell_image = render(cells[key], params, metadata, layout, summary)
    # stitch this rendering into the tile image
    ci = key[0] % params["n_tile"]
    cj = key[1] % params["n_tile"]
    xmin = ci*params["cell_size"]
    ymin = cj*params["cell_size"]
    xmax = (ci+1)*params["cell_size"]
    ymax = (cj+1)*params["cell_size"]

    if params.get("scale_density", False):
      density = len(cells[key]["gi"])
      # scale = density/summary["max_density"]
      scale = math.log(density)/(math.log(summary["max_density"]) or 1)
      owidth = xmax - xmin
      width = int(round(owidth * scale))
      if(width < 1):
        width = 1
      offsetL = int(round((owidth - width)/2))
      offsetR = owidth - width - offsetL # handle odd numbers
      # print("\n")
      # print("width", width, offsetL, offsetR)
      box = [xmin + offsetL, ymin + offsetL, xmax - offsetR, ymax - offsetR]
      resample = params.get("scale_type", Image.NEAREST)
      cell_image = cell_image.resize(size=(width,width), resample=resample)
      # print(cell_image)
    else:
      box = [xmin, ymin, xmax, ymax]

    # print("box", box)
    tile.paste(cell_image, box)
  print("\n")
  return tile


def aggregate_tile(cells, ti, tj, aggregate, params, metadata, layout, summary):
  """
    Call the user defined aggregation function on each cell and combine into a single json object
  """
  tile = []
  keys = cells.keys()
  for i,key in enumerate(keys):
    print("cell", i+1, "/", len(keys), end='\r')
    cell_json = aggregate(cells[key], params, metadata, layout, summary)
    tile.append({"aggregate":cell_json, "i":int(key[0]), "j":int(key[1])})
  return tile
 