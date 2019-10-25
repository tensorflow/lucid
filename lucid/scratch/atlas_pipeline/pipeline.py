"""
Take user input data and run it through the grid and tile rendering pipeline
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import lucid.scratch.atlas_pipeline.render_tile as render_tile
import lucid.scratch.atlas_pipeline.grid as grid
from tensorflow import gfile

# TODO: this is where we would distribute the tile processing
def run(render, aggregate, params, metadata, layout):
  
  gfile.MakeDirs("{directory}".format(**params))

  layers = params["n_cells"]
  for n_layer in layers:
    params["n_layer"] = n_layer
    tiles = grid.grid(metadata, layout, params)

    # TODO: write out summary? might be useful for the client-side renderer
    summary = summarize(tiles, params, layout, metadata)
    print("summary", summary)
    etiles = grid.enumerate_tiles(tiles)
    for i,t in enumerate(etiles):
      ti = t[0]
      tj = t[1]
      tile = t[2] 
      cells = grid.tile_cells(tile)

      print("aggregate tile", i+1, "/", len(etiles))
      tile_json = render_tile.aggregate_tile(cells, ti, tj, aggregate, params, metadata, layout, summary)
      filename = "{directory}/{name}-tile_{n_layer}_{n_tile}_{ti}_{tj}.json".format(ti=ti, tj=tj, **params)
      print("saving", filename)
      with gfile.Open(filename, 'w') as f:
        json.dump(tile_json, f)

      print("render tile", i+1, "/", len(etiles))
      tile_img = render_tile.render_tile(cells, ti, tj, render, params, metadata, layout, summary)
      filename = "{directory}/{name}-tile_{n_layer}_{n_tile}_{ti}_{tj}.png".format(ti=ti, tj=tj, **params)
      print("saving", filename)
      with gfile.Open(filename, 'w') as f:
        tile_img.save(f)

def summarize(tiles, params, layout, metadata):
  # calculate summary statistics that may be useful for rendering
  # e.g. max point density, bounds of x and y
  # we can also track the parameters for convenience
  summary = {
  }
  x = layout["x"]
  y = layout["y"]
  summary["x_min"] = np.min(x)
  summary["x_max"] = np.max(x)
  summary["y_min"] = np.min(y)
  summary["y_max"] = np.max(y)
  # the size of a cell in layout coordinate space
  summary["x_bin"] = abs(summary["x_max"] - summary["x_min"])/params["n_layer"]
  summary["y_bin"] = abs(summary["y_max"] - summary["y_min"])/params["n_layer"]

  # maximum density of a cell
  max_density = 0
  min_density = float("inf")
  user_max_density = 0
  user_min_density = float("inf")

  num_cells = 0
  total_count = 0
  user_total_count = 0
  def density(cell, metadata):
    return len(cell["gi"])
  # if the user doesn't provide a scale function its just the density
  user_density = params.get("density_function", density)

  for t in grid.enumerate_tiles(tiles):
    tile = t[2] 
    cells = grid.tile_cells(tile) 
    keys = cells.keys()
    for i,key in enumerate(keys):
      num_cells += 1
      num_points = density(cells[key], metadata)
      user_num_points = user_density(cells[key], metadata)
      
      total_count += num_points
      user_total_count += user_num_points
      if num_points > max_density:
        max_density = num_points
      if num_points < min_density:
        min_density = num_points
      if user_num_points > user_max_density:
        user_max_density = user_num_points
      if user_num_points < user_min_density:
        user_min_density = user_num_points
  summary["max_density"] = max_density
  summary["min_density"] = min_density
  summary["total_count"] = total_count
  summary["user_max_density"] = user_max_density
  summary["user_min_density"] = user_min_density
  summary["user_total_count"] = user_total_count
  summary["num_cells"] = num_cells
  # TODO: enable summarizing of meta attributes?
  return summary
