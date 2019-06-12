from lucid.misc.graph_analysis.overlay_graph import OverlayGraph, OverlayNode, OverlayStructure

def collapse_sequences(overlay):
  """Detect and collapse sequences of nodes in an overlay."""
  sequences = []
  for node in overlay.nodes:
    if any([node in seq for seq in sequences]): continue
    seq = [node]
    while len(node.consumers) == 1 and len(list(node.consumers)[0].inputs) == 1:
      node = list(node.consumers)[0]
      seq.append(node)
    if len(seq) > 1:
      sequences.append(seq)

  structure_map = {}
  for seq in sequences:
    structure_map[seq[-1]] = OverlayStructure("Sequence", {"sequence": seq})

  return overlay.collapse_structures(structure_map)


def collapse_branches(overlay):
  """Detect and collapse brances of nodes in an overlay."""
  structure_map = {}

  for node in overlay.nodes:
    if len(node.inputs) <= 1: continue
    gcd = node.gcd
    if all(inp == gcd or inp.inputs == set([gcd]) for inp in node.inputs):
      branches = [inp if inp != gcd else None
                  for inp in overlay.sorted(node.inputs)]
      structure_map[node] = OverlayStructure("HeadBranch", {"branches" : branches, "head": node})

  for node in overlay.nodes:
    if len(node.consumers) <= 1: continue
    if all(len(out.consumers) == 0 for out in node.consumers):
      branches = overlay.sorted(node.consumers)
      max_node = overlay.sorted(branches)[-1]
      structure_map[max_node] = OverlayStructure("TailBranch", {"branches" : branches, "tail": node})

  return overlay.collapse_structures(structure_map)


def parse_structure(node):
  """Turn a collapsed node in an OverlayGraph into a heirchaical grpah structure."""
  if node is None:
    return None

  structure = node.sub_structure

  if structure is None:
    return node.name
  elif structure.structure_type == "Sequence":
    return {"Sequence" : [parse_structure(n) for n in structure.structure["sequence"]]}
  elif structure.structure_type == "HeadBranch":
    return {"Sequence" : [
        {"Branch" : [parse_structure(n) for n in structure.structure["branches"]] },
        parse_structure(structure.structure["head"])
    ]}
  elif structure.structure_type == "TailBranch":
    return {"Sequence" : [
        parse_structure(structure.structure["tail"]),
        {"Branch" : [parse_structure(n) for n in structure.structure["branches"]] },
    ]}
  else:
    data = {}
    for k in structure.structure:
      if isinstance(structure.structure[k], list):
        data[k] = [parse_structure(n) for n in structure.structure[k]]
      else:
        data[k] = parse_structure(structure.structure[k])

    return {structure.structure_type : data}


def flatten_sequences(structure):
  """Flatten nested sequences into a single sequence."""
  if isinstance(structure, str) or structure is None:
    return structure
  else:
    structure = structure.copy()
    for k in structure:
      structure[k] = [flatten_sequences(sub) for sub in structure[k]]

  if "Sequence" in structure:
    new_seq = []
    for sub in structure["Sequence"]:
      if isinstance(sub, dict) and "Sequence" in sub:
        new_seq += sub["Sequence"]
      else:
        new_seq.append(sub)
    structure["Sequence"] = new_seq
  return structure


def parse_overlay(overlay):
  new_overlay = overlay
  prev_len = len(overlay.nodes)

  collapsers = [collapse_sequences, collapse_branches]

  while True:
    new_overlay = collapse_branches(collapse_sequences(new_overlay))
    if not len(new_overlay.nodes) < prev_len:
      break
    prev_len = len(new_overlay.nodes)

  if len(new_overlay.nodes) != 1: return None


  return flatten_sequences(parse_structure(new_overlay.nodes[0]))


def _namify(arr):
  return [x.name for x in arr]

def toplevel_group_data(overlay):
  pres = overlay.nodes[-1]
  tops = [pres]
  while pres.inputs:
    pres = pres.gcd
    tops.append(pres)
  tops = tops[::-1]

  groups = {}

  for top in tops:
    if top.op in ["Concat", "ConcatV2"]:
      groups[top.name] = {
          "immediate" : _namify(overlay.sorted(top.inputs)),
          "all" : _namify(overlay.sorted(top.extended_inputs - top.gcd.extended_inputs - set([top.gcd]) | set([top]))),
          "direction" : "backward"
      }
    if len(top.consumers) > 1 and all(out.op == "Split" for out in top.consumers):
      groups[top.name] = {
          "immediate" : _namify(overlay.sorted(top.consumers)),
          "all" : _namify(overlay.sorted(top.extended_consumers - top.lcm.extended_consumers - set([top.lcm]) | set([top]))),
          "direction" : "forward"
      }

  return groups
