import json
import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
from lucid.misc.io import show, load, save
from lucid.misc.io.showing import _image_url, _display_html
import lucid.scratch.web.svelte as lucid_svelte
from collections import defaultdict

from lucid.misc.graph_analysis.overlay_graph import OverlayGraph, OverlayNode
import lucid.misc.graph_analysis.filter_overlay as filter_overlay
import lucid.misc.graph_analysis.property_inference as property_inference
from lucid.misc.graph_analysis.parse_overlay import parse_overlay, toplevel_group_data

from clarity.dask.cluster import get_client


def node_chain_between(a, b):
    pres = b
    chain = []
    while pres.op.inputs:
        pres = pres.op.inputs[0]
        if pres == a:
            return chain
        chain.append(pres)
    return []


def graph_metadata(model):
    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:
        tf.import_graph_def(model.graph_def, name="")
        overlay = OverlayGraph(graph)
        # Get rid of most random nodes we aren't intersted in.
        overlay = filter_overlay.ops_whitelist(overlay)
        # Get rid of nodes that aren't effected by the input.
        overlay = filter_overlay.is_dynamic(overlay)
        # Collapse sequences of nodes taht we want to think of as a single layer.
        overlay = filter_overlay.collapse_sequence(overlay, ["Conv2D", "Relu"])
        overlay = filter_overlay.collapse_sequence(overlay, ["MatMul", "Relu"])
        overlay = filter_overlay.collapse_sequence(overlay, ["MatMul", "Softmax"])

        # collect metadat using overlay graph
        weights_meta = {}
        node_meta = {}
        for node in overlay.nodes:
            meta = {}
            node_meta[node.name] = meta

            meta["inputs"] = [inp.name for inp in node.inputs]
            meta["op_type"] = node.op

            # dataformat
            data_format = property_inference.infer_data_format(node.tf_node)
            if data_format and len(node.tf_node.shape) == 4:
                if data_format != "NHWC":
                    meta["data_format"] = data_format

            # shape
            shape = node.tf_node.shape
            if not str(shape) == "<unknown>":
                #     meta["is_conv"] = len(shape) == 4
                meta["rank"] = len(shape)
                meta["channels"] = int(shape[-1])
                meta["shape"] = shape

            # weights
            if len(node.inputs) == 1:
                potential_convs = node_chain_between(
                    list(node.inputs)[0].tf_node, node.tf_node
                ) + [node.tf_node]
                convs = [t for t in potential_convs if t.op.type == "Conv2D"]

                if len(convs) == 1:
                    weight_name = convs[0].op.inputs[1].name
                    meta["conv_weight"] = weight_name
                    weights_meta[weight_name] = {
                        "input": list(node.inputs)[0].name,
                        "output": node.name,
                        "shape": convs[0].op.inputs[1].shape.as_list(),
                    }

            # handle concat nodes / branches
            if node.op in ["Concat", "ConcatV2"]:
                if node.op == "Concat":
                    raw_inputs = [inp.name for inp in node.tf_node.op.inputs[1:]]
                if node.op == "ConcatV2":
                    raw_inputs = [inp.name for inp in node.tf_node.op.inputs[:-1]]

                node_input_names = [inp.name for inp in node.inputs]
                if set(raw_inputs) == set(node_input_names):
                    n_channel = 0
                    meta["branches"] = {}
                    for branch in raw_inputs:
                        size = int(overlay[branch].tf_node.shape[-1])
                        meta["branches"][branch] = [n_channel, n_channel + size]
                        n_channel += size

        metadata = {
            "type": "model",
            "name": model.name,
            "ops": node_meta,
            "weights": weights_meta,
            "layout": parse_overlay(overlay),
            "groups": toplevel_group_data(overlay),
        }
        return metadata  # save(metadata, "metadata.json")

