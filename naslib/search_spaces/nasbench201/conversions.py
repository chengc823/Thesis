"""
There are three representations
'naslib': the NASBench201SearchSpace object
'op_indices': A list of six ints, which is the simplest representation
'arch_str': The string representation used in the original nasbench201 paper

This file currently has the following conversions:
naslib -> op_indices
op_indices -> naslib
naslib -> arch_str

Note: we could add more conversions, but this is all we need for now
"""

import torch
from naslib.search_spaces.core.primitives import AbstractPrimitive

OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]
OP_NAMES_NB201 = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']

EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
OPS_TO_NB201 = {
    "AvgPool1x1": "avg_pool_3x3",
    "ReLUConvBN1x1": "nor_conv_1x1",
    "ReLUConvBN3x3": "nor_conv_3x3",
    "Identity": "skip_connect",
    "Zero": "none",
}


def convert_naslib_to_op_indices(naslib_object):
    cell = naslib_object._get_child_graphs(single_instances=True)[0]
    ops = []
    for i, j in EDGE_LIST:
        ops.append(cell.edges[i, j]["op"].get_op_name)

    return [OP_NAMES.index(name) for name in ops]


def convert_op_indices_to_naslib(op_indices, naslib_object):
    """
    Converts op indices to a naslib object
    input: op_indices (list of six ints)
    naslib_object is an empty NasBench201SearchSpace() object.
    Do not call this method with a naslib object that has already been
    discretized (i.e., all edges have a single op).

    output: none, but the naslib object now has all edges set
    as in genotype.

    warning: this method will modify the edges in naslib_object.
    """

    # create a dictionary of edges to ops
    edge_op_dict = {}
    for i, index in enumerate(op_indices):
        edge_op_dict[EDGE_LIST[i]] = OP_NAMES[index]

    def add_op_index(edge):
        # function that adds the op index from the dictionary to each edge
        if (edge.head, edge.tail) in edge_op_dict:
            for i, op in enumerate(edge.data.op):
                if op.get_op_name == edge_op_dict[(edge.head, edge.tail)]:
                    index = i
                    break
            edge.data.set("op_index", index, shared=True)

    def update_ops(edge):
        # function that replaces the primitive ops at the edges with the one in op_index
        if isinstance(edge.data.op, list):
            primitives = edge.data.op
        else:
            primitives = edge.data.primitives

        chosen_op = primitives[edge.data.op_index]
        primitives[edge.data.op_index] = update_batchnorms(chosen_op)

        edge.data.set("op", primitives[edge.data.op_index])
        edge.data.set("primitives", primitives)  # store for later use

    def update_batchnorms(op: AbstractPrimitive) -> AbstractPrimitive:
        """ Makes batchnorms in the op affine, if they exist """
        init_params = op.init_params
        has_batchnorm = False

        for module in op.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                has_batchnorm = True
                break

        if not has_batchnorm:
            return op

        if 'affine' in init_params:
            init_params['affine'] = True
        if 'track_running_stats' in init_params:
            init_params['track_running_stats'] = True

        new_op = type(op)(**init_params)
        return new_op

    naslib_object.update_edges(
        add_op_index, scope=naslib_object.OPTIMIZER_SCOPE, private_edge_data=False
    )

    naslib_object.update_edges(
        update_ops, scope=naslib_object.OPTIMIZER_SCOPE, private_edge_data=True
    )


def convert_naslib_to_str(naslib_object):
    """Converts naslib object to string representation."""

    cell = naslib_object.edges[2, 3].op
    edge_op_dict = {
        (i, j): OPS_TO_NB201[cell.edges[i, j]["op"].get_op_name] for i, j in cell.edges
    }
    op_edge_list = [
        "{}~{}".format(edge_op_dict[(i, j)], i - 1)
        for i, j in sorted(edge_op_dict, key=lambda x: x[1])
    ]

    return "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)


def convert_str_to_op_indices(str_encoding):
    """Converts NB201 string representation to op_indices"""
    nodes = str_encoding.split('+')

    def get_op(x):
        return x.split('~')[0]

    node_ops = [list(map(get_op, n.strip()[1:-1].split('|'))) for n in nodes]

    enc = []
    for u, v in EDGE_LIST:
        enc.append(OP_NAMES_NB201.index(node_ops[v - 2][u - 1]))

    return tuple(enc)


def convert_op_indices_to_str(op_indices):
    edge_op_dict = {
        edge: OP_NAMES_NB201[op] for edge, op in zip(EDGE_LIST, op_indices)
    }

    op_edge_list = [
        "{}~{}".format(edge_op_dict[(i, j)], i - 1)
        for i, j in sorted(edge_op_dict, key=lambda x: x[1])
    ]

    return "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)
