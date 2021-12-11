import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools

from coremltools.converters.mil.mil import Builder as mb
from coremltools.models.neural_network.quantization_utils import quantize_weights
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.converters.mil.mil.ops.defs._op_reqs import (
    register_op, Operation, InputSpec, TensorInputType, IntInputType, BoolInputType,
    types
)
from coremltools.converters.mil.frontend.torch.ops import (
    _get_inputs as mil_get_inputs
)
from coremltools.converters.mil.backend.nn.op_mapping import (
    make_input
)
from coremltools.converters.mil import (
    register_torch_op
)
from coremltools.converters.mil.backend.nn.mil_to_nn_mapping_registry import (
    register_mil_to_nn_mapping
)

@register_mil_to_nn_mapping(override=True)
def grid_sample(const_context, builder, op):
    image_name = make_input(const_context, builder, op.input)
    grid_name = make_input(const_context, builder, op.grid)
    out_name = op.outputs[0].name

    suffix = "_prepared"
    input_names1 = [grid_name]
    out_names1 = [out_name + suffix]

    input_names2 = [image_name, out_names1[0]]
    out_names2 = [out_name]

    # transpose the grid to [n, 2, w, h] shape (for encoding it to a coreml 2-channel texture)
    builder.add_transpose(
        name=op.name + suffix,
        axes=(0, 3, 1, 2),
        input_name=input_names1[0],
        output_name=out_names1[0],
    )
    spec_layer = builder._add_generic_layer(op.name, input_names2, out_names2)

    spec_layer_params = spec_layer.custom
    spec_layer_params.className = "GridSampleLayer"
    
@register_op(doc_str="")
class grid_sample(Operation):
    
    input_spec = InputSpec(
        input=TensorInputType(),
        grid=TensorInputType(),
        mode=IntInputType(const=True),
        padding_mode=IntInputType(const=True),
        align_corners=BoolInputType(const=True),
    )

    bindings = {
        "class_name": "grid_sample",
        "input_order": ["input", "grid"],
        "parameters": ["mode", "padding_mode", "align_corners"],
        "description": "PyTorch grid_sample",
    }

    def __init__(self, **kwargs):
        super(grid_sample, self).__init__(**kwargs)

    def type_inference(self):
        input_type = self.input.dtype
        ret_shape = self.input.shape
        return types.tensor(input_type, ret_shape)

@register_torch_op(torch_alias=["grid_sampler"], override=True)
def torch_grid_sample(context, node):
    inputs = mil_get_inputs(context, node, expected=5)
    res = mb.grid_sample(
        input=inputs[0], 
        grid=inputs[1], 
        mode=inputs[2], 
        padding_mode=inputs[3], 
        align_corners=inputs[4],
        name=node.name
    )
    context.add(res)
