import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools

COREMLTOOLS_SUPPORTED_VERSION = '4.1'

assert coremltools.__version__ == COREMLTOOLS_SUPPORTED_VERSION, \
       f"Please install coremltools version {COREMLTOOLS_SUPPORTED_VERSION}: " + \
       f"`python3 -m pip uninstall coremltools && python3 -m pip install coremltools==={COREMLTOOLS_SUPPORTED_VERSION}`\n" + \
       f"current version: {coremltools.__version__}"

from coremltools.converters.mil.mil import Builder as mb
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


########################################################################
######################## Test ml model #################################
########################################################################

IN_WH = 512
GRID_WH = 256

class TestModel(nn.Module):

    def forward(self, x, grid):
        grid_resized = self.resize_grid(grid)
        return F.grid_sample(
            x, grid_resized
        )

    def resize_grid(self, grid):
        # [1, GRID_WH, GRID_WH, 2] => [1, 2, GRID_WH, GRID_WH]
        grid_resized = grid.permute(0, 3, 1, 2)
        # [1, 2, GRID_WH, GRID_WH] => [1, 2, IN_WH, IN_WH]
        grid_resized = F.interpolate(
            grid_resized, 
            size=(IN_WH, IN_WH), 
            mode='nearest'
        )
        # [1, 2, IN_WH, IN_WH] => [1, IN_WH, IN_WH, 2]
        grid_resized = grid_resized.permute(0, 2, 3, 1)
        return grid_resized

########################################################################
########################################################################

def convert(output_path):
    torch_model = TestModel()
    example_input = torch.rand(1, 3, IN_WH, IN_WH) 
    example_grid = torch.rand(1, GRID_WH, GRID_WH, 2) 
    traced_model = torch.jit.trace(torch_model, (example_input, example_grid))

    mlmodel = coremltools.convert(
        traced_model,
        inputs=[
            coremltools.ImageType(name="image_input", shape=example_input.shape), 
            coremltools.TensorType(name="warp_grid", shape=example_grid.shape)
        ],
        minimum_deployment_target=coremltools.target["iOS13"]
    )
    mlmodel_path = output_path + ".mlmodel"
    mlmodel.save(mlmodel_path)

    spec = coremltools.utils.load_spec(mlmodel_path)

    output_layer = spec.description.output[0]
    output_layer.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    output_layer.type.imageType.height, output_layer.type.imageType.width = IN_WH, IN_WH
    coremltools.utils.rename_feature(spec, output_layer.name, 'output')

    coremltools.utils.save_spec(spec, mlmodel_path)

    shutil.copyfile(mlmodel_path, output_path)

    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--output', 
        default='./TorchCoreMLDemo/TorchCoreMLDemo/model.pb', 
        help='Output file'
    )
    args = parser.parse_args()
    convert(args.output)

if __name__ == "__main__":
    main()