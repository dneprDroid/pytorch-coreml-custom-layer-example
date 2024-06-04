import os
import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools

import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import (
    _get_inputs as mil_get_inputs
)
from coremltools.converters.mil import (
    register_torch_op
)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil import (
    Operation,
    types
)
from coremltools.converters.mil.mil.input_type import (
    InputSpec,
    TensorInputType,
)
    
@register_op(is_custom_op=True)
class grid_sample(Operation):
    
    input_spec = InputSpec(
        input=TensorInputType(type_domain="T"),
        grid=TensorInputType(type_domain="T"),

        mode=TensorInputType(const=True, type_domain=types.int32),
        padding_mode=TensorInputType(const=True, type_domain=types.int32),
        align_corners=TensorInputType(const=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (types.fp32,),
    }

    bindings = {
        "class_name": "GridSampleLayer",
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

    grid = inputs[1] 
    grid_transposed = mb.transpose(
        x=grid,
        perm=[0, 3, 1, 2],
        name=node.name + "__grid_transposed"
    )
    res = mb.grid_sample(
        input=inputs[0], 
        grid=grid_transposed, 
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
        convert_to="neuralnetwork",
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
    current_dir = os.path.dirname(os.path.realpath(__file__))
    default_out_path = os.path.join(current_dir, '../TorchCoreMLDemo/TorchCoreMLDemo/Assets/model.pb')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--output', 
        default=default_out_path, 
        help='Output file'
    )
    args = parser.parse_args()
    convert(args.output)

if __name__ == "__main__":
    main()