# pytorch-coreml-custom-layer-example
PyTorch to CoreML: Writing custom layers with Metal

<img width="200" alt="pytorch-demo-image" src="https://user-images.githubusercontent.com/13742733/176658252-533cdf6f-a993-4c1c-9393-2a926398f653.png">

## Convert PyTorch model

```bash

cd Convert
python3 -m pip install -r requirements.txt
python3 convert.py -o /path/to/output/resulting/mlmodel/file

 ```

## Run the demo app
Open `TorchCoreMLDemo.xcodeproj` in Xcode, build and run the app.

## Shader 
The code of the grid sample Metal-shader is [here](./TorchCoreMLDemo/TorchCoreMLDemo/Metal/GridSample.metal).

More information about grid sample (warper) algorithm you can find [here](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).
