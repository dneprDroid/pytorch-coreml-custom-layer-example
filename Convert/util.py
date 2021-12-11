import torch
import json

def generate_grid(path):
    d = torch.linspace(-1, 1, 256)
    meshx, meshy = torch.meshgrid((d, d))

    meshx = meshx * 0.3
    meshy = meshy * 0.9
    
    meshz = torch.sin(torch.sqrt(meshx * meshx + meshy * meshy))
    grid = torch.stack((meshz, meshx), 2)
    grid = grid.unsqueeze(0)
    print(f"grid: {grid.shape}")

    array = grid.numpy().tolist()
    jstr = json.dumps(array)

    with open(path, 'w') as f:
        f.write(jstr)


# generate_grid("./TorchCoreMLDemo/TorchCoreMLDemo/grid.json")