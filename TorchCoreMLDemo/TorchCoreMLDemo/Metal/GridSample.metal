#include <metal_stdlib>

using namespace metal;


kernel void grid_sampler(texture2d_array<half, access::sample> in [[texture(0)]],
                         texture2d_array<half, access::sample> grid [[texture(1)]],
                         texture2d_array<half, access::write>  out [[texture(2)]],
                         ushort3 gid [[thread_position_in_grid]]) {
}
