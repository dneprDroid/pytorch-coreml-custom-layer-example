#include <metal_stdlib>

using namespace metal;

static inline float unnormalize(const float coord, const int size) {
    return ((coord + 1) * size - 1) / 2;
}

static inline float compute_source_index(const float coord, const int size) {
    float new_coord = unnormalize(coord, size);
    // TODO: support padding mode
    return new_coord;
}

kernel void grid_sampler(texture2d_array<half, access::sample> in [[texture(0)]],
                         texture2d_array<half, access::sample> grid [[texture(1)]],
                         texture2d_array<half, access::write>  out [[texture(2)]],
                         ushort3 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() ||
        gid.y >= out.get_height() ||
        gid.z >= out.get_array_size()) {
        return;
    }
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);

    const int inp_H = in.get_height();
    const int inp_W = in.get_width();
    
    const int w = gid.x;
    const int h = gid.y;
    const int n = gid.z;
    
    const float4 grid_XY = float4(grid.sample(sample, float2(w, h), n));
    const float x = grid_XY[0];
    const float y = grid_XY[1];

    const float ix = compute_source_index(x, inp_W);
    const float iy = compute_source_index(y, inp_H);
    
    const int ix_nw = int(floor(ix));
    const int iy_nw = int(floor(iy));

    const int ix_ne = ix_nw + 1;
    const int iy_ne = iy_nw;

    const int ix_sw = ix_nw;
    const int iy_sw = iy_nw + 1;

    const int ix_se = ix_nw + 1;
    const int iy_se = iy_nw + 1;

    const float nw = (ix_se - ix)    * (iy_se - iy);
    const float ne = (ix    - ix_sw) * (iy_sw - iy);
    const float sw = (ix_ne - ix)    * (iy    - iy_ne);
    const float se = (ix    - ix_nw) * (iy    - iy_nw);

    float4 result = float4(0, 0, 0, 1);
    
    result += float4(in.sample(sample, float2(ix_nw, iy_nw), n)) * nw;
    result += float4(in.sample(sample, float2(ix_ne, iy_ne), n)) * ne;
    result += float4(in.sample(sample, float2(ix_sw, iy_sw), n)) * sw;
    result += float4(in.sample(sample, float2(ix_se, iy_se), n)) * se;

    out.write(half4(result), ushort2(w, h), n);
}
