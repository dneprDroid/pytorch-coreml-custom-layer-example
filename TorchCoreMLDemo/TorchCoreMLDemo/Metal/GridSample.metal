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

static inline bool is_safe_index(const int x, const int y, const int w, const int h) {
  return x >= 0 && x < w && y >= 0 && y < h;
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

    const int inp_W = in.get_width();
    
    const int w = gid.x;
    const int h = gid.y;
    const int n = gid.z;
    
    const half4 result = in.sample(sample, float2(inp_W - w, h), n);
    
    out.write(result, ushort2(w, h), n);
}
