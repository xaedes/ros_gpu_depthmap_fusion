#version 430 core

// mask_dilate

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_MASK  %%BUFBIND_IN_MASK%%
#define BUFBIND_OUT_MASK %%BUFBIND_OUT_MASK%%

#define FILTER_SIZE %%FILTER_SIZE%%

layout(std430, binding = BUFBIND_IN_MASK) buffer buf_in_mask
{
    uint in_mask[];
};

layout(std430, binding = BUFBIND_OUT_MASK) buffer buf_out_mask
{
    uint out_mask[];
};

uniform uint width;
uniform uint height;
uniform uint mask_offset = 0;
// const uint mask_offset = 0;
const uint filter_size = FILTER_SIZE;

uvec2 get_pixel(uint linear_idx)
{
    uint x = uint(mod(linear_idx, width));
    uint y = uint(linear_idx / width);
    return uvec2(x, y);
}

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;

    uvec2 pixel = get_pixel(global_idx);

    for (int dx = -filter_size; dx <= filter_size; ++dx)
    {
        if (pixel.x + dx < 0) continue;
        if (pixel.x + dx >= width) continue;
        for (int dy = -filter_size; dy <= filter_size; ++dy)
        {
            if (pixel.y + dy < 0) continue;
            if (pixel.y + dy >= height) continue;
            uint mask_value = in_mask[mask_offset + global_idx + dx + dy*width];
            if (mask_value == 0) 
            {
                out_mask[mask_offset + global_idx] = 0; 
                return;
            }

        }
    }
    out_mask[mask_offset + global_idx] = 0;
}