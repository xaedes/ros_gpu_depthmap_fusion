#version 430

// apply_point_mask

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_POINTS     %%BUFBIND_POINTS%%
#define BUFBIND_OUT_POINTS %%BUFBIND_OUT_POINTS%%
#define BUFBIND_MASK       %%BUFBIND_MASK%%
#define BUFBIND_SCRATCHPAD %%BUFBIND_SCRATCHPAD%%

layout (std430, binding = BUFBIND_POINTS) buffer buf_points
{
    vec4 points[];
};

layout (std430, binding = BUFBIND_OUT_POINTS) buffer buf_out_points
{
    vec4 out_points[];
};

layout (std430, binding = BUFBIND_MASK) buffer buf_mask
{
    uint mask[];
};

layout (std430, binding = BUFBIND_SCRATCHPAD) buffer buf_scratchpad
{
    uint scratchpad[];
};

uniform uint num_items = 0;
uniform uint in_point_offset = 0;
uniform uint out_point_offset = 0;
uniform uint mask_offset = 0;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;
    if (mask[mask_offset+global_idx] > 0)
    {
        uint target_idx = atomicAdd(scratchpad[0],1);
        out_points[out_point_offset+target_idx] = points[in_point_offset+global_idx];
    }
}
