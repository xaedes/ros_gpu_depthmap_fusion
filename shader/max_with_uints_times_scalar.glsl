#version 430 core

// max_with_uints_times_scalar

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_A %%BUFBIND_IN_A%%
#define BUFBIND_IN_B %%BUFBIND_IN_B%%
#define BUFBIND_OUT %%BUFBIND_OUT%%

layout (std430, binding = BUFBIND_IN_A) buffer buf_in_a
{
    uint in_a[];
};

layout (std430, binding = BUFBIND_IN_B) buffer buf_in_b
{
    uint in_b[];
};

layout (std430, binding = BUFBIND_OUT) buffer buf_out
{
    uint data[];
};

uniform uint num_items = 0;
uniform uint offset = 0;
uniform uint multiplier = 0;

// out = max(in_a, in_b * multiplier)
void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;
    uint idx = offset+global_idx;
    data[idx] = max(in_a[idx], in_b[idx] * multiplier);
}
