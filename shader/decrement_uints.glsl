#version 430 core

// decrement_uints
// out = max(min_value, in - decrement)

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN %%BUFBIND_IN%%
#define BUFBIND_OUT %%BUFBIND_OUT%%

layout (std430, binding = BUFBIND_IN) buffer buf_in
{
    uint data_in[];
};

layout (std430, binding = BUFBIND_OUT) buffer buf_out
{
    uint data_out[];
};

uniform uint num_items = 0;
uniform uint offset = 0;
uniform uint decrement = 0;
uniform uint min_value = 0;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;
    uint idx = offset+global_idx;
    uint value_in = data_in[idx];

    // avoid uint underflow
    if (value_in >= min_value + decrement)
    {
        data_out[idx] = value_in - decrement;
    }
    else
    {
        data_out[idx] = min_value;
    }
}
