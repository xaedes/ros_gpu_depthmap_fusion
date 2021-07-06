#version 430 core

// uints_to_chars
// called for each output uint containing 4 chars, 
// reads those 4 char values from 4 seperate uints

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_DATA %%BUFBIND_IN_DATA%%
#define BUFBIND_OUT_DATA %%BUFBIND_OUT_DATA%%

layout (std430, binding = BUFBIND_IN_DATA) buffer buf_in_data
{
    uint in_data[];
};

layout (std430, binding = BUFBIND_OUT_DATA) buffer buf_out_data
{
    uint out_data[];
};

uniform uint num_items = 0;
uniform uint in_data_offset = 0;
uniform uint out_data_offset = 0;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;
    uint out_idx = global_idx;
    uint in_idx = global_idx * 4;

    uint byte_mask = uint(0xff);
    uint output_value = 
        (  (in_data[in_data_offset + in_idx + 0] & byte_mask) 
        | ((in_data[in_data_offset + in_idx + 1] & byte_mask) << 8)
        | ((in_data[in_data_offset + in_idx + 2] & byte_mask) << 16)
        | ((in_data[in_data_offset + in_idx + 3] & byte_mask) << 24));

    out_data[out_data_offset + out_idx] = output_value;
}
