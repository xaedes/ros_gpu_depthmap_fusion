#version 430 core

// vec4_to_vec3

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_DATA %%BUFBIND_IN_DATA%%
#define BUFBIND_OUT_DATA %%BUFBIND_OUT_DATA%%

layout (std430, binding = BUFBIND_IN_DATA) buffer buf_in_data
{
    vec4 in_data[];
};

layout (std430, binding = BUFBIND_OUT_DATA) buffer buf_out_data
{
    vec3 out_data[];
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
    uint in_idx = global_idx;

    vec4 item = in_data[in_data_offset + in_idx];
    out_data[out_data_offset + out_idx] = item.xyz;
}
