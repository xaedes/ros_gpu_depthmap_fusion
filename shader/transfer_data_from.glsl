#version 430

// transfer_data_From

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define DATA_TYPE          %%DATA_TYPE%%

#define BUFBIND_IN_INDICES %%BUFBIND_IN_INDICES%%
#define BUFBIND_IN_DATA    %%BUFBIND_IN_DATA%%
#define BUFBIND_OUT_DATA   %%BUFBIND_OUT_DATA%%

layout (std430, binding = BUFBIND_IN_INDICES) buffer buf_in_indices
{
    uint in_indices[];
};

%%DATA_TYPE_DECLARATION%%

layout (std430, binding = BUFBIND_IN_DATA) buffer buf_in_data
{
    DATA_TYPE in_data[];
};

layout (std430, binding = BUFBIND_OUT_DATA) buffer buf_out_data
{
    DATA_TYPE out_data[];
};

uniform uint num_items = 0;
uniform uint in_indices_offset = 0;
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

    uint in_data_idx = in_indices[in_indices_offset + global_idx];
    out_data[out_data_offset + global_idx] = in_data[in_data_offset + in_data_idx];
}
