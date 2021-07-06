#version 430

// transfer_masked_data

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define DATA_TYPE          %%DATA_TYPE%%

#define BUFBIND_IN_DATA    %%BUFBIND_IN_DATA%%
#define BUFBIND_OUT_Data   %%BUFBIND_OUT_Data%%
#define BUFBIND_IN_MASK    %%BUFBIND_IN_MASK%%
#define BUFBIND_SCRATCHPAD %%BUFBIND_SCRATCHPAD%%

layout (std430, binding = BUFBIND_IN_DATA) buffer buf_in_data
{
    DATA_TYPE in_data[];
};

layout (std430, binding = BUFBIND_OUT_Data) buffer buf_out_data
{
    DATA_TYPE out_data[];
};

layout (std430, binding = BUFBIND_IN_MASK) buffer buf_in_mask
{
    uint in_mask[];
};

layout (std430, binding = BUFBIND_SCRATCHPAD) buffer buf_scratchpad
{
    uint scratchpad[];
};

uniform uint num_items = 0;
uniform uint in_data_offset = 0;
uniform uint out_data_offset = 0;
uniform uint in_mask_offset = 0;
uniform uint scratchpad_offset = 0;
uniform uint num_transfer_per_mask_item = 1;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;
    if (in_mask[in_mask_offset+global_idx] > 0)
    {
        uint target_idx = atomicAdd(scratchpad[scratchpad_offset], num_transfer_per_mask_item);
        uint data_idx = global_idx * num_transfer_per_mask_item;
        for (int i= 0; i < num_transfer_per_mask_item; ++i)
        {
            out_data[out_data_offset+target_idx+i] = in_data[in_data_offset+data_idx+i];
        }
    }
}
