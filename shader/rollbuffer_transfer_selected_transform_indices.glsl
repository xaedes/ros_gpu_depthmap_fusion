#version 430 core

// rollbuffer_transfer_selected_transform_indices

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_SEQ_IDCS %%BUFBIND_IN_SEQ_IDCS%%
#define BUFBIND_OUT_TF_IDCS %%BUFBIND_OUT_TF_IDCS%%

layout (std430, binding = BUFBIND_IN_SEQ_IDCS) buffer buf_in_seq_idcs
{
    uint in_seq_idcs[];
};

layout (std430, binding = BUFBIND_OUT_TF_IDCS) buffer buf_out_tf_idcs
{
    uint out_tf_idcs[];
};


uniform uint num_items = 0;
uniform uint in_seq_idcs_offset = 0;
uniform uint out_tf_idcs_offset = 0;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;

    uint seq_idx0 = in_seq_idcs[in_seq_idcs_offset];
    uint seq_idx = in_seq_idcs[in_seq_idcs_offset + global_idx];
    out_tf_idcs[out_tf_idcs_offset + global_idx] = seq_idx - seq_idx0;
}
