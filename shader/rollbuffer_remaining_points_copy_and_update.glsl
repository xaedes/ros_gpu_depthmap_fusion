#version 430 core

// rollbuffer_remaining_points_copy_and_update

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_INDICES   %%BUFBIND_IN_INDICES%%
#define BUFBIND_IN_POINTS    %%BUFBIND_IN_POINTS%%
#define BUFBIND_IN_SEQ_IDCS  %%BUFBIND_IN_SEQ_IDCS%%
#define BUFBIND_IN_MASK      %%BUFBIND_IN_MASK%%
#define BUFBIND_OUT_POINTS   %%BUFBIND_OUT_POINTS%%
#define BUFBIND_OUT_SEQ_IDCS %%BUFBIND_OUT_SEQ_IDCS%%
#define BUFBIND_OUT_MASK     %%BUFBIND_OUT_MASK%%

layout (std430, binding = BUFBIND_IN_INDICES) buffer buf_in_indices
{
    uint in_indices[];
};

layout (std430, binding = BUFBIND_IN_POINTS) buffer buf_in_points
{
    vec4 in_points[];
};

layout (std430, binding = BUFBIND_IN_SEQ_IDCS) buffer buf_in_seq_idcs
{
    uint in_seq_idcs[];
};

layout (std430, binding = BUFBIND_IN_MASK) buffer buf_in_mask
{
    uint in_mask[];
};

layout (std430, binding = BUFBIND_OUT_POINTS) buffer buf_out_points
{
    vec4 out_points[];
};

layout (std430, binding = BUFBIND_OUT_SEQ_IDCS) buffer buf_out_seq_idcs
{
    uint out_seq_idcs[];
};

layout (std430, binding = BUFBIND_OUT_MASK) buffer buf_out_mask
{
    uint out_mask[];
};

uniform uint num_items = 0;
uniform uint num_discarded_seqs = 0;
uniform uint in_remaining_idcs_offset = 0;
uniform uint in_points_offset = 0;
uniform uint in_seq_idcs_offset = 0;
uniform uint in_mask_offset = 0;
uniform uint out_points_offset = 0;
uniform uint out_seq_idcs_offset = 0;
uniform uint out_mask_offset = 0;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;

    uint idx = in_indices[in_remaining_idcs_offset + global_idx];

    out_points[out_points_offset + global_idx] = in_points[in_points_offset + idx];
    out_seq_idcs[out_seq_idcs_offset + global_idx] = in_seq_idcs[in_points_offset + idx] - num_discarded_seqs;
    out_mask[out_mask_offset + global_idx] = in_mask[in_mask_offset + idx];
}
