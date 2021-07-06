#version 430 core

// rollbuffer_remaining_points_indices
// todo: refactor to uint_enumerate_from

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_OUT_INDICES     %%BUFBIND_OUT_INDICES%%

layout (std430, binding = BUFBIND_OUT_INDICES) buffer buf_out_indices
{
    uint out_indices[];
};


uniform uint num_items = 0;
uniform uint num_discarded_pts = 0;
uniform uint out_offset = 0;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;

    out_indices[out_offset + global_idx] = num_discarded_pts + global_idx;
}
