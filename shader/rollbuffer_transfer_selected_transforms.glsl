#version 430 core

// rollbuffer_transfer_selected_transforms

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_SEQUENCES  %%BUFBIND_IN_SEQUENCES%%
#define BUFBIND_OUT_TFS_WORLD %%BUFBIND_OUT_TFS_WORLD%%
#define BUFBIND_OUT_TFS_CROP  %%BUFBIND_OUT_TFS_CROP%%

struct PointSequence
{
    uint timestampSec;
    uint timestampNSec;
    uint start;
    uint numPoints;
    uint padding[12];
    mat4 transform_move; 
};

layout (std430, binding = BUFBIND_IN_SEQUENCES) buffer buf_in_sequences
{
    PointSequence in_sequences[];
};

layout (std430, binding = BUFBIND_OUT_TFS_WORLD) buffer buf_out_tfs_world
{
    mat4 out_tfs_world[];
};

layout (std430, binding = BUFBIND_OUT_TFS_CROP) buffer buf_out_tfs_crop
{
    mat4 out_tfs_crop[];
};


uniform uint num_items = 0;
uniform uint in_seqs_offset = 0;
uniform uint out_tfs_world_offset = 0;
uniform uint out_tfs_crop_offset = 0;
uniform mat4 tf_world_move;
uniform mat4 tf_crop_move;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;

    uint seq_idx = in_seqs_offset + global_idx;

    // TODO tf multiplication maybe in reverse order? because transform of point is point * transform

    // reverse order, because we store matrices as row-major and gl interpretes them as col-major
    out_tfs_world[out_tfs_world_offset + global_idx] = (in_sequences[seq_idx].transform_move) * tf_world_move;
    out_tfs_crop[out_tfs_crop_offset + global_idx] = (in_sequences[seq_idx].transform_move) * tf_crop_move;
    // (A*B)^T = B^T * A^T
}
