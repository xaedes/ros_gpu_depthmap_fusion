#version 430 core

// rollbuffer_select_timespan_points

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_SELECTED     %%BUFBIND_SELECTED%%
#define BUFBIND_IN_SEQ_IDCS  %%BUFBIND_IN_SEQ_IDCS%%
#define BUFBIND_IN_SEQUENCES %%BUFBIND_IN_SEQUENCES%%

struct PointSequence
{
    uint timestampSec;
    uint timestampNSec;
    uint start;
    uint numPoints;
    uint padding[12];
    mat4 transform_move; 
};

layout (std430, binding = BUFBIND_SELECTED) buffer buf_selected
{
    uint selected[];
};

layout (std430, binding = BUFBIND_IN_SEQ_IDCS) buffer buf_in_seq_idcs
{
    uint in_seq_idcs[];
};

layout (std430, binding = BUFBIND_IN_SEQUENCES) buffer buf_in_sequences
{
    PointSequence in_sequences[];
};


uniform uint num_items = 0;
uniform uint min_timestamp_sec = 0;
uniform uint min_timestamp_nsec = 0;
uniform uint max_timestamp_sec = 0;
uniform uint max_timestamp_nsec = 0;
uniform uint in_seq_idcs_offset = 0;
uniform uint in_sequences_offset = 0;
uniform uint selection_offset = 0;

int compareTime(uint timeSecA, uint timeNSecA, uint timeSecB, uint timeNSecB)
{
    if (timeSecA < timeSecB)        return -1;
    else if (timeSecA > timeSecB)   return +1;
    else if (timeNSecA < timeNSecB) return -1;
    else if (timeNSecA > timeNSecB) return +1;
    return 0;
}

void main()
{
    selected[selection_offset+0] = num_items;
    selected[selection_offset+1] = 0;
    
    barrier();

    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;

    uint seq_idx = in_seq_idcs[in_seq_idcs_offset + global_idx];

    uint t_sec = in_sequences[in_sequences_offset + seq_idx].timestampSec;
    uint t_nsec = in_sequences[in_sequences_offset + seq_idx].timestampNSec;
    int cmpMinAndTime = compareTime(
        min_timestamp_sec,
        min_timestamp_nsec,
        t_sec,
        t_nsec
    );
    int cmpTimeAndMax = compareTime(
        t_sec,
        t_nsec,
        max_timestamp_sec,
        max_timestamp_nsec
    );
    if ((cmpMinAndTime <= 0) && (cmpTimeAndMax <= 0))
    {
        atomicMin(selected[selection_offset+0], global_idx);
        atomicMax(selected[selection_offset+1], global_idx);
    }
    
    barrier();
}
