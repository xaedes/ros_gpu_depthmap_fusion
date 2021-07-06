#version 430 core

// rollbuffer_count_discarded_seqs

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_SEQUENCES  %%BUFBIND_IN_SEQUENCES%%
#define BUFBIND_SCRATCHPAD %%BUFBIND_SCRATCHPAD%%

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

layout (std430, binding = BUFBIND_SCRATCHPAD) buffer buf_scratchpad
{
    uint scratchpad[];
};


uniform uint num_items = 0;
uniform uint min_timestamp_sec = 0;
uniform uint min_timestamp_nsec = 0;
uniform uint in_seqs_offset = 0;
uniform uint scratchpad_offset = 0;

int compareTime(uint timeSecA, uint timeNSecA, uint timeSecB, uint timeNSecB)
{
    if (timeSecA < timeSecB)        return -1;
    else if (timeSecA > timeSecB)   return +1;
    else if (timeNSecA < timeNSecB) return -1;
    else if (timeNSecA > timeNSecB) return +1;
    else return 0;
}

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;

    scratchpad[scratchpad_offset] = 0;
    barrier();
    if (compareTime(
            in_sequences[in_seqs_offset + global_idx].timestampSec,
            in_sequences[in_seqs_offset + global_idx].timestampNSec,
            min_timestamp_sec,
            min_timestamp_nsec) < 0)
    {
        atomicAdd(scratchpad[scratchpad_offset], 1);
    }    
    barrier();
}
