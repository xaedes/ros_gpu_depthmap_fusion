#version 430 core

// transform_points

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_POINTS  %%BUFBIND_IN_POINTS%%
#define BUFBIND_MASK       %%BUFBIND_MASK%%
#define BUFBIND_OUT_POINTS %%BUFBIND_OUT_POINTS%%

layout(std430, binding = BUFBIND_IN_POINTS) buffer buf_in_points
{
    vec4 in_points[];
};

layout(std430, binding = BUFBIND_MASK) buffer buf_mask
{
    uint mask[];
};

layout(std430, binding = BUFBIND_OUT_POINTS) buffer buf_out_pointsa
{
    vec4 out_points[];
};

uniform uint in_points_offset;
uniform uint mask_offset;
uniform uint out_points_offset;
uniform uint num_items;
uniform mat4 transform;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;

    // one work thread for every output point
    uint point_idx = global_idx;
    uint mask_value = mask[mask_offset + point_idx];
    if (mask_value != 0)
    {
        vec4 point = in_points[in_points_offset + point_idx];
        out_points[out_points_offset + point_idx] = point * transform;
    }
}
