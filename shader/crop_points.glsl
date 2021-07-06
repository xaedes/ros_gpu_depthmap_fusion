#version 430 core

// crop_points

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_MASK  %%BUFBIND_IN_MASK%%
#define BUFBIND_OUT_MASK %%BUFBIND_OUT_MASK%%
#define BUFBIND_POINTS   %%BUFBIND_POINTS%%

layout(std430, binding = BUFBIND_IN_MASK) buffer buf_in_mask
{
    uint in_mask[];
};

layout(std430, binding = BUFBIND_OUT_MASK) buffer buf_out_mask
{
    uint out_mask[];
};

layout(std430, binding = BUFBIND_POINTS) buffer buf_points
{
    vec4 points[];
};

uniform uint num_items = 0;
uniform uint point_offset = 0;
uniform uint mask_offset = 0;

uniform vec3 lower_bound;
uniform vec3 upper_bound;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;

    uint mask_value = in_mask[mask_offset + global_idx];
    // out_mask[mask_offset + global_idx] = mask_value;
    if (mask_value == 0)
    {
        out_mask[mask_offset + global_idx] = 0;
    }
    else
    {
        vec4 point = points[point_offset + global_idx];
        if (point.x < lower_bound.x || point.x > upper_bound.x ||
            point.y < lower_bound.y || point.y > upper_bound.y ||
            point.z < lower_bound.z || point.z > upper_bound.z)
        {
            out_mask[mask_offset + global_idx] = 0;
        }
        else
        {
            out_mask[mask_offset + global_idx] = mask_value;
        }
    }

}
