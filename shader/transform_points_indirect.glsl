#version 430 core

// transform_points_indirect

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_TRANSFORMS        %%BUFBIND_IN_TRANSFORMS%%
#define BUFBIND_IN_TRANSFORM_INDICES %%BUFBIND_IN_TRANSFORM_INDICES%%
#define BUFBIND_IN_POINTS            %%BUFBIND_IN_POINTS%%
#define BUFBIND_IN_MASK              %%BUFBIND_IN_MASK%%
#define BUFBIND_OUT_POINTS           %%BUFBIND_OUT_POINTS%%

layout(std430, binding = BUFBIND_IN_TRANSFORMS) buffer buf_in_transforms
{
    mat4 in_transforms[];
};

layout(std430, binding = BUFBIND_IN_TRANSFORM_INDICES) buffer buf_in_transform_indices
{
    uint in_transform_indices[];
};

layout(std430, binding = BUFBIND_IN_POINTS) buffer buf_in_points
{
    vec4 in_points[];
};

layout(std430, binding = BUFBIND_IN_MASK) buffer buf_in_mask
{
    uint in_mask[];
};

layout(std430, binding = BUFBIND_OUT_POINTS) buffer buf_out_points
{
    vec4 out_points[];
};

uniform uint num_items;
uniform uint in_transforms_offset;
uniform uint in_transform_indices_offset;
uniform uint in_points_offset;
uniform uint in_mask_offset;
uniform uint out_points_offset;

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
    uint mask_value = in_mask[in_mask_offset + point_idx];
    if (mask_value != 0)
    {
        vec4 point = in_points[in_points_offset + point_idx];
        uint tf_idx = in_transform_indices[in_transform_indices_offset + point_idx];
        mat4 transform = in_transforms[in_transforms_offset + tf_idx];
        out_points[out_points_offset + point_idx] = point * transform;
    }
}
