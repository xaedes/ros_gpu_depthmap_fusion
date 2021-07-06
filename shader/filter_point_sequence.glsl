#version 430 core

// filter_livox

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_MASK   %%BUFBIND_IN_MASK%%
#define BUFBIND_OUT_MASK  %%BUFBIND_OUT_MASK%%
#define BUFBIND_IN_POINTS %%BUFBIND_IN_POINTS%%

layout(std430, binding = BUFBIND_IN_MASK) buffer buf_in_mask
{
    uint in_mask[];
};

layout(std430, binding = BUFBIND_OUT_MASK) buffer buf_out_mask
{
    uint out_mask[];
};

layout(std430, binding = BUFBIND_IN_POINTS) buffer buf_in_points
{
    vec4 in_points[];
};

uniform uint num_items = 0;
uniform uint in_point_offset = 0;
uniform uint in_mask_offset = 0;
uniform uint out_mask_offset = 0;

uniform uint filter_size = 3;
uniform float threshold_view_angle = 0.4;

void invalidate_point(uint global_idx)
{
    out_mask[out_mask_offset + global_idx] = 0;
}

bool filter_flying_pixels(uint global_idx, vec4 point, uint other_idx)
{
    vec4 next_point = in_points[in_point_offset + other_idx];
    vec3 point_seq_direction = normalize(next_point.xyz - point.xyz);

    // angle between normal vector and vector pointing from point to camera at (0,0,0)
    float cos_between = abs(dot(point_seq_direction, -normalize(point.xyz)));
    // view_angle = cos(90-between)
    // cos(90 - between)  = sin(between)
    // cos(between)^2+sin(between)^2=1
    // sin(between) = sqrt(1-cos(between)^2)
    
    // float c2 = cos_between*cos_between;
    // if (c2 <= 1)
    // {
    //     float sin_between = sqrt(1-cos_between*cos_between);
    //     float view_angle = sin_between;
    //     if (view_angle > threshold_view_angle) { invalidate_point(global_idx); return false; }
    // }

    // if (cos_between > threshold_view_angle) 
    // if (cos_between > 0.4) 
    // if (-cos_between < -1+0.4) 
    // if (1-cos_between < 0.4) 
    if (1-cos_between < threshold_view_angle) { invalidate_point(global_idx); return false; }

    return true;
}

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;

    // exit if out of bounds
    if (global_idx >= num_items) return;
    
    // invalidate_point(global_idx); return;
    
    // invalidate if input mask is not set
    uint mask_value = in_mask[in_mask_offset + global_idx];
    if (mask_value == 0) { invalidate_point(global_idx); return; }

    // invalidate if point is close to zero
    vec4 point = in_points[in_point_offset + global_idx];
    if (length(point.xyz) < 1e-3) { invalidate_point(global_idx); return; }



    bool invalidated = false;
    // TODO: filter flying pixels
    for (uint i = 0; i < filter_size; ++i)
    {
        if ((0 <= global_idx + i - 1) && (global_idx + i - 1 < num_items))
        {
            invalidated = invalidated || !filter_flying_pixels(
                global_idx,
                point,
                global_idx + i - 1
            );
        }
        if ((0 <= global_idx + i + 1) && (global_idx + i + 1 < num_items))
        {
            invalidated = invalidated || !filter_flying_pixels(
                global_idx,
                point,
                global_idx + i + 1
            );
        }
    }
    if (!invalidated)
    {
        // write original mask value
        out_mask[out_mask_offset + global_idx] = mask_value;
    }
}
