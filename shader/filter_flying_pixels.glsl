#version 430 core

// filter_flying_pixels

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
uniform uint width = 1;
uniform uint height = 1;
uniform uint point_offset = 0;
uniform uint mask_offset = 0;

uniform uint filter_size = 1;

uniform float threshold_view_angle;
uniform uint enable_rot45 = 0;
uniform float max_distance = 10;

uvec2 get_pixel(uint linear_idx)
{
    uint x = uint(mod(linear_idx, width));
    uint y = uint(linear_idx / width);
    return uvec2(x, y);
}

void invalidate_point(uint global_idx)
{
    out_mask[mask_offset + global_idx] = 0;
}

bool check_at(uint global_idx, uint du, uint dv, uint offsetx, uint offsety)
{
    uint idx = global_idx + offsetx + offsety*width;
    uvec2 pixel = get_pixel(idx);

    if ((pixel.x-du < 0) || (pixel.x+du > width-1) || (pixel.y-dv < 0) || (pixel.y+dv > height-1)) 
    { invalidate_point(global_idx); return false; }

    if (in_mask[mask_offset + idx ] == 0)           { invalidate_point(global_idx); return false; }
    if (in_mask[mask_offset + idx - dv*width] == 0) { invalidate_point(global_idx); return false; }
    if (in_mask[mask_offset + idx + dv*width] == 0) { invalidate_point(global_idx); return false; }
    if (in_mask[mask_offset + idx - du] == 0)       { invalidate_point(global_idx); return false; }
    if (in_mask[mask_offset + idx + du] == 0)       { invalidate_point(global_idx); return false; }

    vec4 point       = points[point_offset + idx];
    vec4 point_up    = points[point_offset + idx - dv*width];
    vec4 point_down  = points[point_offset + idx + dv*width];
    vec4 point_left  = points[point_offset + idx - du];
    vec4 point_right = points[point_offset + idx + du];

    // dx: from left to right 
    vec4 dx = point_right - point_left;
    // dy: from up to down 
    vec4 dy = point_down  - point_up;

    // normal pointing from surface 
    vec3 normal = normalize(cross(dy.xyz,dx.xyz));

    // angle between normal vector and vector pointing from point to camera at (0,0,0)
    float cos_view_angle = dot(normal, -normalize(point.xyz));

    if (cos_view_angle < threshold_view_angle) { invalidate_point(global_idx); return false; }

    return true;
}

bool check_at_rot45(uint global_idx, uint du, uint dv, uint offsetx, uint offsety)
{
    uint idx = global_idx + offsetx + offsety*width;
    uvec2 pixel = get_pixel(idx);

    if ((pixel.x-du < 0) || (pixel.x+du > width-1) || (pixel.y-dv < 0) || (pixel.y+dv > height-1)) 
    { invalidate_point(global_idx); return false; }

    if (in_mask[mask_offset + idx ] == 0)           { invalidate_point(global_idx); return false; }
    if (in_mask[mask_offset + idx - dv*width - du] == 0) { invalidate_point(global_idx); return false; }
    if (in_mask[mask_offset + idx - dv*width + du] == 0) { invalidate_point(global_idx); return false; }
    if (in_mask[mask_offset + idx + dv*width - du] == 0) { invalidate_point(global_idx); return false; }
    if (in_mask[mask_offset + idx + dv*width + du] == 0) { invalidate_point(global_idx); return false; }

    vec4 point       = points[point_offset + idx];
    vec4 point_up    = points[point_offset + idx - dv*width - du];
    vec4 point_down  = points[point_offset + idx + dv*width + du];
    vec4 point_left  = points[point_offset + idx + dv*width - du];
    vec4 point_right = points[point_offset + idx - dv*width + du];

    // dx: from left to right 
    vec4 dx = point_right - point_left;
    // dy: from up to down 
    vec4 dy = point_down  - point_up;

    // normal pointing from surface 
    vec3 normal = normalize(cross(dy.xyz,dx.xyz));

    // angle between normal vector and vector pointing from point to camera at (0,0,0)
    float view_angle = dot(normal, -normalize(point.xyz));

    if (view_angle < threshold_view_angle) { invalidate_point(global_idx); return false; }

    return true;
}

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;
    uint mask_value = in_mask[mask_offset + global_idx];
    
    // invalidate_point(global_idx); return;
    
    if (mask_value == 0) { invalidate_point(global_idx); return; }
    out_mask[mask_offset + global_idx] = mask_value;

    vec3 point = points[point_offset + global_idx].xyz;
    if (length(point) > max_distance) { invalidate_point(global_idx); return; }

    for (uint i = 0; i < filter_size; ++i)
    {
        check_at(global_idx, i+1, i+1, 0, 0);
        if (enable_rot45 != 0) check_at_rot45(global_idx, i+1, i+1, 0, 0);
        // check_at(global_idx, 1, 1, 0, 0);
        // for (int j = 0; j <= i; ++j)
        // {
        //     // check_at(global_idx, 1, 1, -(1+j), 0);
        //     // check_at(global_idx, 1, 1, +(1+j), 0);
        //     // check_at(global_idx, 1, 1, 0, -(1+j));
        //     // check_at(global_idx, 1, 1, 0, +(1+j));
        //     // check_at(global_idx, j+1, j+1, -(1+j), 0);
        //     // check_at(global_idx, j+1, j+1, +(1+j), 0);
        //     // check_at(global_idx, j+1, j+1, 0, -(1+j));
        //     // check_at(global_idx, j+1, j+1, 0, +(1+j));
        //     check_at(global_idx, i+1, i+1, -(1+j), 0);
        //     check_at(global_idx, i+1, i+1, +(1+j), 0);
        //     check_at(global_idx, i+1, i+1, 0, -(1+j));
        //     check_at(global_idx, i+1, i+1, 0, +(1+j));
        // }
    }
    // if (enable_rot45 != 0)
    // {
    //     for (uint i = 0; i < filter_size; ++i)
    //     {
    //         check_at_rot45(global_idx, i+1, i+1, 0, 0);
    //         // for (int j = 0; j <= i; ++j)
    //         // {
    //         //     // check_at_rot45(global_idx, i, i, -(1+j), 0);
    //         //     // check_at_rot45(global_idx, i, i, +(1+j), 0);
    //         //     // check_at_rot45(global_idx, i, i, 0, -(1+j));
    //         //     // check_at_rot45(global_idx, i, i, 0, +(1+j));
    //         //     check_at_rot45(global_idx, i+1, i+1, -(1+j), 0);
    //         //     check_at_rot45(global_idx, i+1, i+1, +(1+j), 0);
    //         //     check_at_rot45(global_idx, i+1, i+1, 0, -(1+j));
    //         //     check_at_rot45(global_idx, i+1, i+1, 0, +(1+j));
    //         // }
    //     }
    // }
    // check_at(global_idx, 2, 2, 0, 0);
    // check_at(global_idx, 3, 3, 0, 0);
    // check_at(global_idx, 4, 4, 0, 0);
    
    // for (uint offsety = -3; offsety <= +3; offsety++) 
    // {
    //     for (uint offsetx = -3; offsetx <= +3; offsetx++) 
    //     {
    //         if (!check_at(global_idx, 1, 1, offsetx, offsety)) return;
    //         if (!check_at(global_idx, 2, 2, offsetx, offsety)) return;
    //     }
    // }
    // check_at(global_idx, 3, 3);
}
