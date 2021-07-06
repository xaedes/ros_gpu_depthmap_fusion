#version 430

// compute_voxel_coords

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_POINTS     %%BUFBIND_POINTS%%
#define BUFBIND_OUT_VOXEL_COORDS %%BUFBIND_OUT_VOXEL_COORDS%%

layout (std430, binding = BUFBIND_POINTS) buffer buf_points
{
    vec4 points[];
};

layout (std430, binding = BUFBIND_OUT_VOXEL_COORDS) buffer buf_out_coords
{
    uint out_coords[];
};

uniform uint num_items = 0;
uniform uint in_point_offset = 0;
uniform uint out_coord_offset = 0;

uniform vec3 lower_bound;
uniform vec3 upper_bound;
uniform vec3 cell_size;
uniform uvec3 grid_size;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;

    vec3 point = points[in_point_offset+global_idx].xyz;
    vec3 f_coord = vec3(
        clamp((point.x - lower_bound.x) / cell_size.x, 0, grid_size.x-1),
        clamp((point.y - lower_bound.y) / cell_size.y, 0, grid_size.y-1),
        clamp((point.z - lower_bound.z) / cell_size.z, 0, grid_size.z-1));
    uvec3 u_coord = uvec3(
        uint(floor(f_coord.x)),
        uint(floor(f_coord.y)),
        uint(floor(f_coord.z))
    );
    uint coord = u_coord.x + (u_coord.y * grid_size.x) + (u_coord.z * grid_size.x * grid_size.y);
    out_coords[out_coord_offset+global_idx] = coord;
}
