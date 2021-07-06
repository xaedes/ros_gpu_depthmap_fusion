#version 430 core

// voxel_grid_occupancy_of_points

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_IN_COORDS  %%BUFBIND_IN_COORDS%%
#define BUFBIND_OUT_OCCUPANCY %%BUFBIND_OUT_OCCUPANCY%%

layout (std430, binding = BUFBIND_IN_COORDS) buffer buf_in_coords
{
    uint in_coords[];
};

layout (std430, binding = BUFBIND_OUT_OCCUPANCY) buffer buf_out_occupancy
{
    uint out_occupancy[];
};

uniform uint num_items = 0;
uniform uint in_coord_offset = 0;
uniform uint out_occupancy_offset = 0;
uniform uint occupied_value = 1;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;
    uint coord = in_coords[in_coord_offset + global_idx];
    out_occupancy[coord] = occupied_value;
}
