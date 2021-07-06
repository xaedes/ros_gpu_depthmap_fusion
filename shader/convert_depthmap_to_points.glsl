#version 430 core

// convert_depthmap_to_points

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

#define BUFBIND_DEPTH_PAIRS %%BUFBIND_DEPTH_PAIRS%%
#define BUFBIND_MASK        %%BUFBIND_MASK%%
#define BUFBIND_OUT_POINTSA  %%BUFBIND_OUT_POINTSA%%
#define BUFBIND_OUT_POINTSB  %%BUFBIND_OUT_POINTSB%%
#define BUFBIND_OUT_POINTSC  %%BUFBIND_OUT_POINTSC%%
#define BUFBIND_RECTIFY_MAP %%BUFBIND_RECTIFY_MAP%%

layout(std430, binding = BUFBIND_DEPTH_PAIRS) buffer buf_depth_pairs
{
    uint depth_pairs[];
};

layout(std430, binding = BUFBIND_MASK) buffer buf_mask
{
    uint mask[];
};

layout(std430, binding = BUFBIND_OUT_POINTSA) buffer buf_out_pointsa
{
    vec4 out_pointsa[];
};

layout(std430, binding = BUFBIND_OUT_POINTSB) buffer buf_out_pointsb
{
    vec4 out_pointsb[];
};

layout(std430, binding = BUFBIND_OUT_POINTSC) buffer buf_out_pointsc
{
    vec4 out_pointsc[];
};

layout(std430, binding = BUFBIND_RECTIFY_MAP) buffer buf_rectify_map
{
    vec2 rectify_map[];
};

uniform uint width;
uniform float depth_scale;
uniform float fx;
uniform float fy;
uniform float cx;
uniform float cy;

uniform uint depth_pair_offset;
uniform uint point_offset;
uniform uint mask_offset;
uniform uint rectify_offset;
uniform uint num_items;
uniform mat4 transform_world;
uniform mat4 transform_crop;

vec4 depthToPoint(float u, float v, uint depth)
{
    float z = depth*depth_scale;
    //z = 1;
    float x = (u-cx)/fx;
    float y = (v-cy)/fy;

    vec4 point_cam = vec4(x*z, y*z, z, 1);
    return point_cam;
}

vec2 rectify(uint linear_idx)
{
    float x = float(mod(linear_idx, width));
    float y = float(floor(linear_idx / width));
    return vec2(x,y);
    //return rectify_map[rectify_offset + int(x+0.5) + int(y+0.5) * width];
}

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
    // one input depth item of type uint spans two output points
    uint depth_pair_idx = point_idx/2;
    // extract depth
    // int which_point = int(mod(global_idx,2));
    // which_point = 0;
    int which_point = int(point_idx - depth_pair_idx*2);
    uint depth_pair = depth_pairs[depth_pair_offset + depth_pair_idx];
    uint depth = bitfieldExtract(depth_pair, 16*which_point, 16);
    if (depth == 0)
    {
        mask[mask_offset + global_idx] = 0;
        out_pointsa[point_offset + point_idx] = vec4(0,0,0,0);
        out_pointsb[point_offset + point_idx] = vec4(0,0,0,0);
    }
    else
    {
        // rectify pixel coordinate
        vec2 rectified = rectify(point_idx);
        // if (depth == 0) depth = 100;
        vec4 point = depthToPoint(rectified.x, rectified.y, depth);
        // output point
        mask[mask_offset + global_idx] = 1;
        out_pointsa[point_offset + point_idx] = point;
        out_pointsb[point_offset + point_idx] = point * transform_world;
        out_pointsc[point_offset + point_idx] = point * transform_crop;
    }

}
