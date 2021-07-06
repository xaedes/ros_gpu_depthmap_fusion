#version 430 core

layout(std430, binding = 0) buffer depths_buf
{
    uint depths[];
};

layout(std430, binding = 1) buffer points_buf
{
    vec4 points[];
};

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

uniform float depthScale = 0.001;
uniform int width = 400;
uniform float fx = 1;
uniform float fy = 1;
uniform float cx = 0;
uniform float cy = 0;

vec4 depthToPoint(uint depth)
{
    float u = gl_GlobalInvocationID.x;
    float v = gl_GlobalInvocationID.y;
    float z = depth*depthScale;
    //z = 1;
    float x = (u-cx)/fx;
    float y = (v-cy)/fy;
    // to realsense_depth_frame, with x in view direction and z up
    return vec4(z,-x*z,-y*z,1);
    // return vec4(z,-x*z,-y*z,1);
}
void writePoint(vec4 point, int k)
{
    uint point_idx = gl_GlobalInvocationID.x*2+k+gl_GlobalInvocationID.y*width;
    points[point_idx] = point;
}
void main()
{
    uint global_idx_out = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*width;
    uint global_idx_in = global_idx_out/2;
    int k = int(mod(global_idx_out,2));
    uint depth = bitfieldExtract(depths[global_idx_in], 16*k, 16);
    // points[gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*width] = vec4(1,1,1,1);
    // points[global_idx_out] = vec4(1,1,1,1);
    vec4 point = depthToPoint(depth);

    points[global_idx_out] = point;
}
