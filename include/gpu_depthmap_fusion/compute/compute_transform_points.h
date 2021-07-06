#pragma once
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeTransformPoints : public ComputeProgram
{
public:
    virtual ~ComputeTransformPoints() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_in_points, 
        uint bufbind_mask, 
        uint bufbind_out_points)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "transform_points.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_IN_POINTS",  std::to_string(bufbind_in_points)},
            {"BUFBIND_MASK",       std::to_string(bufbind_mask)},
            {"BUFBIND_OUT_POINTS", std::to_string(bufbind_out_points)}
        });
        num_items.init(m_glProgram, "num_items");
        in_point_offset.init(m_glProgram, "in_points_offset");
        out_point_offset.init(m_glProgram, "out_points_offset");
        mask_offset.init(m_glProgram, "mask_offset");
        transform.init(m_glProgram, "transform");
    }

    ProgramUniform<uint> num_items;
    ProgramUniform<uint> in_point_offset;
    ProgramUniform<uint> out_point_offset;
    ProgramUniform<uint> mask_offset;
    ProgramUniform<cv::Matx44f> transform;

    virtual ComputeProgram& dispatch()
    {
        dispatch(num_items.get());
        return *this;
    }
    virtual ComputeProgram& dispatch(int count)
    {
        dispatch(count, 1, 1);
        return *this;
    }
    virtual ComputeProgram& dispatch(int x, int y, int z)
    {
        ComputeProgram::dispatch(x,y,z,m_groupsize_x,m_groupsize_y,m_groupsize_z);
        return *this;
    }
protected:
    uint m_groupsize_x;
    uint m_groupsize_y;
    uint m_groupsize_z;
};