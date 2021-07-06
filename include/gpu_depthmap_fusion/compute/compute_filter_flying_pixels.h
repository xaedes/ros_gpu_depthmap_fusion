#pragma once
#include <vector>
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeFilterFlyingPixels : public ComputeProgram
{
public:
    virtual ~ComputeFilterFlyingPixels() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_in_mask, 
        uint bufbind_out_mask, 
        uint bufbind_points)
    {
        m_groupsize_x = 32;
        m_groupsize_y = 32;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "filter_flying_pixels.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_IN_MASK",  std::to_string(bufbind_in_mask)},
            {"BUFBIND_OUT_MASK", std::to_string(bufbind_out_mask)},
            {"BUFBIND_POINTS",   std::to_string(bufbind_points)}
        });
        width.init(m_glProgram, "width");
        height.init(m_glProgram, "height");
        num_items.init(m_glProgram, "num_items");
        point_offset.init(m_glProgram, "point_offset");
        mask_offset.init(m_glProgram, "mask_offset");
        threshold_view_angle.init(m_glProgram, "threshold_view_angle");
        enable_rot45.init(m_glProgram, "enable_rot45");
        this->filter_size.init(m_glProgram, "filter_size");
    }

    ProgramUniform<uint> width;
    ProgramUniform<uint> height;
    ProgramUniform<uint> num_items;
    ProgramUniform<uint> point_offset;
    ProgramUniform<uint> mask_offset;
    ProgramUniform<float> threshold_view_angle;
    ProgramUniform<uint> enable_rot45;
    ProgramUniform<uint> filter_size;
    virtual ComputeProgram& dispatch()
    {
        dispatch(width.get(),height.get(), 1);
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

